"""Output dtype encoding.

The pipeline computes every algorithm in float32 (NaN = NoData).  For delivery
the result can optionally be quantized to a compact integer COG:

- ``int16`` / ``uint8`` shrink the file ~2x / ~4x, cutting COG-build disk I/O and
  the cost of moving the product to object storage / reading it in QGIS.
- Integer encodings reserve **0 for NoData**.  The data->DN mapping depends on
  whether the algorithm's range is signed (straddles 0) or unsigned:
    * unsigned (e.g. slope 0..90, AO 0..1): data occupies ``[1, MAX]``.
    * signed (e.g. RVI -3..3, LRM -1.5..1.5):
        - ``int16`` uses the full symmetric range ``[-MAX, +MAX]`` so both signs
          get maximum tonal resolution; ``DN = 0`` (== value ~0, i.e. flat ground)
          doubles as NoData -- visually negligible since it is the flat midtone.
        - ``uint8`` centres value 0 at ``128`` and spans ``[1, 255]`` (0 = NoData).
- GDAL ``scale``/``offset`` are recorded (``value = scale*DN + offset``) so the
  physical value is recoverable; for ``DN = 0`` (NoData) the value is undefined.

Compute stays float32 on the GPU; only the final encoding changes -- so the GPU
math and its accuracy are unaffected, and ``float32`` output is byte-for-byte the
previous behaviour.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Native output value range per algorithm (the span that maximises tonal range).
# Verified against each algorithm's final clip / normalization.
# Normalized algorithms map their robust p99 value to display magnitude 1.0
# (see ``algorithms/_normalization.NORMAL_PERCENTILE``); the float result is
# unclipped, so the rare tail runs a little past +/-1.  The integer ranges below
# are set to +/-_NORM_HEADROOM (1.176 = 1/0.85) so value +/-1 lands at ~0.85*MAXPOS
# and the tail up to +/-1.176 is encoded before clipping, preserving dynamic range.
_NORM_HEADROOM = 1.176
OUTPUT_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    # Signed, normalized to +/-1 with int headroom.
    "rvi": (-_NORM_HEADROOM, _NORM_HEADROOM),
    "lrm": (-_NORM_HEADROOM, _NORM_HEADROOM),
    "fractal_anomaly": (-_NORM_HEADROOM, _NORM_HEADROOM),
    # Unsigned [0, 1], physically bounded -- left at native range (no pre-stat).
    "hillshade": (0.0, 1.0),
    "specular": (0.0, 1.0),
    "atmospheric_scattering": (0.0, 1.0),
    "curvature": (0.0, 1.0),
    "ambient_occlusion": (0.0, 1.0),
    "openness": (0.0, 1.0),
    "multi_light_uncertainty": (0.0, 1.0),
    # Unsigned, normalized to [0,1] with int headroom.
    "multiscale_terrain": (0.0, _NORM_HEADROOM),
    "visual_saliency": (0.0, _NORM_HEADROOM),
    "scale_space_surprise": (0.0, _NORM_HEADROOM),
    # Stylized edges clip to [0.2, 1.0].
    "npr_edges": (0.2, 1.0),
    # Physical: slope in degrees spans [0, 90] (default unit).
    "slope": (0.0, 90.0),
}

SUPPORTED_OUTPUT_DTYPES = ("float32", "int16", "uint8")

# Largest positive code per integer dtype (0 is reserved for NoData).
_INT_MAXPOS: Dict[str, int] = {"int16": 32767, "uint8": 255}


def is_integer_output(dtype) -> bool:
    return str(dtype).lower() in _INT_MAXPOS


def output_nodata_for_dtype(dtype) -> float:
    """NoData sentinel for an output dtype: NaN for float, 0 for integers."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return float("nan")
    return 0.0


def resolve_output_range(
    algorithm: str,
    *,
    params: Optional[dict] = None,
    override: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """Resolve the float value range to stretch into the integer codes.

    Priority: explicit ``override`` -> per-algorithm registry (unit-aware for
    slope) -> ``None`` (caller estimates from data percentiles).
    """
    if override is not None:
        lo, hi = float(override[0]), float(override[1])
        if hi > lo:
            return lo, hi

    algo = str(algorithm).lower()
    if algo == "slope" and params is not None:
        unit = str(params.get("unit", "degree")).lower()
        if unit == "radian":
            return (0.0, float(np.pi / 2.0))
        if unit != "degree":
            return None  # 'percent' is unbounded -> estimate from data
    return OUTPUT_VALUE_RANGES.get(algo)


def quantize_params(lo: float, hi: float, dtype: str) -> Dict[str, float]:
    """Linear encoding params mapping a float value range to integer codes.

    ``DN = clip(round(a_coef*value + b_coef), dn_min, dn_max)`` with NaN/NoData->0;
    inversely ``value = scale*DN + offset`` (``scale = 1/a_coef``,
    ``offset = -b_coef/a_coef``), and NoData is always DN ``0``.

    Signed ranges (``lo < 0 < hi``) are encoded symmetrically about value 0 (see
    the module docstring); unsigned ranges fill ``[1, MAXPOS]``.
    """
    lo = float(lo)
    hi = float(hi)
    dt = str(dtype).lower()
    maxpos = _INT_MAXPOS[dt]
    signed = lo < 0.0 < hi

    if signed:
        a = max(abs(lo), abs(hi))  # symmetric half-range
        a = a if a > 0 else 1.0
        if dt == "int16":
            a_coef = maxpos / a            # value 0 -> DN 0 (== NoData)
            b_coef = 0.0
            dn_min, dn_max = -maxpos, maxpos
        else:  # uint8: centre value 0 at 128, data in [1, 255]
            a_coef = (maxpos - 1) / 2.0 / a  # 127/a
            b_coef = (maxpos + 1) / 2.0      # 128
            dn_min, dn_max = 1, maxpos
    else:
        # Unsigned: [lo, hi] -> [1, MAXPOS].
        span = (hi - lo) if (hi - lo) > 0 else 1.0
        a_coef = (maxpos - 1) / span
        b_coef = 1.0 - a_coef * lo
        dn_min, dn_max = 1, maxpos

    scale = 1.0 / a_coef
    offset = -b_coef / a_coef
    return {
        "a_coef": float(a_coef),
        "b_coef": float(b_coef),
        "dn_min": int(dn_min),
        "dn_max": int(dn_max),
        "scale": float(scale),
        "offset": float(offset),
        "nodata": 0.0,
        "signed": bool(signed),
    }


def quantize_array(arr, qp: Dict[str, float], dtype: str):
    """NumPy quantization of a float array to integer codes (NaN/NoData -> 0).

    Mirrors the GPU path: ``DN = clip(round(a_coef*v + b_coef), dn_min, dn_max)``.
    Used by the tile backend (CPU NumPy results).
    """
    a = np.asarray(arr, dtype=np.float32)
    dn = np.rint(np.float32(qp["a_coef"]) * a + np.float32(qp["b_coef"]))
    dn = np.clip(dn, qp["dn_min"], qp["dn_max"])
    dn = np.where(np.isnan(a), 0.0, dn)
    return dn.astype(np.dtype(dtype))


def apply_scale_offset(path: str, scale: float, offset: float) -> bool:
    """Best-effort GDAL band scale/offset write (DN -> physical recovery).

    Returns True on success.  Non-critical: failures are swallowed by callers.
    """
    try:
        from osgeo import gdal
    except Exception:
        return False
    ds = gdal.Open(str(path), gdal.GA_Update)
    if ds is None:
        return False
    try:
        band = ds.GetRasterBand(1)
        band.SetScale(float(scale))
        band.SetOffset(float(offset))
        return True
    finally:
        ds = None
