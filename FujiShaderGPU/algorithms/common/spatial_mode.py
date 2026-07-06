"""Common helpers for local/spatial algorithm modes.

Spatial-mode auto radii/weights use a deterministic, DEM-size-aware rule that is
shared by every spatial-capable algorithm and by both backends (tile + Dask-CUDA):

* radii  : the geometric sequence ``[2, 8, 32, 128, 512, 2048]`` (pixels),
  truncated so its largest element does not exceed ``min(short_side/10, 2048)``
  where ``short_side`` is the shorter side of the input DEM in pixels.  At least
  one radius (``2``) is always kept.
* weights: a ``2**n`` profile normalized to sum 1.0 -- the nearest (smallest)
  radius is weighted most, the farthest least -- for ``n`` radii, regardless of
  how ``n`` arose (auto truncation or an explicit ``--radii`` count).
"""
from __future__ import annotations

from typing import List, Optional, Tuple


# Geometric radii (pixels) and the absolute auto cap.  The sequence top equals
# the cap, so the cap is honoured implicitly; short-side truncation does the rest.
AUTO_RADII_SEQUENCE: Tuple[int, ...] = (2, 8, 32, 128, 512, 2048)
AUTO_RADIUS_MAX: int = 2048

# Radius-driven spatial algorithms whose radii/weights follow the shared auto rule
# (topousm_fast + the per-pixel spatial algorithms).  The intrinsically multi-scale
# algorithms (multiscale_terrain / scale_space_surprise / visual_saliency /
# fractal_anomaly) keep their own --scales / --fractal-radii sets and are excluded.
RADII_DRIVEN_ALGOS = frozenset({
    "topousm_fast",
    "hillshade",
    "slope",
    "specular",
    "atmospheric_scattering",
    "curvature",
    "ambient_occlusion",
    "openness",
    "multi_light_uncertainty",
    "npr_edges",
    "structure_tensor",
    "frangi",
})

# Algorithms whose result is undefined at a single scale (fractal dimension needs a
# radius regression; scale-space surprise needs a scale range; saliency center-
# surround needs >=2 scales).  `--mode local` (radii=[1]) is meaningless for them,
# so they fall back to the spatial default with a warning.
MULTISCALE_REQUIRED_ALGOS = frozenset({
    "fractal_anomaly",
    "scale_space_surprise",
    "visual_saliency",
    "scale_drift",      # drift needs >= 2 scale levels
    "phase_congruency",  # needs a wavelength bank
})

# Canonical single-pixel "local" profile shared by both backends.
LOCAL_RADII = [1]
LOCAL_WEIGHTS = [1.0]


def auto_spatial_radii(short_side_px: Optional[float]) -> List[int]:
    """Geometric radii truncated to ``min(short_side/10, 2048)`` pixels.

    ``short_side_px=None`` means "no DEM-size constraint" (used only as a
    defensive fallback when the caller has no dimensions); the full sequence is
    returned, still capped at ``AUTO_RADIUS_MAX``.
    """
    if short_side_px is None:
        limit = float(AUTO_RADIUS_MAX)
    else:
        limit = min(float(AUTO_RADIUS_MAX), float(short_side_px) / 10.0)
    radii = [r for r in AUTO_RADII_SEQUENCE if float(r) <= limit]
    if not radii:
        # DEM too small for even the smallest radius under the /10 rule: keep one.
        radii = [AUTO_RADII_SEQUENCE[0]]
    return list(radii)


def auto_spatial_weights(n: int) -> List[float]:
    """``2**(n-1 .. 0)`` normalized to sum 1.0 (nearer radii weigh more)."""
    if n <= 0:
        return []
    raw = [2.0 ** (n - 1 - i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def auto_spatial_profile(
    short_side_px: Optional[float],
    radii: Optional[List[int]] = None,
) -> Tuple[List[int], List[float]]:
    """Return (radii, weights) for spatial auto-mode.

    If ``radii`` is given it is used as-is (only the weight count follows it);
    otherwise the DEM-size-aware geometric radii are generated.  Weights always
    follow the ``2**n`` rule for the resulting radius count.
    """
    if radii is None:
        radii = auto_spatial_radii(short_side_px)
    else:
        radii = [int(round(float(r))) for r in radii]
    return radii, auto_spatial_weights(len(radii))
