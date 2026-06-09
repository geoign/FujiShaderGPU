"""
FujiShaderGPU/algorithms/_normalization.py

Per-algorithm statistics and normalization functions.
Module split out from dask_shared.py (Phase 1).
"""
from __future__ import annotations
from typing import Tuple
import cupy as cp

# Robust percentile that maps to display magnitude 1.0.  Set to 99 so that the
# central ~99% of the (overview-derived) value distribution lands within
# [-1,1] / [0,1] -- i.e. "most values fit the standard display range".  The
# normalized float is NOT clipped, so the rare high-amplitude tail passes through
# just past +/-1 and stays informative (see OUTPUT_VALUE_RANGES for how int16/
# uint8 reserve a little headroom for it).
NORMAL_PERCENTILE = 99.0


# --- TopoUSM Fast ---

def topousm_fast_stat_func(data: cp.ndarray) -> Tuple[float]:
    """TopoUSM Fast normalization scale from robust absolute percentile."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        abs_valid = cp.abs(valid_data)
        scale = float(cp.percentile(abs_valid, NORMAL_PERCENTILE))
        if scale > 1e-9:
            return (scale,)
        # Fallback for near-constant tiles/arrays.
        return (float(cp.std(valid_data)) if float(cp.std(valid_data)) > 1e-9 else 1.0,)
    return (1.0,)


def topousm_fast_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """Normalization for TopoUSM Fast."""
    scale_global = stats[0]
    if scale_global > 0:
        # p99(|TopoUSM Fast|) -> magnitude 1.0; tail passes through unclipped.
        return block / scale_global
    return cp.zeros_like(block)


def robust_unsigned_stretch_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Robust contrast-stretch stats for bounded maps concentrated in a narrow
    high band (ambient_occlusion, openness): map [p1, p99] -> [0, 1].

    Returns ``(norm_min=p1, norm_scale=p99-p1)`` so the darkest features anchor at
    0 and the bright bulk fills the code range instead of piling up near the top.
    """
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    lo = float(cp.percentile(valid, 100.0 - NORMAL_PERCENTILE))  # p1
    hi = float(cp.percentile(valid, NORMAL_PERCENTILE))          # p99
    scale = hi - lo
    return (lo, scale if scale > 1e-9 else 1.0)


__all__ = [
    "topousm_fast_stat_func",
    "topousm_fast_norm_func",
    "robust_unsigned_stretch_stat_func",
    "NORMAL_PERCENTILE",
]
