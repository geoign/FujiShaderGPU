"""
FujiShaderGPU/algorithms/_global_stats.py

Global statistics utilities.
Compute statistics from a bounded central full-resolution window and apply
shared normalization helpers.
Module split out from dask_shared.py (Phase 1).
"""
from __future__ import annotations
from typing import Any, Tuple
import cupy as cp
import dask.array as da

from ._nan_utils import restore_nan


def compute_global_stats(gpu_arr: da.Array,
                        stat_func: callable,
                        algorithm_func: callable,
                        algorithm_params: dict,
                        depth: int = None,
                        algorithm_name: str = None) -> Tuple[Any, ...]:
    """
    Shared helper that runs the algorithm on a bounded central crop and computes statistics.

    Striding the full-resolution array (``gpu_arr[::n, ::n]``) forces every chunk
    (the entire dataset) to be read and copied to the GPU for the decimation,
    which stalls on huge rasters before any write progress. Reading a contiguous central
    window materializes only the overlapping chunks and estimates the global scale cheaply.
    The algorithm runs directly as a full-resolution block function on that window,
    so scale-type parameters do not need to be reduced.

    Parameters:
    -----------
    gpu_arr : da.Array
        Input data
    stat_func : callable
        Function that computes statistics; takes a CuPy array and returns a tuple of stats
    algorithm_func : callable
        Algorithm processing function (no-normalization version, a block function)
    algorithm_params : dict
        Algorithm parameters
    depth : int
        Algorithm halo width (used to size the central window)

    Returns:
    --------
    Tuple of statistics
    """
    h, w = gpu_arr.shape
    halo = int(depth) if depth else 1
    # The window comfortably contains the algorithm footprint while staying bounded
    # regardless of the full raster size.
    win = int(min(int(h), int(w), max(4096, halo * 4)))
    win = max(256, win)
    y0 = max(0, (int(h) - win) // 2)
    x0 = max(0, (int(w) - win) // 2)
    y1 = min(int(h), y0 + win)
    x1 = min(int(w), x0 + win)

    sample_block = gpu_arr[y0:y1, x0:x1].compute()
    if getattr(sample_block, "size", 0) == 0:
        return stat_func(cp.zeros((1, 1), dtype=cp.float32))

    result_small = algorithm_func(
        sample_block.astype(cp.float32, copy=False), **algorithm_params
    )

    # Compute statistics
    stats = stat_func(result_small)

    return stats


def apply_global_normalization(block: cp.ndarray,
                              norm_func: callable,
                              stats: Tuple[Any, ...],
                              nan_mask: cp.ndarray = None) -> cp.ndarray:
    """
    Shared helper that applies normalization using global statistics.

    Parameters:
    -----------
    block : cp.ndarray
        Block to process
    norm_func : callable
        Normalization function; takes (block, stats, nan_mask) and returns the normalized block
    stats : tuple
        Global statistics
    nan_mask : cp.ndarray
        NaN mask (optional)

    Returns:
    --------
    Normalized block
    """
    if nan_mask is None:
        nan_mask = cp.isnan(block)

    normalized = norm_func(block, stats, nan_mask)

    # Restore NaN positions
    normalized = restore_nan(normalized, nan_mask)

    return normalized.astype(cp.float32)


def apply_display_stretch_dask(result: da.Array, stats) -> da.Array:
    """Stretch a bounded result by a robust ``(norm_min, norm_scale)`` range.

    Maps ``[norm_min, norm_min+norm_scale]`` (typically the [p1, p99] band) to
    ``[0, 1]`` at the Dask level, clamping the dark tail to 0 and letting the
    bright tail run past 1 (encoded into the integer headroom).  NaN/NoData are
    preserved.  Used by maps that are physically bounded but concentrated in a
    narrow high band (ambient_occlusion, openness).  A no-op when ``stats`` is
    missing/degenerate, so behaviour is unchanged without a global-stats pre-pass.
    """
    if not (isinstance(stats, (tuple, list)) and len(stats) >= 2):
        return result
    lo = float(stats[0])
    scale = float(stats[1])
    if not (scale > 1e-12):
        return result

    def _stretch(block: cp.ndarray) -> cp.ndarray:
        return cp.maximum((block - lo) / scale, cp.float32(0.0)).astype(cp.float32)

    return result.map_blocks(
        _stretch, dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32)
    )


def robust_unsigned_stretch_stat_func(values: cp.ndarray) -> Tuple[float, float]:
    """Robust ``(lo, scale)`` = ``(p1, p99 - p1)`` for a bounded unsigned map.

    Consumed as ``global_stats`` by ``apply_display_stretch_dask`` (which maps
    ``[lo, lo+scale]`` i.e. ``[p1, p99]`` to ``[0, 1]``).  Physically bounded maps
    such as ambient_occlusion and openness concentrate in a narrow high band, so
    without this stretch the integer-encoded output is washed-out white (most codes
    near 255).  Stretching the robust [p1, p99] band fills the code range and
    restores contrast, while the bright tail past p99 runs into the unclipped
    headroom.  Returns ``(0.0, 0.0)`` (a no-op stretch) for an empty/degenerate
    sample, so behaviour is unchanged when stats cannot be estimated.
    """
    if values is None:
        return (0.0, 0.0)
    v = values[cp.isfinite(values)]
    if v.size == 0:
        return (0.0, 0.0)
    lo = float(cp.percentile(v, 1.0))
    hi = float(cp.percentile(v, 99.0))
    scale = hi - lo
    if not (scale > 1e-12):
        return (lo, 0.0)
    return (lo, scale)


__all__ = [
    "compute_global_stats",
    "apply_global_normalization",
    "apply_display_stretch_dask",
    "robust_unsigned_stretch_stat_func",
]
