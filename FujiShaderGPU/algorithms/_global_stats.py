"""
FujiShaderGPU/algorithms/_global_stats.py

Global statistics utilities.
Compute statistics and apply normalization on downsampled representative data.
Module split out from dask_shared.py (Phase 1).
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import cupy as cp
import dask.array as da

from ._nan_utils import restore_nan


def determine_optimal_downsample_factor(
    data_shape: Tuple[int, int],
    algorithm_name: str = None,
    target_pixels: int = 500000,  # target pixel count (1000x1000)
    min_factor: int = 5,
    max_factor: int = 100,
    algorithm_complexity: Dict[str, float] = None) -> int:
    """
    Determine the optimal downsample factor from data size and algorithm characteristics.

    Parameters:
    -----------
    data_shape : Tuple[int, int]
        Shape of the input data (height, width)
    algorithm_name : str
        Algorithm name (for complexity adjustment)
    target_pixels : int
        Target pixel count after downsampling
    min_factor : int
        Minimum downsample factor
    max_factor : int
        Maximum downsample factor
    algorithm_complexity : Dict[str, float]
        Per-algorithm complexity factors (defaults to a built-in dict)

    Returns:
    --------
    int : the optimal downsample factor
    """
    # Algorithm complexity factors (higher = more expensive)
    if algorithm_complexity is None:
        algorithm_complexity = {
            'topousm_fast': 1.2,                    # multiscale processing
            'hillshade': 0.8,              # simple gradient computation
            'slope': 0.8,                  # simple gradient computation
            'specular': 1.5,               # roughness computation is heavy
            'atmospheric_scattering': 0.9,
            'multiscale_terrain': 1.5,     # multiscale processing
            'curvature': 1.0,              # second derivative
            'visual_saliency': 1.4,        # multiscale feature extraction
            'npr_edges': 1.1,              # edge detection
            'ambient_occlusion': 2.0,      # most expensive
            'openness': 1.8,               # multi-direction search
            'fractal_anomaly': 1.6,        # multiscale regression
        }

    # Current pixel count
    current_pixels = data_shape[0] * data_shape[1]

    # Base downsample factor (computed via square root)
    base_factor = cp.sqrt(current_pixels / target_pixels).get()

    # Adjust by algorithm complexity
    complexity = algorithm_complexity.get(algorithm_name, 1.0)
    adjusted_factor = base_factor * complexity

    # Convert to int and clamp to range
    downsample_factor = int(cp.clip(adjusted_factor, min_factor, max_factor))

    # Reduce the factor when the data is small
    if current_pixels < 1_000_000:  # under 1M pixels
        downsample_factor = min(downsample_factor, 2)
    elif current_pixels < 10_000_000:  # under 10M pixels
        downsample_factor = min(downsample_factor, 4)
    return downsample_factor


def compute_global_stats(gpu_arr: da.Array,
                        stat_func: callable,
                        algorithm_func: callable,
                        algorithm_params: dict,
                        downsample_factor: int = None,  # kept for backward compat (unused)
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
    "determine_optimal_downsample_factor",
    "compute_global_stats",
    "apply_global_normalization",
    "apply_display_stretch_dask",
    "robust_unsigned_stretch_stat_func",
]
