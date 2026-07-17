"""
FujiShaderGPU/algorithms/_impl_topousm_fast.py

TopoUSM Fast algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import logging
from typing import List, Optional
import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm
from ._nan_utils import (
    handle_nan_with_gaussian, handle_nan_with_uniform,
    restore_nan,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
    _bilinear_sample_coarse,
)
from ._global_stats import (
    apply_global_normalization,
)
from ._normalization import topousm_fast_stat_func, topousm_fast_norm_func

logger = logging.getLogger(__name__)


def high_pass(block: cp.ndarray, *, sigma: float) -> cp.ndarray:
    """High-pass filter via CuPy Gaussian blur then difference (NaN-aware)."""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid_mask = (~nan_mask).astype(cp.float32)

        blurred_values = gaussian_filter(filled * valid_mask, sigma=sigma, mode="nearest", truncate=4.0)
        blurred_weights = gaussian_filter(valid_mask, sigma=sigma, mode="nearest", truncate=4.0)

        blurred = cp.where(blurred_weights > 0, blurred_values / blurred_weights, 0)
    else:
        blurred = gaussian_filter(block, sigma=sigma, mode="nearest", truncate=4.0)

    result = block - blurred
    result = restore_nan(result, nan_mask)

    return result


def compute_topousm_fast_efficient_block(block: cp.ndarray, *,
                               radii: List[int] = [4, 16, 64],
                               weights: Optional[List[float]] = None,
                               pixel_size: float = 1.0) -> cp.ndarray:
    """Efficient TopoUSM Fast computation (memory-optimized)."""
    nan_mask = cp.isnan(block)

    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    else:
        if not isinstance(weights, cp.ndarray):
            weights = cp.array(weights, dtype=cp.float32)
        if len(weights) != len(radii):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of radii ({len(radii)})")

    topousm_fast_combined = None

    for i, (radius, weight) in enumerate(zip(radii, weights)):
        ds_factor = _radius_to_downsample_factor(
            float(radius),
            block_shape=block.shape,
            pixel_size=pixel_size,
            algorithm_name="topousm_fast",
        )
        if ds_factor > 1:
            small = _downsample_nan_aware(block, ds_factor)
            r_small = max(1, int(round(float(radius) / ds_factor)))
            if r_small <= 1:
                mean_small, _ = handle_nan_with_gaussian(small, sigma=1.0, mode='nearest')
            else:
                mean_small, _ = handle_nan_with_uniform(small, size=2 * r_small + 1, mode='reflect')
            mean_elev = _upsample_to_shape(mean_small, block.shape)
        elif radius <= 1:
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=1.0, mode='nearest')
        else:
            kernel_size = 2 * radius + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='reflect')

        diff = weight * (block - mean_elev)

        if topousm_fast_combined is None:
            topousm_fast_combined = diff
        else:
            topousm_fast_combined += diff

        del mean_elev, diff

    topousm_fast_combined = restore_nan(topousm_fast_combined, nan_mask)

    return topousm_fast_combined


def topousm_fast_default_large_radius_threshold(min_chunk: int) -> int:
    """Radii above this are computed from the COG overview (no large halo).

    Default = max(256, min_chunk // 16): radii within this are cheap to halo at
    full resolution; larger radii are low-frequency and better taken from the
    stored overview.
    """
    return int(max(256, int(min_chunk) // 16))


def split_radii_by_threshold(radii, weights, threshold):
    """Split (radii, weights) into (small, large) groups by ``threshold`` px.

    Weights are NOT renormalized: each group keeps its absolute weights so the
    two partial TopoUSM Fast sums add back to the original full-radii TopoUSM Fast.
    """
    n = len(radii)
    if weights is None or len(weights) != n:
        weights = [1.0 / n] * n
    small_r, small_w, large_r, large_w = [], [], [], []
    for r, w in zip(radii, weights):
        if int(r) > int(threshold):
            large_r.append(int(r))
            large_w.append(float(w))
        else:
            small_r.append(int(r))
            small_w.append(float(w))
    return small_r, small_w, large_r, large_w


def compute_topousm_fast_large_coarse_field(
    coarse_dem: cp.ndarray,
    *,
    large_radii: List[int],
    large_weights: List[float],
    decimation: float,
) -> cp.ndarray:
    """Weighted sum of large-radius local means on the coarse (overview) grid.

    Returns ``Sum_i w_i * mean_{r_i}(coarse_dem)`` (NaN-aware), where ``r_i`` is
    the full-resolution radius scaled into the coarse grid by ``decimation``.
    The full large-radius TopoUSM Fast is then ``W_large * block - upsample(field)``.
    """
    field = None
    for r, w in zip(large_radii, large_weights):
        r_coarse = max(1, int(round(float(r) / max(decimation, 1.0))))
        if r_coarse <= 1:
            mean_c, _ = handle_nan_with_gaussian(coarse_dem, sigma=1.0, mode="nearest")
        else:
            mean_c, _ = handle_nan_with_uniform(coarse_dem, size=2 * r_coarse + 1, mode="reflect")
        term = cp.float32(w) * mean_c
        field = term if field is None else field + term
    return field.astype(cp.float32)


def _topousm_fast_add_large_block(
    block: cp.ndarray,
    *,
    coarse_field: cp.ndarray,
    w_large: float,
    off_r: int,
    off_c: int,
    full_h: int,
    full_w: int,
    block_info=None,
) -> cp.ndarray:
    """Large-radius TopoUSM Fast contribution for one block: W_large*block - upsample(field).

    The coarse field is sampled at the block's *global* pixel positions
    (block-local location + (off_r, off_c)) so the result is seam-free across
    chunks (Dask, offset 0) and tiles (offset = tile window origin).
    """
    if block_info is not None and block_info.get(0) is not None:
        (r0, r1), (c0, c1) = block_info[0]["array-location"][0], block_info[0]["array-location"][1]
    else:  # pragma: no cover - direct (non-dask) call fallback
        r0, c0 = 0, 0
        r1, c1 = block.shape[0], block.shape[1]
    r0 += int(off_r)
    r1 += int(off_r)
    c0 += int(off_c)
    c1 += int(off_c)

    up = _bilinear_sample_coarse(coarse_field, r0, r1, c0, c1, full_h, full_w)
    return (cp.float32(w_large) * block - up).astype(cp.float32)


def multiscale_topousm_fast(gpu_arr: da.Array, *,
                   radii: List[int],
                   weights: Optional[List[float]] = None,
                   pixel_size: float = 1.0) -> da.Array:
    """Efficient multiscale TopoUSM Fast (Dask version)."""
    if not radii:
        raise ValueError("At least one radius value is required")

    max_radius = max(radii)
    # A uniform mean of radius R needs exactly R pixels of halo on each side, so
    # R + a small margin is seam-free for the trimmed core (results identical to
    # the previous 2*R+1 halo) while reading ~half the data per chunk for large
    # radii.  Dask map_overlap auto-rechunks when the halo exceeds an edge chunk,
    # exactly as it did for the old (larger) depth.
    depth = int(max_radius + 16)

    result = gpu_arr.map_overlap(
        compute_topousm_fast_efficient_block,
        depth=depth,
        boundary="reflect",
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        radii=radii,
        weights=weights,
        pixel_size=pixel_size,
    )

    return result


def compute_topousm_fast_input_sample_stats(
    gpu_arr: da.Array,
    *,
    radii: List[int],
    weights: Optional[List[float]] = None,
    pixel_size: float = 1.0,
) -> tuple:
    """Estimate TopoUSM Fast scale from a bounded central crop of the DEM.

    A strided sample of the full-resolution Dask array (``gpu_arr[::n, ::n]``)
    looks cheap but forces *every* chunk -- i.e. the entire dataset -- to be read
    from disk and copied to the GPU before any write progress is visible, which
    stalls on very large rasters.  Reading a single contiguous central window
    only materializes the few chunks overlapping that window, giving a stable
    global scale at a tiny, bounded cost.  ``compute_topousm_fast_efficient_block`` already
    downsamples large radii internally, so the original radii are used as-is.
    """
    h, w = gpu_arr.shape
    max_radius = max((int(r) for r in radii), default=1)
    # Window must comfortably contain the largest radius footprint while staying
    # bounded regardless of the full raster size.
    win = int(min(int(h), int(w), max(4096, max_radius * 4)))
    win = max(256, win)
    y0 = max(0, (int(h) - win) // 2)
    x0 = max(0, (int(w) - win) // 2)
    y1 = min(int(h), y0 + win)
    x1 = min(int(w), x0 + win)

    logger.info(
        "Estimating TopoUSM Fast global stats from central %dx%d window (radii=%s)",
        x1 - x0,
        y1 - y0,
        list(radii),
    )

    sample = gpu_arr[y0:y1, x0:x1].compute()
    if getattr(sample, "size", 0) == 0:
        return (1.0,)

    topousm_fast_sample = compute_topousm_fast_efficient_block(
        sample.astype(cp.float32, copy=False),
        radii=[max(1, int(r)) for r in radii],
        weights=weights,
        pixel_size=float(pixel_size),
    )
    stats = topousm_fast_stat_func(topousm_fast_sample)
    logger.info("TopoUSM Fast global normalization stats estimated: abs_p99=%.6f", float(stats[0]))
    return stats


class TopoUSMFastAlgorithm(DaskAlgorithm):
    """TopoUSM Fast algorithm (efficient implementation)."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        pixel_size = params.get('pixel_size', 1.0)
        radii = params.get('radii', None)
        weights = params.get('weights', None)

        if radii is None:
            radii = self._determine_optimal_radii(pixel_size)

        # Large-radius-from-overview fast path: when the orchestrator supplies a
        # precomputed coarse field (Sum w*mean for large radii on the overview),
        # large radii contribute as `W_large*block - upsample(field)` (no halo),
        # and only the small radii are computed at full resolution.  Seam-free
        # via global coords (offset 0 for Dask, tile-window origin for tiles).
        coarse_field = params.get("_topousm_fast_coarse_field", None)
        if coarse_field is not None:
            small_r = params.get("_topousm_fast_small_radii", [])
            small_w = params.get("_topousm_fast_small_weights", None)
            w_large = float(params.get("_topousm_fast_w_large", 0.0))
            off_r, off_c = params.get("_topousm_fast_field_offset", (0, 0))
            full_h, full_w = params.get("_topousm_fast_full_shape", gpu_arr.shape)
            large_part = gpu_arr.map_blocks(
                _topousm_fast_add_large_block,
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                coarse_field=coarse_field,
                w_large=w_large,
                off_r=int(off_r),
                off_c=int(off_c),
                full_h=int(full_h),
                full_w=int(full_w),
            )
            if small_r:
                topousm_fast = multiscale_topousm_fast(
                    gpu_arr, radii=small_r, weights=small_w, pixel_size=pixel_size,
                ) + large_part
            else:
                topousm_fast = large_part
        else:
            topousm_fast = multiscale_topousm_fast(gpu_arr, radii=radii, weights=weights, pixel_size=pixel_size)

        # Prefer externally supplied global stats (tile backend computes once).
        stats = params.get("global_stats", None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 1
            and float(stats[0]) > 1e-9
        )
        if not stats_ok:
            stats = compute_topousm_fast_input_sample_stats(
                gpu_arr,
                radii=radii,
                weights=weights,
                pixel_size=pixel_size,
            )
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9):
            stats = (1.0,)

        return topousm_fast.map_blocks(
            lambda block: apply_global_normalization(block, topousm_fast_norm_func, stats),
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )

    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """Determine optimal radii based on pixel size."""
        target_distances = [5, 20, 80, 320]
        radii = []

        for dist in target_distances:
            radius = int(dist / pixel_size)
            radius = max(2, min(radius, 256))
            radii.append(radius)

        radii = sorted(list(set(radii)))

        return radii

    def get_default_params(self) -> dict:
        return {
            'mode': 'radius',
            'radii': None,
            'weights': None,
            'sigmas': None,
            'agg': 'mean',
            'auto_sigma': False,
        }


__all__ = [
    "high_pass",
    "compute_topousm_fast_efficient_block",
    "multiscale_topousm_fast",
    "TopoUSMFastAlgorithm",
    "topousm_fast_default_large_radius_threshold",
    "split_radii_by_threshold",
    "compute_topousm_fast_large_coarse_field",
]
