"""
FujiShaderGPU/algorithms/_impl_multiscale_terrain.py

Multiscale Terrain algorithm implementation.
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import logging
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_with_gaussian, restore_nan,
    coarsen_factor_for_shape, coarse_large_radius_response,
)
from ._normalization import NORMAL_PERCENTILE

logger = logging.getLogger(__name__)


def _resolve_scales(params):
    """Scales for the multiscale detail field.

    Explicit spatial ``radii`` (the unified --radii/--mode spatial path) take
    precedence and are used directly as gaussian scales (a radius *is* a spatial
    scale here); otherwise the algorithm's own ``scales`` are used.  Returns
    ``(scales, is_spatial)``.
    """
    radii = params.get("radii")
    if radii:
        return [float(r) for r in radii], True
    scales = params.get("scales") or [1, 10, 50, 100]
    return [float(s) for s in scales], False


def _detail_block(block, *, scale, **_ignored):
    """Full-resolution detail = block - gaussian(block, sigma=scale) (NaN-aware)."""
    smoothed, nan_mask = handle_nan_with_gaussian(
        block, sigma=max(float(scale), 0.5), mode='nearest')
    detail = block - smoothed
    return restore_nan(detail, nan_mask).astype(cp.float32)


def _smooth_only_block(block, *, scale, **_ignored):
    """Just the NaN-aware gaussian smooth (a low-frequency field).

    Used for large scales: the *smooth* term is low-frequency so it can be
    computed on a coarsened copy and upsampled without losing accuracy, while the
    fine detail is recovered by subtracting it from the full-resolution block.
    Computing the whole ``block - smooth`` detail on the coarse grid (as RVI /
    specular do for their already-smooth responses) would instead drop every
    feature finer than the coarse pixel -- wrong for a detail field.
    """
    smoothed, _ = handle_nan_with_gaussian(
        block, sigma=max(float(scale), 0.5), mode='nearest')
    return smoothed.astype(cp.float32)


def compute_multiscale_combined_raw(block, *, scales=None, weights=None, radii=None, **_ignored):
    """Raw (un-normalized) weighted multiscale detail field.

    Mirrors the scale/weight resolution in ``process()`` (explicit ``radii`` win
    over ``scales``; inverse-scale weights when none are supplied, then
    normalized) so the global-stats pre-pass computes the normalization range
    with the same combination as the main pass.
    """
    if radii:
        scales = [float(r) for r in radii]
    if not scales:
        scales = [1, 10, 50, 100]
    if weights is None:
        weights = [1.0 / float(s) for s in scales]
    w = cp.asarray(weights, dtype=cp.float32)
    w = w / w.sum()
    n = min(len(scales), int(w.shape[0]))
    nan_mask = cp.isnan(block)
    combined = cp.zeros_like(block, dtype=cp.float32)
    for i in range(n):
        smoothed, _ = handle_nan_with_gaussian(
            block, sigma=max(float(scales[i]), 0.5), mode='nearest')
        detail = block - smoothed
        valid = ~cp.isnan(detail)
        combined[valid] += detail[valid] * float(w[i])
    combined[nan_mask] = cp.nan
    return combined.astype(cp.float32)


def multiscale_stat_func(combined):
    """Unsigned (p1-anchored, p99-scaled) stats: returns ``(norm_min, norm_scale)``."""
    valid = combined[~cp.isnan(combined)]
    if valid.size == 0:
        return (0.0, 1.0)
    norm_min = float(cp.percentile(valid, 1))
    norm_scale = float(cp.percentile(cp.maximum(valid - norm_min, 0.0), NORMAL_PERCENTILE))
    return (norm_min, norm_scale if norm_scale > 1e-9 else 1.0)


class MultiscaleDaskAlgorithm(DaskAlgorithm):
    """Multiscale terrain algorithm."""

    def _resolve_norm_stats(self, gpu_arr, scales, weights, params):
        """(norm_min, norm_scale): use the pre-pass global stats, else estimate
        from a bounded central full-resolution crop with the same scales."""
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)
        stats = params.get('global_stats', None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 2
            and float(stats[1]) > 1e-9
        )
        if stats_ok:
            return float(stats[0]), float(stats[1])

        # Estimate global normalization from a bounded central crop computed at
        # full resolution.  Striding the full array would force every chunk to be
        # read/copied to the GPU before any write progress; a contiguous window
        # only reads the chunks it overlaps and uses the same scales as output.
        h, w = gpu_arr.shape
        win = int(min(int(h), int(w), max(4096, int(common_depth) * 4)))
        win = max(256, win)
        y0 = max(0, (int(h) - win) // 2)
        x0 = max(0, (int(w) - win) // 2)
        y1 = min(int(h), y0 + win)
        x1 = min(int(w), x0 + win)

        sample_block = gpu_arr[y0:y1, x0:x1].compute()
        nan_mask_s = cp.isnan(sample_block)
        combined_small = cp.zeros_like(sample_block, dtype=cp.float32)
        for i, scale in enumerate(scales):
            smoothed_s, _ = handle_nan_with_gaussian(
                sample_block, sigma=max(float(scale), 0.5), mode='nearest')
            detail_s = sample_block - smoothed_s
            valid_s = ~cp.isnan(detail_s)
            combined_small[valid_s] += detail_s[valid_s] * float(weights[i])
        combined_small[nan_mask_s] = cp.nan
        valid_data = combined_small[~cp.isnan(combined_small)]
        if valid_data.size > 0:
            norm_min = float(cp.percentile(valid_data, 1))
            norm_scale = float(cp.percentile(cp.maximum(valid_data - norm_min, 0.0), NORMAL_PERCENTILE))
        else:
            norm_min, norm_scale = 0.0, 1.0
        if norm_scale <= 1e-9:
            norm_scale = 1.0
        return norm_min, norm_scale

    def process(self, gpu_arr, **params):
        scales, is_spatial = _resolve_scales(params)
        weights = params.get('weights', None)
        if weights is None or len(weights) != len(scales):
            weights = [1.0 / float(s) for s in scales]
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()

        norm_min, norm_scale = self._resolve_norm_stats(gpu_arr, scales, weights, params)
        if params.get('verbose', False):
            logger.info("Multiscale Terrain global stats: min=%.3f, p80_scale=%.3f", norm_min, norm_scale)

        is_geo = bool(params.get('is_geographic_dem', False))
        pixel_size = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        # Coarse path is only used in the explicit spatial (radii) mode; the
        # default-scales path keeps its original uniform-depth behavior exactly.
        # Coarsen for large scales on geographic DEMs too (pixel-based, correct).
        F = coarsen_factor_for_shape(gpu_arr.shape) if is_spatial else 1
        common_depth = min(int(4 * max(scales)), Constants.MAX_DEPTH)
        cache = {}

        results = []
        for scale in scales:
            # The gaussian at sigma==scale needs a halo of ~4*scale to be seamless
            # across chunks.  When that exceeds MAX_DEPTH, compute the (low-freq)
            # smooth on a coarsened copy and subtract it from the full-res block,
            # keeping fine detail, seams, and VRAM all under control.
            if F > 1 and int(4 * scale) > Constants.MAX_DEPTH:
                smooth_up = coarse_large_radius_response(
                    gpu_arr, block_fn=_smooth_only_block, radius_kw='scale',
                    radius=float(scale), factor=F,
                    depth_for_radius=lambda sc: min(int(4 * sc) + 1, Constants.MAX_DEPTH),
                    pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
                    coarse_cache=cache,
                    coarse_dem=params.get("_overview_coarse_dem"),
                    coarse_decimation=params.get("_overview_decimation"),
                )
                results.append(gpu_arr - smooth_up)
            else:
                depth = min(int(4 * scale), Constants.MAX_DEPTH) if is_spatial else common_depth
                results.append(gpu_arr.map_overlap(
                    _detail_block, depth=depth, boundary='reflect',
                    dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
                    scale=float(scale)))

        def weighted_combine_and_normalize(*blocks):
            """Normalize using global statistics."""
            nan_mask = cp.isnan(blocks[0])
            result = cp.zeros_like(blocks[0])
            for i, block in enumerate(blocks):
                valid = ~cp.isnan(block)
                result[valid] += block[valid] * weights[i]
            result = (result - norm_min) / norm_scale
            result = cp.maximum(result, 0.0)  # p99 -> 1.0; tail passes through
            result = cp.power(result, Constants.DEFAULT_GAMMA)
            result[nan_mask] = cp.nan
            return result.astype(cp.float32)

        combined = da.map_blocks(
            weighted_combine_and_normalize, *results,
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32))
        return combined

    def get_default_params(self):
        return {
            'scales': [1, 10, 50, 100], 'weights': None,
            'downsample_factor': None, 'verbose': False,
        }


__all__ = ["MultiscaleDaskAlgorithm"]
