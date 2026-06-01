"""
FujiShaderGPU/algorithms/_impl_multiscale_terrain.py

Multiscale Terrain (マルチスケール地形) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import logging
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import handle_nan_with_gaussian, restore_nan
from ._normalization import NORMAL_PERCENTILE, OVERFLOW_LIMIT

logger = logging.getLogger(__name__)


class MultiscaleDaskAlgorithm(DaskAlgorithm):
    """マルチスケール地形アルゴリズム"""
    def process(self, gpu_arr, **params):
        scales = params.get('scales', [1, 10, 50, 100])
        weights = params.get('weights', None)
        if weights is None:
            weights = [1.0 / s for s in scales]
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)

        stats = params.get('global_stats', None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 2
            and float(stats[1]) > 1e-9
        )
        if stats_ok:
            norm_min, norm_scale = float(stats[0]), float(stats[1])
        else:
            # Estimate global normalization from a bounded central crop computed
            # at full resolution.  Striding the full array (gpu_arr[::n, ::n])
            # would force every chunk -- the entire dataset -- to be read and
            # copied to the GPU before any write progress is visible, stalling on
            # very large rasters.  A contiguous window only reads the chunks it
            # overlaps and uses the same scales as the final output.
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
        if params.get('verbose', False):
            logger.info("Multiscale Terrain global stats: min=%.3f, p80_scale=%.3f", norm_min, norm_scale)
        results = []
        for scale in scales:
            def compute_detail_with_smooth(block, *, scale):
                smoothed, nan_mask = handle_nan_with_gaussian(
                    block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            detail = gpu_arr.map_overlap(
                compute_detail_with_smooth, depth=common_depth,
                boundary='reflect', dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32), scale=scale)
            results.append(detail)

        def weighted_combine_and_normalize(*blocks):
            """グローバル統計値を使用して正規化"""
            nan_mask = cp.isnan(blocks[0])
            result = cp.zeros_like(blocks[0])
            for i, block in enumerate(blocks):
                valid = ~cp.isnan(block)
                result[valid] += block[valid] * weights[i]
            result = (result - norm_min) / norm_scale
            result = cp.clip(result, 0, OVERFLOW_LIMIT)
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
