"""
FujiShaderGPU/algorithms/_impl_multiscale_terrain.py

Multiscale Terrain (マルチスケール地形) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import handle_nan_with_gaussian, restore_nan
from ._global_stats import determine_optimal_downsample_factor


class MultiscaleDaskAlgorithm(DaskAlgorithm):
    """マルチスケール地形アルゴリズム"""
    def process(self, gpu_arr, **params):
        scales = params.get('scales', [1, 10, 50, 100])
        weights = params.get('weights', None)
        downsample_factor = params.get('downsample_factor', None)
        if downsample_factor is None:
            downsample_factor = determine_optimal_downsample_factor(
                gpu_arr.shape, algorithm_name='multiscale_terrain')
        if weights is None:
            weights = [1.0 / s for s in scales]
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)
        downsampled = gpu_arr[::downsample_factor, ::downsample_factor]
        results_small = []
        max_scale_small = max([max(1, s // downsample_factor) for s in scales])
        common_depth_small = min(int(4 * max_scale_small), Constants.MAX_DEPTH)

        def create_weighted_combiner(weights):
            def weighted_combine_for_stats(*blocks):
                nan_mask = cp.isnan(blocks[0])
                result = cp.zeros_like(blocks[0])
                for i, block in enumerate(blocks):
                    valid = ~cp.isnan(block)
                    result[valid] += block[valid] * weights[i]
                result[nan_mask] = cp.nan
                return result
            return weighted_combine_for_stats

        for i, scale in enumerate(scales):
            scale_small = max(1, scale // downsample_factor)
            def compute_detail_small(block, *, scale):
                smoothed, nan_mask = handle_nan_with_gaussian(
                    block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            detail_small = downsampled.map_overlap(
                compute_detail_small, depth=common_depth_small,
                boundary='reflect', dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32), scale=scale_small)
            results_small.append(detail_small)

        combined_small = da.map_blocks(
            create_weighted_combiner(weights), *results_small,
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32)
        ).compute()
        valid_data = combined_small[~cp.isnan(combined_small)]
        if len(valid_data) > 0:
            norm_min = float(cp.percentile(valid_data, 5))
            norm_max = float(cp.percentile(valid_data, 95))
        else:
            norm_min, norm_max = 0.0, 1.0
        if params.get('verbose', False):
            print(f"Multiscale Terrain global stats: min={norm_min:.3f}, max={norm_max:.3f}")
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
            if norm_max > norm_min:
                result = (result - norm_min) / (norm_max - norm_min)
                result = cp.clip(result, 0, 1)
            else:
                result = cp.full_like(result, 0.5)
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
