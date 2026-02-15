"""
FujiShaderGPU/algorithms/_impl_rvi.py

RVI (Ridge-Valley Index) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 2)。
"""
from __future__ import annotations
from typing import List, Optional
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm
from ._nan_utils import (
    handle_nan_with_gaussian, handle_nan_with_uniform,
    restore_nan,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
)
from ._global_stats import compute_global_stats, apply_global_normalization
from ._normalization import rvi_stat_func, rvi_norm_func


def high_pass(block: cp.ndarray, *, sigma: float) -> cp.ndarray:
    """CuPy でガウシアンぼかし後に差分を取るhigh pass フィルタ（NaN対応）"""
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


def compute_rvi_efficient_block(block: cp.ndarray, *,
                               radii: List[int] = [4, 16, 64],
                               weights: Optional[List[float]] = None,
                               pixel_size: float = 1.0) -> cp.ndarray:
    """効率的なRVI計算（メモリ最適化版）"""
    nan_mask = cp.isnan(block)

    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    else:
        if not isinstance(weights, cp.ndarray):
            weights = cp.array(weights, dtype=cp.float32)
        if len(weights) != len(radii):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of radii ({len(radii)})")

    rvi_combined = None

    for i, (radius, weight) in enumerate(zip(radii, weights)):
        ds_factor = _radius_to_downsample_factor(
            float(radius),
            block_shape=block.shape,
            pixel_size=pixel_size,
            algorithm_name="rvi",
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

        if rvi_combined is None:
            rvi_combined = diff
        else:
            rvi_combined += diff

        del mean_elev, diff

    rvi_combined = restore_nan(rvi_combined, nan_mask)

    return rvi_combined


def multiscale_rvi(gpu_arr: da.Array, *,
                   radii: List[int],
                   weights: Optional[List[float]] = None,
                   pixel_size: float = 1.0) -> da.Array:
    """効率的なマルチスケールRVI（Dask版）"""
    if not radii:
        raise ValueError("At least one radius value is required")

    max_radius = max(radii)
    depth = max_radius * 2 + 1

    result = gpu_arr.map_overlap(
        compute_rvi_efficient_block,
        depth=depth,
        boundary="reflect",
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        radii=radii,
        weights=weights,
        pixel_size=pixel_size,
    )

    return result


class RVIAlgorithm(DaskAlgorithm):
    """Ridge-Valley Indexアルゴリズム（効率的実装）"""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        pixel_size = params.get('pixel_size', 1.0)
        radii = params.get('radii', None)
        weights = params.get('weights', None)

        if radii is None:
            radii = self._determine_optimal_radii(pixel_size)
        max_radius = max(radii)
        rvi = multiscale_rvi(gpu_arr, radii=radii, weights=weights, pixel_size=pixel_size)

        # Prefer externally supplied global stats (tile backend computes once).
        stats = params.get("global_stats", None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 1
            and float(stats[0]) > 1e-9
        )
        if not stats_ok:
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    rvi,
                    rvi_stat_func,
                    compute_rvi_efficient_block,
                    {'radii': radii, 'weights': weights, 'pixel_size': pixel_size},
                    params.get('downsample_factor', None),
                    depth=max_radius * 2 + 1
                )
            else:
                stats = (1.0,)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9):
            stats = (1.0,)

        return rvi.map_blocks(
            lambda block: apply_global_normalization(block, rvi_norm_func, stats),
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )

    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """ピクセルサイズに基づいて最適な半径を決定"""
        target_distances = [5, 20, 80, 320]
        radii = []

        for dist in target_distances:
            radius = int(dist / pixel_size)
            radius = max(2, min(radius, 256))
            radii.append(radius)

        radii = sorted(list(set(radii)))

        if len(radii) > 4:
            indices = np.logspace(0, np.log10(len(radii)-1), 4).astype(int)
            radii = [radii[int(i)] for i in indices]

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
    "compute_rvi_efficient_block",
    "multiscale_rvi",
    "RVIAlgorithm",
]
