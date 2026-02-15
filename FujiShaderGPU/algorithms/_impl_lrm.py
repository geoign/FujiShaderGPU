"""
FujiShaderGPU/algorithms/_impl_lrm.py

LRM (Local Relief Model) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da

from ._base import DaskAlgorithm
from ._nan_utils import handle_nan_with_gaussian, restore_nan
from ._global_stats import compute_global_stats
from ._normalization import lrm_stat_func


def compute_lrm_block(block, *, kernel_size=25, pixel_size=1.0,
                     std_global=None, normalize=True):
    """Compute Local Relief Model (DEM minus local trend)."""
    nan_mask = cp.isnan(block)
    sigma = kernel_size / 3.0
    trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
    lrm = block - trend
    result = lrm
    if normalize:
        if std_global is not None and std_global > 1e-9:
            scale = float(std_global)
        else:
            scale = lrm_stat_func(lrm)[0]
            if not cp.isfinite(scale) or scale <= 1e-9:
                scale = 1.0
        result = cp.tanh(lrm / (2.5 * scale))
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


class LRMAlgorithm(DaskAlgorithm):
    """局所起伏モデルアルゴリズム"""
    def process(self, gpu_arr, **params):
        kernel_size = params.get('kernel_size', 25)
        stats = params.get('global_stats', None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) == 1
            and float(stats[0]) > 1e-9
        )
        if not stats_ok:
            stats = compute_global_stats(
                gpu_arr,
                lambda data: lrm_stat_func(compute_lrm_block(data, kernel_size=kernel_size, normalize=False)),
                compute_lrm_block,
                {'kernel_size': kernel_size, 'normalize': False},
                downsample_factor=None, depth=int(kernel_size * 2))
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9):
            stats = (1.0,)
        return gpu_arr.map_overlap(
            compute_lrm_block, depth=int(kernel_size * 2),
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            kernel_size=kernel_size, std_global=stats[0], normalize=True)

    def get_default_params(self):
        return {'kernel_size': 25}


__all__ = ["compute_lrm_block", "LRMAlgorithm"]
