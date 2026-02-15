"""
FujiShaderGPU/algorithms/_impl_visual_saliency.py

Visual Saliency (視覚的顕著性) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
from typing import List, Tuple
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm
from ._nan_utils import restore_nan
from ._global_stats import compute_global_stats


def _compress_saliency_feature(feature):
    """Tile-stable feature compression without block-global normalization."""
    return cp.log1p(cp.clip(feature, 0.0, None)).astype(cp.float32)


def visual_saliency_stat_func(data):
    """Global robust range for saliency normalization."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    return (float(cp.percentile(valid_data, 0.5)),
            float(cp.percentile(valid_data, 99.5)))


def compute_visual_saliency_block(block, *, scales=[2, 4, 8, 16],
                                 pixel_size=1.0, pixel_scale_x=None,
                                 pixel_scale_y=None, normalize=True,
                                 norm_min=None, norm_max=None):
    """Itti-style saliency (intensity + orientation conspicuity) for DEM."""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        fill = cp.nanmean(block)
        fill = cp.where(cp.isfinite(fill), fill, 0.0)
        work = cp.where(nan_mask, fill, block).astype(cp.float32)
    else:
        work = block.astype(cp.float32, copy=False)
    use_scales = [max(0.5, float(s)) for s in scales]
    if len(use_scales) < 4:
        use_scales = [2.0, 4.0, 8.0, 16.0]
    c_indices = [0, 1]
    deltas = [2, 3]
    intensity_maps = []
    for ci in c_indices:
        for d in deltas:
            si = ci + d
            if si >= len(use_scales):
                continue
            c_map = gaussian_filter(work, sigma=use_scales[ci], mode='nearest')
            s_map = gaussian_filter(work, sigma=use_scales[si], mode='nearest')
            fm = cp.abs(c_map - s_map)
            intensity_maps.append(_compress_saliency_feature(fm))
    if intensity_maps:
        I = cp.mean(cp.stack(intensity_maps, axis=0), axis=0)
    else:
        I = cp.zeros_like(work, dtype=cp.float32)
    ori_maps = []
    orientations = [0.0, cp.pi / 4, cp.pi / 2, 3 * cp.pi / 4]
    for sigma in use_scales[:3]:
        sm = gaussian_filter(work, sigma=sigma, mode='nearest')
        step_y = float(pixel_scale_y if pixel_scale_y is not None else pixel_size)
        step_x = float(pixel_scale_x if pixel_scale_x is not None else pixel_size)
        if abs(step_y) < 1e-9:
            step_y = float(pixel_size if pixel_size else 1.0)
        if abs(step_x) < 1e-9:
            step_x = float(pixel_size if pixel_size else 1.0)
        gy, gx = cp.gradient(sm, step_y, step_x)
        mag = cp.sqrt(gx * gx + gy * gy) + 1e-8
        theta = cp.arctan2(gy, gx)
        for o in orientations:
            resp = mag * cp.maximum(cp.cos(2.0 * (theta - o)), 0.0)
            ori_maps.append(_compress_saliency_feature(resp))
    if ori_maps:
        O = cp.mean(cp.stack(ori_maps, axis=0), axis=0)
    else:
        O = cp.zeros_like(work, dtype=cp.float32)
    sal = 0.5 * (I + O)
    if normalize:
        if norm_min is None or norm_max is None:
            norm_min, norm_max = visual_saliency_stat_func(sal)
        if norm_max > norm_min:
            result = (sal - norm_min) / (norm_max - norm_min)
        else:
            result = cp.zeros_like(sal)
        result = cp.clip(result, 0, 1)
    else:
        result = sal
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


class VisualSaliencyAlgorithm(DaskAlgorithm):
    """Visual saliency based on Itti-style conspicuity maps."""
    def process(self, gpu_arr, **params):
        scales = params.get('scales', [2, 4, 8, 16])
        max_scale = max(scales)
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        stats = params.get('global_stats', None)
        stats_ok = isinstance(stats, (tuple, list)) and len(stats) >= 2
        if not stats_ok:
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    gpu_arr, visual_saliency_stat_func,
                    compute_visual_saliency_block,
                    {'scales': scales, 'pixel_size': pixel_size,
                     'pixel_scale_x': pixel_scale_x,
                     'pixel_scale_y': pixel_scale_y, 'normalize': False},
                    downsample_factor=params.get('downsample_factor', None),
                    depth=int(max_scale * 8))
            else:
                stats = (0.0, 1.0)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2):
            stats = (0.0, 1.0)
        return gpu_arr.map_overlap(
            compute_visual_saliency_block, depth=int(max_scale * 8),
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            normalize=True, norm_min=stats[0], norm_max=stats[1])

    def get_default_params(self):
        return {
            'scales': [2, 4, 8, 16], 'pixel_size': 1.0,
            'downsample_factor': None, 'verbose': False,
        }


__all__ = [
    "_compress_saliency_feature", "visual_saliency_stat_func",
    "compute_visual_saliency_block", "VisualSaliencyAlgorithm",
]
