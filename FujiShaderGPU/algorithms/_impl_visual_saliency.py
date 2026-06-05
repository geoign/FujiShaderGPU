"""
FujiShaderGPU/algorithms/_impl_visual_saliency.py

Visual Saliency algorithm implementation.
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
from typing import List, Tuple
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm
from ._nan_utils import restore_nan, resolve_block_weights
from ._global_stats import compute_global_stats
from ._normalization import NORMAL_PERCENTILE


def _weighted_mean_maps(maps, weights, ref):
    """Mean of per-scale feature maps, weighted when ``weights`` is given.

    ``weights`` is a list of per-map scalar weights aligned with ``maps``; None
    or empty falls back to a plain mean (original behavior).  ``ref`` supplies
    the fallback shape when ``maps`` is empty.
    """
    if not maps:
        return cp.zeros_like(ref, dtype=cp.float32)
    if not weights:
        return cp.mean(cp.stack(maps, axis=0), axis=0)
    w = cp.asarray(weights, dtype=cp.float32)
    s = float(w.sum())
    if s <= 1e-12:
        return cp.mean(cp.stack(maps, axis=0), axis=0)
    w = (w / s).reshape((-1,) + (1,) * maps[0].ndim)
    return cp.sum(cp.stack(maps, axis=0) * w, axis=0)


def _compress_saliency_feature(feature):
    """Tile-stable feature compression without block-global normalization."""
    return cp.log1p(cp.clip(feature, 0.0, None)).astype(cp.float32)


def visual_saliency_stat_func(data):
    """Global unsigned scale: p80 maps to +1."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    scale = float(cp.percentile(cp.maximum(valid_data, 0.0), NORMAL_PERCENTILE))
    return (0.0, scale if scale > 1e-9 else 1.0)


def compute_visual_saliency_block(block, *, scales=[2, 4, 8, 16], radii=None,
                                 pixel_size=1.0, pixel_scale_x=None,
                                 pixel_scale_y=None, normalize=True,
                                 norm_min=None, norm_scale=None, weights=None):
    """Itti-style saliency (intensity + orientation conspicuity) for DEM.

    The unified ``--weights`` (length-matching the conspicuity scales) weights
    each scale's contribution to the intensity and orientation conspicuity means;
    absent/mismatched weights keep the original equal averaging.
    """
    if radii:  # unified --radii feeds the conspicuity scales
        scales = [float(r) for r in radii]
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
    # Per-scale weights (unified --weights); None -> equal averaging.
    wvec = resolve_block_weights(weights, len(use_scales))
    c_indices = [0, 1]
    deltas = [2, 3]
    intensity_maps = []
    intensity_w = []
    for ci in c_indices:
        for d in deltas:
            si = ci + d
            if si >= len(use_scales):
                continue
            c_map = gaussian_filter(work, sigma=use_scales[ci], mode='nearest')
            s_map = gaussian_filter(work, sigma=use_scales[si], mode='nearest')
            fm = cp.abs(c_map - s_map)
            intensity_maps.append(_compress_saliency_feature(fm))
            if wvec is not None:
                intensity_w.append(float(wvec[ci]))
    I = _weighted_mean_maps(intensity_maps, intensity_w if wvec is not None else None, work)
    ori_maps = []
    ori_w = []
    orientations = [0.0, cp.pi / 4, cp.pi / 2, 3 * cp.pi / 4]
    for j, sigma in enumerate(use_scales[:3]):
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
            if wvec is not None:
                ori_w.append(float(wvec[j]))
    O = _weighted_mean_maps(ori_maps, ori_w if wvec is not None else None, work)
    sal = 0.5 * (I + O)
    if normalize:
        if norm_min is None or norm_scale is None:
            norm_min, norm_scale = visual_saliency_stat_func(sal)
        if norm_scale > 1e-9:
            result = (sal - norm_min) / norm_scale
        else:
            result = cp.zeros_like(sal)
        result = cp.maximum(result, 0.0)  # p99 -> 1.0; tail passes through unclipped
    else:
        result = sal
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


class VisualSaliencyAlgorithm(DaskAlgorithm):
    """Visual saliency based on Itti-style conspicuity maps."""
    def process(self, gpu_arr, **params):
        radii = params.get('radii')
        scales = [float(r) for r in radii] if radii else params.get('scales', [2, 4, 8, 16])
        weights = params.get('weights', None)
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
                     'pixel_scale_y': pixel_scale_y, 'normalize': False,
                     'weights': weights},
                    downsample_factor=params.get('downsample_factor', None),
                    depth=int(max_scale * 5))
            else:
                stats = (0.0, 1.0)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2):
            stats = (0.0, 1.0)
        return gpu_arr.map_overlap(
            compute_visual_saliency_block, depth=int(max_scale * 5),
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            normalize=True, norm_min=stats[0], norm_scale=stats[1], weights=weights)

    def get_default_params(self):
        return {
            'scales': [2, 4, 8, 16], 'pixel_size': 1.0,
            'downsample_factor': None, 'verbose': False,
        }


__all__ = [
    "_compress_saliency_feature", "visual_saliency_stat_func",
    "compute_visual_saliency_block", "VisualSaliencyAlgorithm",
]
