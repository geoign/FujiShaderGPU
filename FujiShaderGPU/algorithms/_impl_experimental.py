"""
FujiShaderGPU/algorithms/_impl_experimental.py

実験的アルゴリズム実装：
  - Scale-Space Surprise
  - Multi-Light Uncertainty
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
from typing import List
import cupy as cp
import dask.array as da

from ._base import DaskAlgorithm
from ._global_stats import compute_global_stats
from ._nan_utils import (
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
)
from ._normalization import NORMAL_PERCENTILE, OVERFLOW_LIMIT
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)

###############################################################################
# Scale-Space Surprise
###############################################################################

def compute_scale_space_surprise_block(block, *, scales, enhancement=2.0,
                                      normalize=True, norm_min=None, norm_scale=None):
    """Scale-Space Surprise Map: スケール間での特徴変化量を強調"""
    nan_mask = cp.isnan(block)
    surprise = kernel_scale_space_surprise(
        block, scales=scales, enhancement=enhancement,
        normalize=False, nan_mask=nan_mask)
    if normalize:
        if norm_scale is None:
            _min, _scale = scale_space_surprise_stat_func(surprise)
            norm_min = _min if norm_min is None else norm_min
            norm_scale = _scale
        scale = float(norm_scale)
        if scale > 1e-9:
            offset = 0.0 if norm_min is None else float(norm_min)
            surprise = cp.clip((surprise - offset) / scale, 0, OVERFLOW_LIMIT)
            surprise = cp.power(surprise, 1.0 / max(1e-3, enhancement))
        else:
            surprise = cp.zeros_like(surprise)
        surprise = cp.where(nan_mask, cp.nan, surprise)
    return surprise.astype(cp.float32)


def scale_space_surprise_stat_func(data):
    """Global unsigned scale: p80 maps to +1."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    scale = float(cp.percentile(cp.maximum(valid_data, 0.0), NORMAL_PERCENTILE))
    return (0.0, scale if scale > 1e-9 else 1.0)


class ScaleSpaceSurpriseAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr, **params):
        scales = params.get('scales', [1.0, 2.0, 4.0, 8.0, 16.0])
        enhancement = float(params.get('enhancement', 2.0))
        normalize = bool(params.get('normalize', True))
        depth = int(max(1, cp.ceil(max(scales) * 3).item())) + 1
        stats = params.get('global_stats', None)
        stats_ok = isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9
        if normalize and not stats_ok:
            stats = compute_global_stats(
                gpu_arr,
                scale_space_surprise_stat_func,
                compute_scale_space_surprise_block,
                {'scales': scales, 'enhancement': enhancement, 'normalize': False},
                downsample_factor=params.get('downsample_factor', None),
                depth=depth,
                algorithm_name='scale_space_surprise',
            )
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
            stats = (0.0, 1.0)
        return gpu_arr.map_overlap(
            compute_scale_space_surprise_block, depth=depth,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, enhancement=enhancement, normalize=normalize,
            norm_min=stats[0], norm_scale=stats[1])

    def get_default_params(self):
        return {
            'scales': [1.0, 2.0, 4.0, 8.0, 16.0],
            'enhancement': 2.0, 'normalize': True,
        }


###############################################################################
# Multi-Light Uncertainty
###############################################################################

def compute_multi_light_uncertainty_block(block, *, azimuths,
                                         altitude=45.0, z_factor=1.0,
                                         uncertainty_weight=0.7,
                                         pixel_size=1.0,
                                         pixel_scale_x=None,
                                         pixel_scale_y=None):
    """Multi-light Uncertainty Shading"""
    nan_mask = cp.isnan(block)
    return kernel_multi_light_uncertainty(
        block, azimuths=azimuths, altitude=altitude, z_factor=z_factor,
        uncertainty_weight=uncertainty_weight, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
        nan_mask=nan_mask)


def compute_multi_light_uncertainty_spatial_block(
    block, *, azimuths, altitude=45.0, z_factor=1.0,
    uncertainty_weight=0.7, pixel_size=1.0,
    pixel_scale_x=None, pixel_scale_y=None, radius=4.0):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size,
                                  algorithm_name="multi_light_uncertainty")
    return compute_multi_light_uncertainty_block(
        smoothed, azimuths=azimuths, altitude=altitude, z_factor=z_factor,
        uncertainty_weight=uncertainty_weight, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y)


class MultiLightUncertaintyAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr, **params):
        azimuths = params.get('azimuths', [315.0, 45.0, 135.0, 225.0])
        altitude = float(params.get('altitude', 45.0))
        z_factor = float(params.get('z_factor', 1.0))
        uw = float(params.get('uncertainty_weight', 0.7))
        ps = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), ps)
        agg = params.get("agg", "mean")
        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                responses.append(gpu_arr.map_overlap(
                    compute_multi_light_uncertainty_spatial_block, depth=depth,
                    boundary='reflect', dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    azimuths=azimuths, altitude=altitude, z_factor=z_factor,
                    uncertainty_weight=uw, pixel_size=ps,
                    pixel_scale_x=psx, pixel_scale_y=psy,
                    radius=float(radius)))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_multi_light_uncertainty_block, depth=1,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            azimuths=azimuths, altitude=altitude, z_factor=z_factor,
            uncertainty_weight=uw, pixel_size=ps,
            pixel_scale_x=psx, pixel_scale_y=psy)

    def get_default_params(self):
        return {
            'azimuths': [315.0, 45.0, 135.0, 225.0],
            'altitude': 45.0, 'z_factor': 1.0,
            'uncertainty_weight': 0.7,
            'mode': 'local', 'radii': None, 'weights': None,
        }


__all__ = [
    "compute_scale_space_surprise_block", "scale_space_surprise_stat_func",
    "ScaleSpaceSurpriseAlgorithm",
    "compute_multi_light_uncertainty_block",
    "compute_multi_light_uncertainty_spatial_block",
    "MultiLightUncertaintyAlgorithm",
]
