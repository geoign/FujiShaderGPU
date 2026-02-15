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
from ._nan_utils import (
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
)
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)

###############################################################################
# Scale-Space Surprise
###############################################################################

def compute_scale_space_surprise_block(block, *, scales, enhancement=2.0,
                                      normalize=True):
    """Scale-Space Surprise Map: スケール間での特徴変化量を強調"""
    nan_mask = cp.isnan(block)
    return kernel_scale_space_surprise(
        block, scales=scales, enhancement=enhancement,
        normalize=normalize, nan_mask=nan_mask)


class ScaleSpaceSurpriseAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr, **params):
        scales = params.get('scales', [1.0, 2.0, 4.0, 8.0, 16.0])
        enhancement = float(params.get('enhancement', 2.0))
        normalize = bool(params.get('normalize', True))
        depth = int(max(1, cp.ceil(max(scales) * 3).item())) + 1
        return gpu_arr.map_overlap(
            compute_scale_space_surprise_block, depth=depth,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, enhancement=enhancement, normalize=normalize)

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
    "compute_scale_space_surprise_block", "ScaleSpaceSurpriseAlgorithm",
    "compute_multi_light_uncertainty_block",
    "compute_multi_light_uncertainty_spatial_block",
    "MultiLightUncertaintyAlgorithm",
]
