"""
FujiShaderGPU/algorithms/_impl_atmospheric_scattering.py

Atmospheric Scattering (大気散乱効果) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
from typing import Optional
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
)


def compute_atmospheric_scattering_block(block, *, scattering_strength=0.5,
                                        intensity=None, pixel_size=1.0,
                                        pixel_scale_x=None,
                                        pixel_scale_y=None):
    """大気散乱によるシェーディング（Rayleigh散乱の簡易版）"""
    if intensity is not None:
        scattering_strength = intensity
    nan_mask = cp.isnan(block)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=1, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y)
    slope = cp.sqrt(dx**2 + dy**2)
    zenith_angle = cp.arctan(slope)
    air_mass = 1.0 / (cp.cos(zenith_angle) + 0.001)
    scattering = 1.0 - cp.exp(-scattering_strength * air_mass)
    ambient = 0.4 + 0.6 * scattering
    azimuth_rad = cp.radians(Constants.DEFAULT_AZIMUTH)
    altitude_rad = cp.radians(Constants.DEFAULT_ALTITUDE)
    aspect = cp.arctan2(-dy, dx)
    hillshade = (cp.cos(altitude_rad) * cp.cos(slope) +
                 cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad))
    result = ambient * 0.3 + hillshade * 0.7
    result = cp.clip(result, 0, 1)
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


def compute_atmospheric_scattering_spatial_block(
    block, *, scattering_strength=0.5, intensity=None, pixel_size=1.0,
    pixel_scale_x=None, pixel_scale_y=None, radius=4.0):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size,
                                  algorithm_name="atmospheric_scattering")
    return compute_atmospheric_scattering_block(
        smoothed, scattering_strength=scattering_strength,
        intensity=intensity, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y)


class AtmosphericScatteringAlgorithm(DaskAlgorithm):
    """大気散乱効果アルゴリズム"""
    def process(self, gpu_arr, **params):
        ss = params.get('scattering_strength', 0.5)
        intensity = params.get('intensity', None)
        ps = params.get('pixel_size', 1.0)
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
                    compute_atmospheric_scattering_spatial_block, depth=depth,
                    boundary='reflect', dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    scattering_strength=ss, intensity=intensity,
                    pixel_size=ps, pixel_scale_x=psx,
                    pixel_scale_y=psy, radius=float(radius)))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_atmospheric_scattering_block, depth=1,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scattering_strength=ss, intensity=intensity,
            pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy)

    def get_default_params(self):
        return {
            'scattering_strength': 0.5, 'pixel_size': 1.0,
            'mode': 'local', 'radii': None, 'weights': None,
        }


__all__ = [
    "compute_atmospheric_scattering_block",
    "compute_atmospheric_scattering_spatial_block",
    "AtmosphericScatteringAlgorithm",
]
