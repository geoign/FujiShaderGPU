"""
FujiShaderGPU/algorithms/_impl_atmospheric_scattering.py

Atmospheric Scattering algorithm implementation.
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import cupy as cp

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    large_radius_threshold, multiscale_response_fields,
    _smooth_for_radius,
)


def compute_atmospheric_scattering_block(block, *, scattering_strength=0.5,
                                        intensity=None, pixel_size=1.0,
                                        pixel_scale_x=None,
                                        pixel_scale_y=None):
    """Shading from atmospheric scattering (simplified Rayleigh scattering)."""
    if intensity is not None:
        scattering_strength = intensity
    nan_mask = cp.isnan(block)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=1, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y)
    slope_mag = cp.sqrt(dx**2 + dy**2)        # tan of the slope angle
    zenith_angle = cp.arctan(slope_mag)       # slope angle in radians
    air_mass = 1.0 / (cp.cos(zenith_angle) + 0.001)
    scattering = 1.0 - cp.exp(-scattering_strength * air_mass)
    ambient = 0.4 + 0.6 * scattering
    # Lambertian hillshade term via the unit surface normal (sign-aware east/
    # north handling, same convention as compute_hillshade_block).  The previous
    # formulation passed the gradient *magnitude* (a tan value) to cos()/sin()
    # as if it were an angle, which wrapped around on steep slopes.
    sign_x = 1.0 if (pixel_scale_x is None or float(pixel_scale_x) >= 0.0) else -1.0
    sign_y = 1.0 if (pixel_scale_y is None or float(pixel_scale_y) >= 0.0) else -1.0
    dz_d_east = dx * sign_x
    dz_d_north = dy * sign_y
    inv_norm = cp.float32(1.0) / cp.sqrt(
        dz_d_east * dz_d_east + dz_d_north * dz_d_north + cp.float32(1.0))
    azimuth_rad = cp.radians(Constants.DEFAULT_AZIMUTH)
    altitude_rad = cp.radians(Constants.DEFAULT_ALTITUDE)
    lx = cp.sin(azimuth_rad) * cp.cos(altitude_rad)
    ly = cp.cos(azimuth_rad) * cp.cos(altitude_rad)
    lz = cp.sin(altitude_rad)
    hillshade = cp.clip(
        (-dz_d_east * lx - dz_d_north * ly + lz) * inv_norm, 0.0, 1.0)
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
    """Atmospheric scattering algorithm."""
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
            is_geo = bool(params.get("is_geographic_dem", False))
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            responses = multiscale_response_fields(
                gpu_arr, [float(r) for r in radii],
                block_fn=compute_atmospheric_scattering_spatial_block, radius_kw="radius",
                depth_for_scale=lambda rr: max(2, int(float(rr) * 2 + 1)),
                is_large=lambda rr: int(round(float(rr))) > thr,
                pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy, is_geographic=is_geo,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                scattering_strength=ss, intensity=intensity)
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
