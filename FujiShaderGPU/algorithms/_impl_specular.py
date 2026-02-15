"""
FujiShaderGPU/algorithms/_impl_specular.py

Specular (鏡面反射効果) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 2)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask, _smooth_for_radius,
)


def compute_specular_block(block, *, roughness_scale=50.0, shininess=20.0,
                          pixel_size=1.0, pixel_scale_x=None, pixel_scale_y=None,
                          roughness_norm_scale=None, geographic_mode=False,
                          light_azimuth=Constants.DEFAULT_AZIMUTH,
                          light_altitude=Constants.DEFAULT_ALTITUDE):
    """鏡面反射効果の計算（Cook-Torranceモデルの簡易版）"""
    nan_mask = cp.isnan(block)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=1, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
    )
    sign_x = 1.0 if (pixel_scale_x is None or float(pixel_scale_x) >= 0.0) else -1.0
    sign_y = 1.0 if (pixel_scale_y is None or float(pixel_scale_y) >= 0.0) else -1.0
    dz_d_east = dx * sign_x
    dz_d_north = dy * sign_y
    normal = cp.stack([-dz_d_east, -dz_d_north, cp.ones_like(dx)], axis=-1)
    normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
    kernel_size = max(3, int(roughness_scale))
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        mean_values = uniform_filter(filled * valid, size=kernel_size, mode='constant')
        mean_weights = uniform_filter(valid, size=kernel_size, mode='constant')
        mean_f = cp.where(mean_weights > 0, mean_values / mean_weights, 0)
        sq_values = uniform_filter((filled**2) * valid, size=kernel_size, mode='constant')
        mean_sq_f = cp.where(mean_weights > 0, sq_values / mean_weights, 0)
    else:
        mean_f = uniform_filter(block, size=kernel_size, mode='constant')
        mean_sq_f = uniform_filter(block**2, size=kernel_size, mode='constant')
    roughness = cp.sqrt(cp.maximum(mean_sq_f - mean_f**2, 0))
    roughness_valid = roughness[~nan_mask] if nan_mask.any() else roughness
    if len(roughness_valid) > 0:
        if roughness_norm_scale is not None and float(roughness_norm_scale) > 1e-9:
            denom = float(roughness_norm_scale)
        else:
            p95_local = float(cp.percentile(roughness_valid, 95))
            denom = p95_local if p95_local > 1e-9 else float(cp.max(roughness_valid))
        if denom > 1e-9:
            roughness = roughness / (roughness + denom)
            roughness = cp.clip(roughness, 0.05, 1.0)
        else:
            roughness = cp.full_like(block, 0.5)
    else:
        roughness = cp.full_like(block, 0.5)
    eff_az = float((light_azimuth + 180.0) % 360.0) if geographic_mode else float(light_azimuth)
    light_az_rad = cp.radians(eff_az)
    light_alt_rad = cp.radians(light_altitude)
    light_dir = cp.array([
        cp.sin(light_az_rad) * cp.cos(light_alt_rad),
        cp.cos(light_az_rad) * cp.cos(light_alt_rad),
        cp.sin(light_alt_rad),
    ])
    view_dir = cp.array([0, 0, 1])
    half_vec = (light_dir + view_dir) / cp.linalg.norm(light_dir + view_dir)
    n_dot_h = cp.sum(normal * half_vec.reshape(1, 1, 3), axis=-1)
    n_dot_h = cp.clip(n_dot_h, 0, 1)
    exponent = shininess * (1.0 - roughness * 0.8)
    specular = cp.power(n_dot_h, exponent)
    gloss_boost = 0.95 + 0.70 * (1.0 - roughness)
    specular = cp.clip(specular * gloss_boost, 0.0, 1.0)
    n_dot_v = cp.clip(normal[..., 2], 0.0, 1.0)
    f0 = cp.float32(0.06)
    fresnel = f0 + (1.0 - f0) * cp.power(1.0 - n_dot_v, 5.0)
    specular = cp.clip(specular * (0.80 + 0.45 * fresnel), 0.0, 1.0)
    specular = specular / (1.0 + 0.35 * specular)
    n_dot_l = cp.sum(normal * light_dir.reshape(1, 1, 3), axis=-1)
    n_dot_l = cp.clip(n_dot_l, 0, 1)
    diffuse = n_dot_l * 0.28
    result = diffuse * 0.36 + specular * 0.64
    result = cp.clip(result, 0, 1)
    result = cp.power(result, 0.88)
    micro = specular - gaussian_filter(specular, sigma=1.1, mode='nearest')
    result = result + 0.10 * micro * (1.0 - 0.6 * roughness)
    result = cp.clip(result, 0, 1)
    result = 0.5 + 0.5 * cp.tanh((result - 0.5) / 0.82)
    result = 0.04 + 0.92 * result
    if geographic_mode:
        result = 1.0 - result
    result = cp.clip(result, 0, 1)
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


def compute_specular_spatial_block(
    block, *, roughness_scale=50.0, shininess=20.0, pixel_size=1.0,
    pixel_scale_x=None, pixel_scale_y=None, roughness_norm_scale=None,
    geographic_mode=False, light_azimuth=Constants.DEFAULT_AZIMUTH,
    light_altitude=Constants.DEFAULT_ALTITUDE, radius=4.0,
):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="specular")
    return compute_specular_block(
        smoothed, roughness_scale=roughness_scale, shininess=shininess,
        pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y, roughness_norm_scale=roughness_norm_scale,
        geographic_mode=geographic_mode,
        light_azimuth=light_azimuth, light_altitude=light_altitude,
    )


class SpecularAlgorithm(DaskAlgorithm):
    """鏡面反射効果アルゴリズム"""
    def process(self, gpu_arr, **params):
        rs = params.get('roughness_scale', 50.0)
        sh = params.get('shininess', 20.0)
        ps = params.get('pixel_size', 1.0)
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        rns = params.get('roughness_norm_scale', None)
        geo = bool(params.get('is_geographic_dem', False))
        laz = params.get('light_azimuth', Constants.DEFAULT_AZIMUTH)
        lal = params.get('light_altitude', Constants.DEFAULT_ALTITUDE)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), ps)
        agg = params.get("agg", "mean")
        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(int(rs), int(float(radius) * 2 + 1))
                responses.append(gpu_arr.map_overlap(
                    compute_specular_spatial_block, depth=depth,
                    boundary='reflect', dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    roughness_scale=rs, shininess=sh, pixel_size=ps,
                    pixel_scale_x=psx, pixel_scale_y=psy,
                    roughness_norm_scale=rns, geographic_mode=geo,
                    light_azimuth=laz, light_altitude=lal, radius=float(radius)))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_specular_block, depth=int(rs), boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            roughness_scale=rs, shininess=sh, pixel_size=ps,
            pixel_scale_x=psx, pixel_scale_y=psy,
            roughness_norm_scale=rns, geographic_mode=geo,
            light_azimuth=laz, light_altitude=lal)

    def get_default_params(self):
        return {
            'roughness_scale': 20.0, 'shininess': 10.0,
            'light_azimuth': Constants.DEFAULT_AZIMUTH,
            'light_altitude': Constants.DEFAULT_ALTITUDE,
            'pixel_size': 1.0, 'roughness_norm_scale': None,
            'mode': 'local', 'radii': None, 'weights': None,
        }


__all__ = [
    "compute_specular_block", "compute_specular_spatial_block", "SpecularAlgorithm",
]
