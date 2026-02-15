"""
FujiShaderGPU/algorithms/_impl_hillshade.py

Hillshade アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import cupy as cp
import numpy as np
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights, _smooth_for_radius,
)


def compute_hillshade_block(block, *, azimuth=Constants.DEFAULT_AZIMUTH,
                           altitude=Constants.DEFAULT_ALTITUDE, z_factor=1.0,
                           pixel_size=1.0, pixel_scale_x=None,
                           pixel_scale_y=None, geographic_mode=False):
    """1ブロックに対するHillshade計算"""
    nan_mask = cp.isnan(block)
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=z_factor, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
    )
    sign_x = 1.0 if (pixel_scale_x is None or float(pixel_scale_x) >= 0.0) else -1.0
    sign_y = 1.0 if (pixel_scale_y is None or float(pixel_scale_y) >= 0.0) else -1.0
    dz_d_east = dx * sign_x
    dz_d_north = dy * sign_y
    normal = cp.stack([-dz_d_east, -dz_d_north, cp.ones_like(dx)], axis=-1)
    normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
    effective_azimuth = float((azimuth + 180.0) % 360.0) if geographic_mode else float(azimuth)
    eff_az_rad = cp.radians(effective_azimuth)
    light_dir = cp.array([
        cp.sin(eff_az_rad) * cp.cos(altitude_rad),
        cp.cos(eff_az_rad) * cp.cos(altitude_rad),
        cp.sin(altitude_rad),
    ])
    hillshade = cp.sum(normal * light_dir.reshape(1, 1, 3), axis=-1)
    hillshade = cp.clip(hillshade, 0.0, 1.0).astype(cp.float32)
    if geographic_mode:
        hillshade = 1.0 - hillshade
    hillshade = restore_nan(hillshade, nan_mask)
    return hillshade


def compute_hillshade_spatial_block(block, *, azimuth=Constants.DEFAULT_AZIMUTH,
                                   altitude=Constants.DEFAULT_ALTITUDE,
                                   z_factor=1.0, pixel_size=1.0,
                                   pixel_scale_x=None, pixel_scale_y=None,
                                   geographic_mode=False, radius=4.0):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="hillshade")
    return compute_hillshade_block(
        smoothed, azimuth=azimuth, altitude=altitude, z_factor=z_factor,
        pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y, geographic_mode=geographic_mode,
    )


class HillshadeAlgorithm(DaskAlgorithm):
    """Hillshadeアルゴリズム"""
    def process(self, gpu_arr, **params):
        azimuth = params.get('azimuth', Constants.DEFAULT_AZIMUTH)
        altitude = params.get('altitude', Constants.DEFAULT_ALTITUDE)
        z_factor = params.get('z_factor', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        geographic_mode = bool(params.get('is_geographic_dem', False))
        multiscale = params.get('multiscale', False)
        radii = params.get('radii', [1])
        weights = params.get('weights', None)
        agg = params.get('agg', 'mean')
        mode = str(params.get("mode", "local")).lower()
        if mode == "spatial":
            radii, auto_weights = _resolve_spatial_radii_weights(radii, weights, pixel_size)
            if weights is None:
                weights = auto_weights
            multiscale = True
        else:
            if not isinstance(radii, (list, tuple)) or len(radii) == 0:
                radii = [1]
            radii = [max(1.0, float(r)) for r in radii]
            multiscale = bool(multiscale or len(radii) > 1)
        if mode == "spatial" or (multiscale and len(radii) > 1):
            results = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                hs = gpu_arr.map_overlap(
                    compute_hillshade_spatial_block, depth=depth,
                    boundary='reflect', dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    azimuth=azimuth, altitude=altitude, z_factor=z_factor,
                    pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                    pixel_scale_y=pixel_scale_y, geographic_mode=geographic_mode,
                    radius=float(radius),
                )
                results.append(hs)
            stacked = da.stack(results, axis=0)
            if agg == "stack":
                return stacked
            elif agg == "mean":
                if isinstance(weights, (list, tuple)) and len(weights) == len(radii):
                    w = np.asarray(weights, dtype=np.float32)
                    if np.isfinite(w).all() and w.sum() > 0:
                        w = w / w.sum()
                        w_da = da.from_array(w.astype(np.float32), chunks=(len(radii),))
                        return da.sum(stacked * w_da[:, None, None], axis=0)
                return da.mean(stacked, axis=0)
            elif agg == "min":
                return da.min(stacked, axis=0)
            elif agg == "max":
                return da.max(stacked, axis=0)
            else:
                return da.mean(stacked, axis=0)
        else:
            return gpu_arr.map_overlap(
                compute_hillshade_block, depth=1, boundary='reflect',
                dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
                azimuth=azimuth, altitude=altitude, z_factor=z_factor,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, geographic_mode=geographic_mode,
            )

    def get_default_params(self):
        return {
            'azimuth': Constants.DEFAULT_AZIMUTH,
            'altitude': Constants.DEFAULT_ALTITUDE,
            'z_factor': 1.0, 'pixel_size': 1.0,
            'multiscale': False, 'radii': None,
            'weights': None, 'agg': 'mean', 'mode': 'local',
        }


__all__ = [
    "compute_hillshade_block", "compute_hillshade_spatial_block",
    "HillshadeAlgorithm",
]
