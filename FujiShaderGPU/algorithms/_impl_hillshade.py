"""
FujiShaderGPU/algorithms/_impl_hillshade.py

Hillshade algorithm implementation.
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import cupy as cp
import numpy as np
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights, _smooth_for_radius,
    large_radius_threshold, multiscale_response_fields,
)


def compute_hillshade_block(block, *, azimuth=Constants.DEFAULT_AZIMUTH,
                           altitude=Constants.DEFAULT_ALTITUDE, z_factor=1.0,
                           pixel_size=1.0, pixel_scale_x=None,
                           pixel_scale_y=None):
    """Hillshade computation for a single block.

    Geotransform orientation is handled uniformly through the signs of
    ``pixel_scale_x`` / ``pixel_scale_y`` (east/north derivative correction), so
    geographic and projected DEMs share the exact same formula and tone scale.
    (The old ``geographic_mode`` azimuth-flip + ``1 - hillshade`` inversion was a
    leftover workaround and produced a different, inverted tone on geographic
    DEMs; it has been removed.)
    """
    nan_mask = cp.isnan(block)
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
    az_rad = cp.radians(float(azimuth))
    light_dir = cp.array([
        cp.sin(az_rad) * cp.cos(altitude_rad),
        cp.cos(az_rad) * cp.cos(altitude_rad),
        cp.sin(altitude_rad),
    ])
    hillshade = cp.sum(normal * light_dir.reshape(1, 1, 3), axis=-1)
    hillshade = cp.clip(hillshade, 0.0, 1.0).astype(cp.float32)
    hillshade = restore_nan(hillshade, nan_mask)
    return hillshade


def compute_hillshade_spatial_block(block, *, azimuth=Constants.DEFAULT_AZIMUTH,
                                   altitude=Constants.DEFAULT_ALTITUDE,
                                   z_factor=1.0, pixel_size=1.0,
                                   pixel_scale_x=None, pixel_scale_y=None,
                                   radius=4.0):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="hillshade")
    return compute_hillshade_block(
        smoothed, azimuth=azimuth, altitude=altitude, z_factor=z_factor,
        pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y,
    )


class HillshadeAlgorithm(DaskAlgorithm):
    """Hillshade algorithm."""
    def process(self, gpu_arr, **params):
        azimuth = params.get('azimuth', Constants.DEFAULT_AZIMUTH)
        altitude = params.get('altitude', Constants.DEFAULT_ALTITUDE)
        z_factor = params.get('z_factor', 1.0)
        if z_factor is None:
            z_factor = 1.0
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        is_geo = bool(params.get('is_geographic_dem', False))
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
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            results = multiscale_response_fields(
                gpu_arr, [float(r) for r in radii],
                block_fn=compute_hillshade_spatial_block, radius_kw="radius",
                depth_for_scale=lambda rr: max(2, int(float(rr) * 2 + 1)),
                is_large=lambda rr: int(round(float(rr))) > thr,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, is_geographic=is_geo,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                azimuth=azimuth, altitude=altitude, z_factor=z_factor)
            stacked = da.stack(results, axis=0)
            if agg == "stack":
                return stacked
            elif agg == "mean":
                if isinstance(weights, (list, tuple)) and len(weights) == len(radii):
                    w = np.asarray(weights, dtype=np.float32)
                    if np.isfinite(w).all() and w.sum() > 0:
                        w = w / w.sum()
                        # Scalar-weighted sum (backend-agnostic): a numpy-backed
                        # weight dask array would multiply cupy blocks by numpy and
                        # raise "Unsupported type numpy.ndarray".  Mirrors
                        # _combine_multiscale_dask.
                        out = results[0] * float(w[0])
                        for i in range(1, len(results)):
                            out = out + results[i] * float(w[i])
                        return out
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
                pixel_scale_y=pixel_scale_y,
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
