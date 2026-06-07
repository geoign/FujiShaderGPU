"""
FujiShaderGPU/algorithms/_impl_openness.py

Openness algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    restore_nan,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
    large_radius_threshold, multiscale_response_fields,
)
from ._global_stats import apply_display_stretch_dask
from ._normalization import robust_unsigned_stretch_stat_func


def compute_openness_vectorized(block: cp.ndarray, *,
                              openness_type: str = 'positive',
                              num_directions: int = 16,
                              max_distance: int = 50,
                              pixel_size: float = 1.0,
                              pixel_scale_x: float = None,
                              pixel_scale_y: float = None) -> cp.ndarray:
    """Openness computation (optimized)."""
    h, w = block.shape
    nan_mask = cp.isnan(block)

    angles = cp.linspace(0, 2 * cp.pi, num_directions, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)

    init_val = -cp.pi/2 if openness_type == 'positive' else cp.pi/2
    max_angles = cp.full((h, w), init_val, dtype=cp.float32)

    distances = cp.unique(cp.linspace(0.1, 1.0, 10) * max_distance).astype(int)
    distances = distances[distances > 0]

    pad_value = Constants.NAN_FILL_VALUE_POSITIVE if openness_type == 'positive' else Constants.NAN_FILL_VALUE_NEGATIVE

    for r in distances:
        offsets = cp.round(r * directions).astype(int)

        for offset in offsets:
            offset_x, offset_y = int(offset[0]), int(offset[1])

            if offset_x == 0 and offset_y == 0:
                continue

            pad_left = max(0, -offset_x)
            pad_right = max(0, offset_x)
            pad_top = max(0, -offset_y)
            pad_bottom = max(0, offset_y)

            if nan_mask.any():
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=pad_value)
            else:
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='edge')

            start_y = pad_top + offset_y
            start_x = pad_left + offset_x
            shifted = padded[start_y:start_y+h, start_x:start_x+w]

            _sx = abs(float(pixel_scale_x)) if pixel_scale_x is not None else float(pixel_size)
            _sy = abs(float(pixel_scale_y)) if pixel_scale_y is not None else float(pixel_size)
            if _sx < 1e-9:
                _sx = float(pixel_size) if pixel_size else 1.0
            if _sy < 1e-9:
                _sy = float(pixel_size) if pixel_size else 1.0
            phys_dx = float(offset_x) * _sx
            phys_dy = float(offset_y) * _sy
            phys_dist = max(float(cp.sqrt(phys_dx ** 2 + phys_dy ** 2)), 1e-9)
            angle = cp.arctan((shifted - block) / phys_dist)

            if openness_type == 'positive':
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.maximum(max_angles, angle), max_angles)
            else:
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.minimum(max_angles, angle), max_angles)

    openness = (cp.pi/2 - max_angles if openness_type == 'positive'
                else cp.pi/2 + max_angles)
    openness = cp.clip(openness / (cp.pi/2), 0, 1)

    result = cp.power(openness, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


def compute_openness_spatial_block(
    block: cp.ndarray,
    *,
    openness_type: str = 'positive',
    num_directions: int = 16,
    max_distance: int = 50,
    pixel_size: float = 1.0,
    pixel_scale_x: float = None,
    pixel_scale_y: float = None,
) -> cp.ndarray:
    ds_factor = _radius_to_downsample_factor(
        float(max_distance), block_shape=block.shape,
        pixel_size=pixel_size, algorithm_name="openness",
    )
    if ds_factor <= 1:
        return compute_openness_vectorized(
            block, openness_type=openness_type, num_directions=num_directions,
            max_distance=max_distance, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
        )
    small = _downsample_nan_aware(block, ds_factor)
    ds_psx = float(abs(float(pixel_scale_x)) * ds_factor) if pixel_scale_x is not None else None
    ds_psy = float(abs(float(pixel_scale_y)) * ds_factor) if pixel_scale_y is not None else None
    result_small = compute_openness_vectorized(
        small, openness_type=openness_type, num_directions=num_directions,
        max_distance=max(2, int(round(float(max_distance) / float(ds_factor)))),
        pixel_size=float(pixel_size) * float(ds_factor),
        pixel_scale_x=ds_psx, pixel_scale_y=ds_psy,
    )
    return _upsample_to_shape(result_small, block.shape)


class OpennessAlgorithm(DaskAlgorithm):
    """Openness algorithm (simplified vectorized version)."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        openness_type = params.get('openness_type', 'positive')
        num_directions = params.get('num_directions', 16)
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), pixel_size,
        )
        agg = params.get("agg", "mean")

        if mode == "spatial":
            is_geo = bool(params.get("is_geographic_dem", False))
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            # Large radius from a coarsened DEM (no large per-chunk halo).
            responses = multiscale_response_fields(
                gpu_arr, [float(int(max(2, round(float(r))))) for r in radii],
                block_fn=compute_openness_spatial_block, radius_kw="max_distance",
                depth_for_scale=lambda md: int(md) + 1,
                is_large=lambda md: int(md) > thr,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, is_geographic=is_geo,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                openness_type=openness_type, num_directions=num_directions)
            result = _combine_multiscale_dask(responses, weights=weights, agg=agg)
        else:
            result = gpu_arr.map_overlap(
                compute_openness_vectorized,
                depth=max_distance+1,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                openness_type=openness_type, num_directions=num_directions,
                max_distance=max_distance, pixel_size=pixel_size,
                pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            )
        # Data-driven [p1, p99] -> [0, 1] contrast stretch (openness concentrates
        # in a narrow high band).  No-op unless 'global_stats' was injected.
        return apply_display_stretch_dask(result, params.get("global_stats"))

    def get_default_params(self) -> dict:
        return {
            'openness_type': 'positive',
            'num_directions': 16,
            'max_distance': 50,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }


__all__ = [
    "compute_openness_vectorized",
    "compute_openness_spatial_block",
    "OpennessAlgorithm",
]
