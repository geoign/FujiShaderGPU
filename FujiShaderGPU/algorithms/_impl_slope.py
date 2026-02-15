"""
FujiShaderGPU/algorithms/_impl_slope.py

Slope (傾斜) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da

from ._base import DaskAlgorithm
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
)


def compute_slope_block(block, *, unit='degree', pixel_size=1.0,
                       pixel_scale_x=None, pixel_scale_y=None):
    """1ブロックに対する傾斜計算"""
    nan_mask = cp.isnan(block)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=1, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
    )
    slope_rad = cp.arctan(cp.sqrt(dx**2 + dy**2))
    if unit == 'degree':
        slope = cp.degrees(slope_rad)
    elif unit == 'percent':
        slope = cp.tan(slope_rad) * 100
    else:
        slope = slope_rad
    slope = restore_nan(slope, nan_mask)
    return slope.astype(cp.float32)


def compute_slope_spatial_block(block, *, unit="degree", pixel_size=1.0,
                               pixel_scale_x=None, pixel_scale_y=None,
                               radius=4.0):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="slope")
    return compute_slope_block(
        smoothed, unit=unit, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
    )


class SlopeAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr, **params):
        unit = params.get('unit', 'degree')
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), pixel_size)
        agg = params.get("agg", "mean")
        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                responses.append(gpu_arr.map_overlap(
                    compute_slope_spatial_block, depth=depth,
                    boundary="reflect", dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    unit=unit, pixel_size=pixel_size,
                    pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                    radius=float(radius)))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_slope_block, depth=1, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            unit=unit, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y)

    def get_default_params(self):
        return {
            'unit': 'degree', 'pixel_size': 1.0,
            'mode': 'local', 'radii': None, 'weights': None,
        }


__all__ = [
    "compute_slope_block", "compute_slope_spatial_block", "SlopeAlgorithm",
]
