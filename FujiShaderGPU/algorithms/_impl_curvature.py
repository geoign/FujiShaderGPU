"""
FujiShaderGPU/algorithms/_impl_curvature.py

Curvature (曲率) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 3)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    restore_nan,
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
)


def compute_curvature_block(block, *, curvature_type='mean', pixel_size=1.0,
                          pixel_scale_x=None, pixel_scale_y=None):
    """曲率計算（平均曲率、ガウス曲率、平面・縦面曲率）"""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    step_y = float(pixel_scale_y if pixel_scale_y is not None else pixel_size)
    step_x = float(pixel_scale_x if pixel_scale_x is not None else pixel_size)
    if abs(step_y) < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if abs(step_x) < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    dy, dx = cp.gradient(filled, step_y, step_x, edge_order=2)
    dyy, dyx = cp.gradient(dy, step_y, step_x, edge_order=2)
    dxy, dxx = cp.gradient(dx, step_y, step_x, edge_order=2)
    if curvature_type == 'mean':
        p, q, r = dx, dy, dxx
        s = (dxy + dyx) / 2
        t = dyy
        denominator = cp.power(1 + p**2 + q**2, 1.5)
        numerator = (1 + q**2) * r - 2 * p * q * s + (1 + p**2) * t
        curvature = -numerator / (2 * denominator + 1e-10)
    elif curvature_type == 'gaussian':
        curvature = (dxx * dyy - dxy**2) / cp.power(1 + dx**2 + dy**2, 2)
    elif curvature_type == 'planform':
        curvature = (-2 * (dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) /
                    (cp.power(dx**2 + dy**2, 1.5) + 1e-10))
    else:  # profile
        curvature = (-2 * (dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) /
                    ((dx**2 + dy**2) * cp.power(1 + dx**2 + dy**2, 0.5) + 1e-10))
    curvature_normalized = cp.tanh(curvature * 100)
    result = (curvature_normalized + 1) / 2
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


class CurvatureAlgorithm(DaskAlgorithm):
    """曲率アルゴリズム"""
    def process(self, gpu_arr, **params):
        ct = params.get('curvature_type', 'mean')
        ps = params.get('pixel_size', 1.0)
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), ps)
        agg = params.get("agg", "mean")
        if mode == "spatial":
            def _curv_spatial(block, *, radius, curvature_type, pixel_size):
                smoothed = _smooth_for_radius(block, radius,
                    pixel_size=pixel_size, algorithm_name="curvature")
                return compute_curvature_block(smoothed,
                    curvature_type=curvature_type, pixel_size=pixel_size)
            responses = []
            for radius in radii:
                depth = max(3, int(float(radius) * 2 + 2))
                responses.append(gpu_arr.map_overlap(
                    _curv_spatial, depth=depth, boundary='reflect',
                    dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
                    radius=float(radius), curvature_type=ct,
                    pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_curvature_block, depth=2, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            curvature_type=ct, pixel_size=ps,
            pixel_scale_x=psx, pixel_scale_y=psy)

    def get_default_params(self):
        return {
            'curvature_type': 'mean', 'pixel_size': 1.0,
            'mode': 'local', 'radii': None, 'weights': None,
        }


__all__ = ["compute_curvature_block", "CurvatureAlgorithm"]
