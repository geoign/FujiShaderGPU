"""
FujiShaderGPU/algorithms/_impl_ambient_occlusion.py

Ambient Occlusion algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    restore_nan,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
    large_radius_threshold, coarsen_factor_for_shape, coarse_large_radius_response,
)
from ._global_stats import apply_display_stretch_dask
from ._normalization import robust_unsigned_stretch_stat_func


def compute_ambient_occlusion_block(block: cp.ndarray, *,
                                    num_samples: int = 16,
                                    radius: float = 10.0,
                                    intensity: float = 1.0,
                                    pixel_size: float = 1.0,
                                    pixel_scale_x: float = None,
                                    pixel_scale_y: float = None) -> cp.ndarray:
    """Terrain version of screen-space ambient occlusion (SAO), vectorized with normals."""
    h, w = block.shape
    nan_mask = cp.isnan(block)

    angles = cp.linspace(0, 2 * cp.pi, num_samples, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)

    r_factors = cp.array([0.25, 0.5, 0.75, 1.0])

    occlusion_total = cp.zeros((h, w), dtype=cp.float32)
    sample_count = cp.zeros((h, w), dtype=cp.float32)

    for r_factor in r_factors:
        r = radius * r_factor

        dx_all = cp.round(r * directions[:, 0]).astype(int)
        dy_all = cp.round(r * directions[:, 1]).astype(int)

        for i in range(num_samples):
            dx = int(dx_all[i])
            dy = int(dy_all[i])

            if dx == 0 and dy == 0:
                continue

            pad_left = max(0, -dx)
            pad_right = max(0, dx)
            pad_top = max(0, -dy)
            pad_bottom = max(0, dy)

            padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='edge')

            start_y = pad_top + dy
            start_x = pad_left + dx
            shifted = padded[start_y:start_y+h, start_x:start_x+w]

            height_diff = shifted - block
            _sx = abs(float(pixel_scale_x)) if pixel_scale_x is not None else float(pixel_size)
            _sy = abs(float(pixel_scale_y)) if pixel_scale_y is not None else float(pixel_size)
            if _sx < 1e-9:
                _sx = float(pixel_size) if pixel_size else 1.0
            if _sy < 1e-9:
                _sy = float(pixel_size) if pixel_size else 1.0
            phys_dx_ao = float(dx) * _sx
            phys_dy_ao = float(dy) * _sy
            distance = max(float(cp.sqrt(phys_dx_ao ** 2 + phys_dy_ao ** 2)), 1e-9)
            occlusion_angle = cp.arctan(height_diff / distance)

            max_angle = cp.pi / 4
            occlusion = cp.maximum(0, occlusion_angle) / max_angle
            occlusion = cp.minimum(occlusion, 1.0)

            distance_factor = 1.0 - (r_factor * 0.3)

            valid = ~(cp.isnan(shifted) | nan_mask)
            occlusion_total += cp.where(valid, occlusion * distance_factor, 0)
            sample_count += cp.where(valid, 1.0, 0)

    sample_count = cp.maximum(sample_count, 1.0)
    mean_occlusion = occlusion_total / sample_count

    ao = 1.0 - mean_occlusion * intensity
    ao = cp.clip(ao, 0, 1)

    if nan_mask.any():
        filled_ao = cp.where(nan_mask, 1.0, ao)
        ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
    else:
        ao = gaussian_filter(ao, sigma=1.0, mode='nearest')

    result = cp.power(ao, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


def compute_ambient_occlusion_spatial_block(
    block: cp.ndarray,
    *,
    num_samples: int = 16,
    radius: float = 10.0,
    intensity: float = 1.0,
    pixel_size: float = 1.0,
    pixel_scale_x: float = None,
    pixel_scale_y: float = None,
) -> cp.ndarray:
    ds_factor = _radius_to_downsample_factor(
        float(radius),
        block_shape=block.shape,
        pixel_size=pixel_size,
        algorithm_name="ambient_occlusion",
    )
    if ds_factor <= 1:
        return compute_ambient_occlusion_block(
            block,
            num_samples=num_samples, radius=radius, intensity=intensity,
            pixel_size=pixel_size, pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
        )
    small = _downsample_nan_aware(block, ds_factor)
    ds_psx_ao = float(abs(float(pixel_scale_x)) * ds_factor) if pixel_scale_x is not None else None
    ds_psy_ao = float(abs(float(pixel_scale_y)) * ds_factor) if pixel_scale_y is not None else None
    result_small = compute_ambient_occlusion_block(
        small,
        num_samples=num_samples,
        radius=max(1.0, float(radius) / float(ds_factor)),
        intensity=intensity,
        pixel_size=float(pixel_size) * float(ds_factor),
        pixel_scale_x=ds_psx_ao, pixel_scale_y=ds_psy_ao,
    )
    return _upsample_to_shape(result_small, block.shape)


class AmbientOcclusionAlgorithm(DaskAlgorithm):
    """Ambient occlusion algorithm."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        num_samples = params.get('num_samples', 16)
        radius = params.get('radius', 10.0)
        intensity = params.get('intensity', 1.0)
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
            F = coarsen_factor_for_shape(gpu_arr.shape) if not is_geo else 1
            _depth = lambda rr: int(rr) + 1
            cache = {}
            responses = []
            for r in radii:
                r_use = float(max(1, int(round(float(r)))))
                if F > 1 and int(r_use) > thr:
                    responses.append(coarse_large_radius_response(
                        gpu_arr, block_fn=compute_ambient_occlusion_spatial_block,
                        radius_kw="radius", radius=r_use, factor=F,
                        depth_for_radius=_depth, pixel_size=pixel_size,
                        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                        coarse_cache=cache,
                        num_samples=num_samples, intensity=intensity,
                    ))
                else:
                    responses.append(
                        gpu_arr.map_overlap(
                            compute_ambient_occlusion_spatial_block,
                            depth=_depth(r_use),
                            boundary='reflect',
                            dtype=cp.float32,
                            meta=cp.empty((0, 0), dtype=cp.float32),
                            num_samples=num_samples, radius=r_use,
                            intensity=intensity, pixel_size=pixel_size,
                            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                        )
                    )
            result = _combine_multiscale_dask(responses, weights=weights, agg=agg)
        else:
            result = gpu_arr.map_overlap(
                compute_ambient_occlusion_block,
                depth=int(radius + 1),
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                num_samples=num_samples, radius=radius, intensity=intensity,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            )
        # Data-driven [p1, p99] -> [0, 1] contrast stretch (AO concentrates in a
        # narrow high band; without this most code values are unused).  No-op
        # unless the global-stats pre-pass injected 'global_stats'.
        return apply_display_stretch_dask(result, params.get("global_stats"))

    def get_default_params(self) -> dict:
        return {
            'num_samples': 16,
            'radius': 10.0,
            'intensity': 1.0,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }


__all__ = [
    "compute_ambient_occlusion_block",
    "compute_ambient_occlusion_spatial_block",
    "AmbientOcclusionAlgorithm",
]
