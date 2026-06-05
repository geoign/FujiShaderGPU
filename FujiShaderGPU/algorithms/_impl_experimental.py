"""
FujiShaderGPU/algorithms/_impl_experimental.py

Experimental algorithm implementations:
  - Scale-Space Surprise
  - Multi-Light Uncertainty
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import logging
from typing import List
import cupy as cp
import dask.array as da

from ._base import DaskAlgorithm, Constants
from ._global_stats import compute_global_stats
from ._nan_utils import (
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius,
    large_radius_threshold, coarsen_factor_for_shape, coarse_large_radius_response,
)
from ._normalization import NORMAL_PERCENTILE
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)

logger = logging.getLogger(__name__)

###############################################################################
# Scale-Space Surprise
###############################################################################

def compute_scale_space_surprise_block(block, *, scales=None, radii=None, enhancement=2.0,
                                      normalize=True, norm_min=None, norm_scale=None,
                                      weights=None):
    """Scale-Space Surprise Map: emphasize the amount of feature change across scales."""
    if radii:  # unified --radii feeds the scale-space scales
        scales = [float(r) for r in radii]
    if not scales:
        scales = [1.0, 2.0, 4.0, 8.0, 16.0]
    nan_mask = cp.isnan(block)
    surprise = kernel_scale_space_surprise(
        block, scales=scales, enhancement=enhancement,
        normalize=False, nan_mask=nan_mask, weights=weights)
    if normalize:
        if norm_scale is None:
            _min, _scale = scale_space_surprise_stat_func(surprise)
            norm_min = _min if norm_min is None else norm_min
            norm_scale = _scale
        scale = float(norm_scale)
        if scale > 1e-9:
            offset = 0.0 if norm_min is None else float(norm_min)
            # Normalize (p99 -> 1.0) then apply the gamma-style enhancement.  The
            # non-negative base maps [0,1] -> [0,1] (p99 -> 1.0); the tail passes
            # through just past 1.0 unclipped.
            base = cp.clip((surprise - offset) / scale, 0.0, None)
            surprise = cp.power(base, 1.0 / max(1e-3, enhancement))
        else:
            surprise = cp.zeros_like(surprise)
        surprise = cp.where(nan_mask, cp.nan, surprise)
    return surprise.astype(cp.float32)


def scale_space_surprise_stat_func(data):
    """Global unsigned scale: p80 maps to +1."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    scale = float(cp.percentile(cp.maximum(valid_data, 0.0), NORMAL_PERCENTILE))
    return (0.0, scale if scale > 1e-9 else 1.0)


class ScaleSpaceSurpriseAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr, **params):
        radii = params.get('radii')
        scales = [float(r) for r in radii] if radii else params.get('scales', [1.0, 2.0, 4.0, 8.0, 16.0])
        weights = params.get('weights', None)
        enhancement = float(params.get('enhancement', 2.0))
        normalize = bool(params.get('normalize', True))
        # 4-sigma Gaussian kernel needs ~4*max_scale of halo for a seam-free
        # core (was 3-sigma, which left a slight tile-boundary discontinuity).
        seamless_depth = int(max(1, cp.ceil(max(scales) * 4).item())) + 1
        # Bound the halo: a depth approaching/exceeding the chunk size forces
        # dask to rechunk-merge whole strips of chunks together, which exhausts
        # VRAM on big rasters with large --radii (the RMM-pool OOM).  Cap at
        # MAX_DEPTH like the other multi-scale algorithms and below the chunk;
        # scales beyond the cap are approximated (truncated halo) instead of
        # crashing the run.
        chunk_cap = int(min(gpu_arr.chunksize)) - 1 if hasattr(gpu_arr, "chunksize") else seamless_depth
        depth = max(1, min(seamless_depth, Constants.MAX_DEPTH, int(min(gpu_arr.shape)) - 1, chunk_cap))
        if seamless_depth > depth:
            logger.warning(
                "scale_space_surprise: max scale %.0f needs a %d-px halo (> cap %d); "
                "very large radii are approximated to stay within memory.",
                float(max(scales)), seamless_depth, depth,
            )
        stats = params.get('global_stats', None)
        stats_ok = isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9
        if normalize and not stats_ok:
            stats = compute_global_stats(
                gpu_arr,
                scale_space_surprise_stat_func,
                compute_scale_space_surprise_block,
                {'scales': scales, 'enhancement': enhancement, 'normalize': False,
                 'weights': weights},
                downsample_factor=params.get('downsample_factor', None),
                depth=depth,
                algorithm_name='scale_space_surprise',
            )
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
            stats = (0.0, 1.0)
        return gpu_arr.map_overlap(
            compute_scale_space_surprise_block, depth=depth,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, enhancement=enhancement, normalize=normalize,
            norm_min=stats[0], norm_scale=stats[1], weights=weights)

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
            is_geo = bool(params.get("is_geographic_dem", False))
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            F = coarsen_factor_for_shape(gpu_arr.shape) if not is_geo else 1
            _depth = lambda rr: max(2, int(float(rr) * 2 + 1))
            cache = {}
            responses = []
            for radius in radii:
                if F > 1 and int(round(float(radius))) > thr:
                    responses.append(coarse_large_radius_response(
                        gpu_arr, block_fn=compute_multi_light_uncertainty_spatial_block,
                        radius_kw="radius", radius=float(radius), factor=F,
                        depth_for_radius=_depth, pixel_size=ps,
                        pixel_scale_x=psx, pixel_scale_y=psy, coarse_cache=cache,
                        azimuths=azimuths, altitude=altitude, z_factor=z_factor,
                        uncertainty_weight=uw,
                    ))
                else:
                    responses.append(gpu_arr.map_overlap(
                        compute_multi_light_uncertainty_spatial_block, depth=_depth(radius),
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
    "compute_scale_space_surprise_block", "scale_space_surprise_stat_func",
    "ScaleSpaceSurpriseAlgorithm",
    "compute_multi_light_uncertainty_block",
    "compute_multi_light_uncertainty_spatial_block",
    "MultiLightUncertaintyAlgorithm",
]
