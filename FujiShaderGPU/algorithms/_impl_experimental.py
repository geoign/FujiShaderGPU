"""
FujiShaderGPU/algorithms/_impl_experimental.py

Experimental algorithm implementations:
  - Scale-Space Surprise
  - Multi-Light Uncertainty
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import logging
import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)

from ._base import DaskAlgorithm, Constants
from ._global_stats import compute_global_stats
from ._nan_utils import (
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    _smooth_for_radius, multiscale_response_fields,
    large_radius_threshold, hybrid_multiscale_response_combine,
)
from ._normalization import NORMAL_PERCENTILE


def sss_large_scale_predicate(scale) -> bool:
    """A scale-space-surprise scale is "large" when its gaussian halo (~4*sigma)
    would exceed MAX_DEPTH.  Such scales are taken from the overview via the hybrid
    coarse path instead of a MAX_DEPTH-truncated halo (the truncation drove the
    tile-boundary seams)."""
    return int(max(1, round(float(scale) * 4)) + 1) > Constants.MAX_DEPTH

logger = logging.getLogger(__name__)


def _sorted_scales_and_pair_weights(scales, weights):
    """Sort scales ascending (carrying weights) and derive per-consecutive-pair
    weights, mirroring ``kernel_scale_space_surprise``.  Returns
    ``(sorted_scales, pair_weights_or_None)``."""
    scale_list = [float(s) for s in scales]
    wl = None
    if weights is not None and len(list(weights)) == len(scale_list):
        wl = [float(w) for w in weights]
    kept = [(s, (wl[i] if wl is not None else None))
            for i, s in enumerate(scale_list) if s > 0]
    kept.sort(key=lambda t: t[0])
    if len(kept) < 2:
        kept = [(1.0, None), (2.0, None), (4.0, None)]
    sorted_scales = [s for s, _ in kept]
    sorted_w = [w for _, w in kept] if all(w is not None for _, w in kept) else None
    pair_w = None
    if sorted_w is not None and len(sorted_scales) >= 2:
        pw = [0.5 * (sorted_w[i] + sorted_w[i + 1]) for i in range(len(sorted_scales) - 1)]
        psum = float(sum(pw))
        if psum > 1e-12:
            pair_w = [p / psum for p in pw]
    return sorted_scales, pair_w


def _sss_smooth_block(block, *, scale, pixel_size=1.0, pixel_scale_x=None,
                      pixel_scale_y=None, **_ignored):
    """One scale's gaussian smooth, matching ``kernel_scale_space_surprise``
    (NaN -> per-block nanmean fill, then gaussian, mode='reflect')."""
    nan_mask = cp.isnan(block)
    work = cp.where(nan_mask, cp.nanmean(block), block) if bool(nan_mask.any()) else block
    return gaussian_filter(work, sigma=max(float(scale), 0.5), mode='reflect').astype(cp.float32)


def _sss_combine_block(block, *smooths, pair_w=None, norm_min=0.0, norm_scale=1.0,
                       enhancement=2.0, normalize=True):
    """Combine per-scale smooths into the surprise map.

    ``surprise = Σ pair_w[i]·|smooth[i+1] - smooth[i]|`` (the per-pixel ``work``
    cancels in ``response[i+1]-response[i] = smooth[i]-smooth[i+1]``), then the
    same p99-normalization + gamma enhancement as the original kernel."""
    nan_mask = cp.isnan(block)
    n = len(smooths)
    surprise = cp.zeros(block.shape, dtype=cp.float32)
    if n >= 2:
        if pair_w is not None:
            for i in range(n - 1):
                surprise += cp.float32(pair_w[i]) * cp.abs(smooths[i + 1] - smooths[i])
        else:
            for i in range(n - 1):
                surprise += cp.abs(smooths[i + 1] - smooths[i])
            surprise /= cp.float32(max(1, n - 1))
    if normalize:
        scale = float(norm_scale)
        if scale > 1e-9:
            base = cp.clip((surprise - float(norm_min)) / scale, 0.0, None)
            surprise = cp.power(base, 1.0 / max(1e-3, float(enhancement)))
        else:
            surprise = cp.zeros_like(surprise)
    return cp.where(nan_mask, cp.float32(cp.nan), surprise).astype(cp.float32)

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
        is_geo = bool(params.get('is_geographic_dem', False))
        ps = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)

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
                depth=min(int(max(1, cp.ceil(max(scales) * 4).item())) + 1, Constants.MAX_DEPTH),
                algorithm_name='scale_space_surprise',
            )
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
            stats = (0.0, 1.0)

        # Per-scale gaussian smooths: small scales at full resolution, large scales
        # on a coarsened overview (no oversized halo -> accurate large --radii, no
        # rechunk-merge OOM).  The surprise then combines |smooth[i+1]-smooth[i]|.
        sorted_scales, pair_w = _sorted_scales_and_pair_weights(scales, weights)
        # Hybrid coarse path (TopoUSM Fast-style): large scales' smooths come precomputed
        # from the COG overview (sampled per-block by global coords, no halo) and
        # the small scales are full-resolution bounded-halo fields, all fused in one
        # depth-0 combine.  Accurate large scales (no MAX_DEPTH halo truncation -> no
        # tile-boundary seam) and bounded VRAM on huge streaming rasters.
        large_fields = params.get("_sss_large_fields")
        if large_fields:
            full_shape = params.get("_sss_full_shape", tuple(int(s) for s in gpu_arr.shape))
            return hybrid_multiscale_response_combine(
                gpu_arr, sorted_scales, small_block_fn=_sss_smooth_block,
                combine_fn=_sss_combine_block,
                depth_for_scale=lambda s: int(max(1, round(float(s) * 4))) + 1,
                large_fields=large_fields, full_shape=full_shape,
                tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                radius_kw="scale", pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
                combine_kwargs=dict(
                    pair_w=pair_w, norm_min=float(stats[0]), norm_scale=float(stats[1]),
                    enhancement=enhancement, normalize=normalize))
        smooths = multiscale_response_fields(
            gpu_arr, sorted_scales, block_fn=_sss_smooth_block,
            depth_for_scale=lambda s: int(max(1, round(s * 4))) + 1,
            pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy, is_geographic=is_geo,
            coarse_dem=params.get("_overview_coarse_dem"),
            coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"))
        return da.map_blocks(
            _sss_combine_block, gpu_arr, *smooths,
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            pair_w=pair_w, norm_min=float(stats[0]), norm_scale=float(stats[1]),
            enhancement=enhancement, normalize=normalize)

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
            responses = multiscale_response_fields(
                gpu_arr, [float(r) for r in radii],
                block_fn=compute_multi_light_uncertainty_spatial_block, radius_kw="radius",
                depth_for_scale=lambda rr: max(2, int(float(rr) * 2 + 1)),
                is_large=lambda rr: int(round(float(rr))) > thr,
                pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy, is_geographic=is_geo,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                azimuths=azimuths, altitude=altitude, z_factor=z_factor,
                uncertainty_weight=uw)
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
