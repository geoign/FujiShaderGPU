"""
FujiShaderGPU/algorithms/_impl_visual_saliency.py

Visual Saliency algorithm implementation.
Module split out from dask_shared.py (Phase 3).
"""
from __future__ import annotations
import logging
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm, Constants
from ._nan_utils import (
    restore_nan, resolve_block_weights, hybrid_multiscale_response_combine,
)
from ._global_stats import compute_global_stats
from ._normalization import NORMAL_PERCENTILE

logger = logging.getLogger(__name__)


def vs_large_scale_predicate(scale) -> bool:
    """A conspicuity scale is "large" when its gaussian halo (~5*sigma) would
    exceed MAX_DEPTH, i.e. the single-block path would truncate it.  Such scales
    are taken from the overview via the hybrid coarse path instead."""
    return int(float(scale) * 5) > Constants.MAX_DEPTH


def _vs_fill(block):
    """NaN -> finite per-block fill (nanmean), matching compute_visual_saliency_block."""
    nan_mask = cp.isnan(block)
    if not bool(nan_mask.any()):
        return block.astype(cp.float32, copy=False)
    fill = cp.nanmean(block)
    fill = cp.where(cp.isfinite(fill), fill, cp.float32(0.0))
    return cp.where(nan_mask, fill, block).astype(cp.float32)


def _vs_smooth_block(block, *, scale, pixel_size=1.0, pixel_scale_x=None,
                     pixel_scale_y=None, **_ignored):
    """One conspicuity scale's gaussian smooth of the NaN-filled DEM (mode='nearest')."""
    work = _vs_fill(block)
    return gaussian_filter(work, sigma=max(0.5, float(scale)), mode='nearest').astype(cp.float32)


def _weighted_mean_maps(maps, weights, ref):
    """Mean of per-scale feature maps, weighted when ``weights`` is given.

    ``weights`` is a list of per-map scalar weights aligned with ``maps``; None
    or empty falls back to a plain mean (original behavior).  ``ref`` supplies
    the fallback shape when ``maps`` is empty.
    """
    if not maps:
        return cp.zeros_like(ref, dtype=cp.float32)
    if not weights:
        return cp.mean(cp.stack(maps, axis=0), axis=0)
    w = cp.asarray(weights, dtype=cp.float32)
    s = float(w.sum())
    if s <= 1e-12:
        return cp.mean(cp.stack(maps, axis=0), axis=0)
    w = (w / s).reshape((-1,) + (1,) * maps[0].ndim)
    return cp.sum(cp.stack(maps, axis=0) * w, axis=0)


def _compress_saliency_feature(feature):
    """Tile-stable feature compression without block-global normalization."""
    return cp.log1p(cp.clip(feature, 0.0, None)).astype(cp.float32)


def visual_saliency_stat_func(data):
    """Global unsigned scale: robust p99 (``NORMAL_PERCENTILE``) maps to +1."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    scale = float(cp.percentile(cp.maximum(valid_data, 0.0), NORMAL_PERCENTILE))
    return (0.0, scale if scale > 1e-9 else 1.0)


def compute_visual_saliency_block(block, *, scales=None, radii=None,
                                 pixel_size=1.0, pixel_scale_x=None,
                                 pixel_scale_y=None, normalize=True,
                                 norm_min=None, norm_scale=None, weights=None):
    """Itti-style saliency, simplified and adapted for single-band terrain (DEM).

    This is an approximation of the Itti-Koch-Niebur (1998) saliency model,
    deliberately reduced to fit single-channel elevation data and the seam-free
    tiled/streaming backend.  It keeps the model's skeleton -- multiscale
    center-surround + orientation conspicuity, then a combined map -- but is NOT
    the original algorithm.  Differences from Itti et al. (1998):

    * Channels: elevation is used directly as the intensity channel; there is no
      colour channel (RG/BY) because the input is single-band.
    * Center-surround (intensity): difference-of-Gaussians ``|G(sigma_c)-G(sigma_s)|``
      across the conspicuity scales (a standard DoG approximation of the original
      across-pyramid-level subtraction).
    * Orientation: a gradient-orientation response ``mag * max(cos(2*(theta-o)), 0)``
      at four orientations (0/45/90/135deg) -- NOT Gabor filters, and with no
      center-surround on the orientation channel.
    * Normalization: the model's defining map-normalization operator N(.) (which
      promotes maps with a few strong peaks via ``(M - mean_local_maxima)^2``) is
      omitted because it is global/iterative and would seam across tiles.  It is
      replaced by a tile-stable ``log1p`` feature compression plus one global
      percentile scale (``visual_saliency_stat_func``) for display.
    * Winner-take-all / inhibition-of-return (attention scan-path dynamics) are
      not implemented -- only the saliency map itself is produced.

    The unified ``--weights`` (length-matching the conspicuity scales) weights
    each scale's contribution to the intensity and orientation conspicuity means;
    absent/mismatched weights keep equal averaging.
    """
    if radii:  # unified --radii feeds the conspicuity scales
        scales = [float(r) for r in radii]
    if scales is None:
        scales = [2, 4, 8, 16]
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        fill = cp.nanmean(block)
        fill = cp.where(cp.isfinite(fill), fill, 0.0)
        work = cp.where(nan_mask, fill, block).astype(cp.float32)
    else:
        work = block.astype(cp.float32, copy=False)
    use_scales = [max(0.5, float(s)) for s in scales]
    if len(use_scales) < 4:
        use_scales = [2.0, 4.0, 8.0, 16.0]
    # Per-scale weights (unified --weights); None -> equal averaging.
    wvec = resolve_block_weights(weights, len(use_scales))
    w_host = cp.asnumpy(wvec).tolist() if wvec is not None else None
    # Compute every per-scale gaussian smooth once and reuse it: the previous
    # per-(center, surround) recomputation ran each gaussian up to twice.
    smooths = [gaussian_filter(work, sigma=s, mode='nearest') for s in use_scales]
    c_indices = [0, 1]
    deltas = [2, 3]
    intensity_maps = []
    intensity_w = []
    for ci in c_indices:
        for d in deltas:
            si = ci + d
            if si >= len(use_scales):
                continue
            fm = cp.abs(smooths[ci] - smooths[si])
            intensity_maps.append(_compress_saliency_feature(fm))
            if wvec is not None:
                intensity_w.append(w_host[ci])
    intensity = _weighted_mean_maps(intensity_maps, intensity_w if wvec is not None else None, work)
    ori_maps = []
    ori_w = []
    orientations = [0.0, cp.pi / 4, cp.pi / 2, 3 * cp.pi / 4]
    step_y = float(pixel_scale_y if pixel_scale_y is not None else pixel_size)
    step_x = float(pixel_scale_x if pixel_scale_x is not None else pixel_size)
    if abs(step_y) < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if abs(step_x) < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    for j in range(min(3, len(use_scales))):
        gy, gx = cp.gradient(smooths[j], step_y, step_x)
        mag = cp.sqrt(gx * gx + gy * gy) + 1e-8
        theta = cp.arctan2(gy, gx)
        for o in orientations:
            resp = mag * cp.maximum(cp.cos(2.0 * (theta - o)), 0.0)
            ori_maps.append(_compress_saliency_feature(resp))
            if wvec is not None:
                ori_w.append(w_host[j])
    orientation = _weighted_mean_maps(ori_maps, ori_w if wvec is not None else None, work)
    sal = 0.5 * (intensity + orientation)
    if normalize:
        if norm_min is None or norm_scale is None:
            norm_min, norm_scale = visual_saliency_stat_func(sal)
        if norm_scale > 1e-9:
            result = (sal - norm_min) / norm_scale
        else:
            result = cp.zeros_like(sal)
        result = cp.maximum(result, 0.0)  # p99 -> 1.0; tail passes through unclipped
    else:
        result = sal
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


def _vs_combine_block(block, *smooths, weights=None, pixel_size=1.0,
                      pixel_scale_x=None, pixel_scale_y=None,
                      normalize=True, norm_min=None, norm_scale=None):
    """Itti intensity + orientation conspicuity from precomputed per-scale smooths.

    Equivalent to compute_visual_saliency_block, but the smooths arrive as
    arguments (large scales computed via the coarse-overview path).  NaN that the
    coarse path re-masked into a smooth is refilled so large/small scales are
    treated identically; the true NoData footprint is restored at the end.
    """
    nan_mask = cp.isnan(block)
    fillv = cp.nanmean(block)
    fillv = cp.where(cp.isfinite(fillv), fillv, cp.float32(0.0))
    sm = [cp.where(cp.isnan(s), fillv, s).astype(cp.float32) for s in smooths]
    n = len(sm)
    wvec = resolve_block_weights(weights, n)
    w_host = cp.asnumpy(wvec).tolist() if wvec is not None else None
    # Intensity conspicuity: running weighted mean of |center - surround|.  The
    # maps are accumulated in place (not stacked) so peak per-block VRAM stays low
    # on large rasters -- a stacked list of all intensity+orientation maps exhausts
    # the RMM pool.  Equivalent to _weighted_mean_maps over the same maps/weights.
    I_acc = cp.zeros_like(block, dtype=cp.float32)
    I_w = 0.0
    for ci in (0, 1):
        for d in (2, 3):
            si = ci + d
            if si >= n:
                continue
            fm = _compress_saliency_feature(cp.abs(sm[ci] - sm[si]))
            wj = w_host[ci] if w_host is not None else 1.0
            I_acc += fm * cp.float32(wj)
            I_w += wj
    intensity = I_acc / cp.float32(I_w) if I_w > 0 else I_acc
    # Orientation conspicuity: running weighted mean over scales x orientations.
    O_acc = cp.zeros_like(block, dtype=cp.float32)
    O_w = 0.0
    orientations = [0.0, cp.pi / 4, cp.pi / 2, 3 * cp.pi / 4]
    step_y = float(pixel_scale_y if pixel_scale_y is not None else pixel_size)
    step_x = float(pixel_scale_x if pixel_scale_x is not None else pixel_size)
    if abs(step_y) < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if abs(step_x) < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    for j in range(min(3, n)):
        gy, gx = cp.gradient(sm[j], step_y, step_x)
        mag = cp.sqrt(gx * gx + gy * gy) + 1e-8
        theta = cp.arctan2(gy, gx)
        wj = w_host[j] if w_host is not None else 1.0
        for o in orientations:
            resp = _compress_saliency_feature(mag * cp.maximum(cp.cos(2.0 * (theta - o)), 0.0))
            O_acc += resp * cp.float32(wj)
            O_w += wj
    orientation = O_acc / cp.float32(O_w) if O_w > 0 else O_acc
    sal = 0.5 * (intensity + orientation)
    if normalize:
        if norm_min is None or norm_scale is None:
            norm_min, norm_scale = visual_saliency_stat_func(sal)
        if float(norm_scale) > 1e-9:
            result = (sal - float(norm_min)) / float(norm_scale)
        else:
            result = cp.zeros_like(sal)
        result = cp.maximum(result, 0.0)
    else:
        result = sal
    return restore_nan(result, nan_mask).astype(cp.float32)


class VisualSaliencyAlgorithm(DaskAlgorithm):
    """Visual saliency from Itti-style conspicuity maps, simplified for terrain.

    A terrain-adapted approximation of Itti-Koch-Niebur (1998): multiscale
    center-surround (intensity) + gradient-orientation conspicuity, combined and
    percentile-normalized for display.  It omits the model's N(.) normalization
    operator, Gabor orientation, colour channel, and attention dynamics -- see
    ``compute_visual_saliency_block`` for the full list of simplifications."""
    def process(self, gpu_arr, **params):
        radii = params.get('radii')
        scales = [float(r) for r in radii] if radii else params.get('scales', [2, 4, 8, 16])
        weights = params.get('weights', None)
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        # Conspicuity scales (>=4, matching compute_visual_saliency_block).
        use_scales = [max(0.5, float(s)) for s in scales]
        if len(use_scales) < 4:
            use_scales = [2.0, 4.0, 8.0, 16.0]

        stats = params.get('global_stats', None)
        stats_ok = isinstance(stats, (tuple, list)) and len(stats) >= 2
        if not stats_ok:
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    gpu_arr, visual_saliency_stat_func,
                    compute_visual_saliency_block,
                    {'scales': scales, 'pixel_size': pixel_size,
                     'pixel_scale_x': pixel_scale_x,
                     'pixel_scale_y': pixel_scale_y, 'normalize': False,
                     'weights': weights},
                    depth=min(int(max(use_scales) * 5), Constants.MAX_DEPTH))
            else:
                stats = (0.0, 1.0)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2):
            stats = (0.0, 1.0)

        # Hybrid coarse path (TopoUSM Fast-style): when the orchestrator supplies the
        # large-scale smooth fields precomputed from the COG overview, compute the
        # small conspicuity scales at full resolution (bounded halo) and sample the
        # large scales from their concrete overview fields inside one depth-0
        # combine.  Accurate for large --radii (true low-frequency smooth, not a
        # MAX_DEPTH-truncated gaussian) and bounded in VRAM on huge streaming
        # rasters (no per-large-scale Dask field whose GPU intermediates accumulate).
        large_fields = params.get("_vs_large_fields")
        if large_fields:
            full_shape = params.get("_vs_full_shape", tuple(int(s) for s in gpu_arr.shape))
            return hybrid_multiscale_response_combine(
                gpu_arr, [float(s) for s in use_scales],
                small_block_fn=_vs_smooth_block,
                combine_fn=_vs_combine_block,
                depth_for_scale=lambda s: int(float(s) * 5),
                large_fields=large_fields, full_shape=full_shape,
                tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                radius_kw="scale", pixel_size=pixel_size,
                pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                combine_kwargs=dict(
                    weights=weights, pixel_size=pixel_size,
                    pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                    normalize=True, norm_min=stats[0], norm_scale=stats[1]))

        # Single self-contained block per output tile (bounds device memory).  The
        # coarse-overview decomposition into many per-scale smooth fields fed a deep
        # dask graph whose GPU intermediates accumulated to tens of GB on large
        # streaming rasters (ALOS/Kyoto) and exhausted the RMM pool.  The halo is
        # capped at MAX_DEPTH (and below the chunk), so very large radii are
        # approximated (truncated halo) instead of OOM; default/small radii are
        # unaffected.
        min_chunk = min((min(ax) for ax in gpu_arr.chunks), default=1) if hasattr(gpu_arr, "chunks") else 1
        depth = max(1, min(int(max(use_scales) * 5), Constants.MAX_DEPTH, int(min_chunk) - 1))
        return gpu_arr.map_overlap(
            compute_visual_saliency_block, depth=depth,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales, weights=weights, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            normalize=True, norm_min=stats[0], norm_scale=stats[1])

    def get_default_params(self):
        return {
            'scales': [2, 4, 8, 16], 'pixel_size': 1.0,
            'verbose': False,
        }


__all__ = [
    "_compress_saliency_feature", "visual_saliency_stat_func",
    "compute_visual_saliency_block", "VisualSaliencyAlgorithm",
]
