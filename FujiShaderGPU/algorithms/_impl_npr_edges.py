"""
FujiShaderGPU/algorithms/_impl_npr_edges.py

NPR Edges (non-photorealistic rendering outlines) algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import logging
from typing import Optional
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, convolve, binary_dilation

from ._base import Constants, DaskAlgorithm, classify_resolution
from ._nan_utils import (
    restore_nan,
    _resolve_spatial_radii_weights, _combine_multiscale_dask,
    large_radius_threshold, multiscale_response_fields,
    _smooth_for_radius,
)

logger = logging.getLogger(__name__)


def compute_npr_edges_block(block: cp.ndarray, *, edge_sigma: float = 1.0,
                          threshold_low: float = 0.2, threshold_high: float = 0.5,
                          pixel_size: float = 1.0, grad_stats=None,
                          _return_grad: bool = False) -> cp.ndarray:
    """NPR-style outline extraction (simplified v2).

    ``grad_stats`` = (base_threshold, threshold_range, grad_mean) supplies the
    edge-detection threshold from a GLOBAL gradient distribution instead of the
    per-block one, which differs tile-to-tile and produces tile-boundary seams.
    ``_return_grad`` returns the gradient magnitude (used by the global-stats
    prepass that derives ``grad_stats``)."""
    nan_mask = cp.isnan(block)
    resolution_class = classify_resolution(pixel_size)

    # Resolution-dependent smoothing
    if resolution_class in ['ultra_high', 'very_high']:
        adaptive_sigma = 0.5
    elif resolution_class in ['high', 'medium']:
        adaptive_sigma = 1.0
    elif resolution_class == 'low':
        adaptive_sigma = 0.5
    else:
        adaptive_sigma = 0.3

    if edge_sigma != 1.0:
        adaptive_sigma = edge_sigma

    # Denoise (minimal)
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(filled, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = filled
    else:
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(block, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = block

    # Gradient computation using a Sobel filter
    sobel_x = cp.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=cp.float32) / 8.0
    sobel_y = cp.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=cp.float32) / 8.0

    dx = convolve(smoothed, sobel_x, mode='nearest')
    dy = convolve(smoothed, sobel_y, mode='nearest')

    gradient_mag = cp.sqrt(dx**2 + dy**2)

    # Resolution-adaptive gradient enhancement
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        local_max = maximum_filter(smoothed, size=3, mode='nearest')
        local_min = minimum_filter(smoothed, size=3, mode='nearest')
        local_range = local_max - local_min
        gradient_mag = cp.maximum(gradient_mag, local_range * 0.3)

    # Global-stats prepass hook: return the gradient magnitude so the caller can
    # pool it across stratified tiles into a global threshold.
    if _return_grad:
        return restore_nan(gradient_mag, nan_mask).astype(cp.float32)

    gradient_dir = cp.arctan2(dy, dx)

    # Adaptive thresholding
    valid_grad = gradient_mag[~nan_mask] if nan_mask.any() else gradient_mag.ravel()
    if grad_stats is not None:
        # Global threshold (seam-free): base/range/mean computed once over the
        # whole raster instead of per block.
        base_threshold = float(grad_stats[0])
        threshold_range = float(grad_stats[1])
        grad_mean = float(grad_stats[2])
        actual_threshold_low = base_threshold + threshold_range * threshold_low * 0.5
        actual_threshold_high = base_threshold + threshold_range * threshold_high
        min_threshold = grad_mean * 0.1
        actual_threshold_low = max(actual_threshold_low, min_threshold)
        actual_threshold_high = max(actual_threshold_high, min_threshold * 2)
    elif len(valid_grad) > 0:
        grad_std = cp.std(valid_grad)
        grad_mean = cp.mean(valid_grad)

        if resolution_class in ['low', 'very_low', 'ultra_low']:
            base_threshold = grad_mean
            threshold_range = grad_std * 1.5
        else:
            base_threshold = cp.percentile(valid_grad, 50)
            threshold_range = cp.percentile(valid_grad, 90) - base_threshold

        actual_threshold_low = base_threshold + threshold_range * threshold_low * 0.5
        actual_threshold_high = base_threshold + threshold_range * threshold_high

        min_threshold = grad_mean * 0.1
        actual_threshold_low = cp.maximum(actual_threshold_low, min_threshold)
        actual_threshold_high = cp.maximum(actual_threshold_high, min_threshold * 2)
    else:
        actual_threshold_low = 0.1
        actual_threshold_high = 0.3

    # Non-maximum suppression (simplified)
    angle = gradient_dir * 180.0 / cp.pi
    angle[angle < 0] += 180

    nms = gradient_mag.copy()

    # Non-maximum suppression over 8 directions
    shifted_pos = cp.roll(gradient_mag, 1, axis=1)
    shifted_neg = cp.roll(gradient_mag, -1, axis=1)
    mask = ((angle < 22.5) | (angle >= 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    # 45deg bucket: with row-down (y-down) array coordinates an angle in
    # [22.5, 67.5) means the gradient points along the MAIN diagonal
    # (row+1, col+1), so suppress against those neighbours.  (The previous
    # pairing compared the anti-diagonal -- a y-up-convention port artifact --
    # which thinned diagonal edges along the wrong direction.)
    shifted_pos = cp.roll(cp.roll(gradient_mag, -1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, 1, axis=0), 1, axis=1)
    mask = ((angle >= 22.5) & (angle < 67.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    shifted_pos = cp.roll(gradient_mag, 1, axis=0)
    shifted_neg = cp.roll(gradient_mag, -1, axis=0)
    mask = ((angle >= 67.5) & (angle < 112.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    # 135deg bucket: gradient along the ANTI-diagonal (row-1, col+1).
    shifted_pos = cp.roll(cp.roll(gradient_mag, 1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, -1, axis=0), 1, axis=1)
    mask = ((angle >= 112.5) & (angle < 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    # Double thresholding
    strong = nms > actual_threshold_high
    weak = (nms > actual_threshold_low) & (nms <= actual_threshold_high)

    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5

    # Hysteresis processing
    for _ in range(3):
        dilated = cp.maximum(
            cp.maximum(cp.roll(edges, 1, axis=0), cp.roll(edges, -1, axis=0)),
            cp.maximum(cp.roll(edges, 1, axis=1), cp.roll(edges, -1, axis=1))
        )
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, 1, axis=0), 1, axis=1))
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, -1, axis=0), -1, axis=1))
        edges = cp.where(weak & (dilated > 0.5), 1.0, edges)

    # Post-processing: adjust edge thickness by resolution
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        structure = cp.ones((3, 3))
        edges_binary = edges > 0.5
        edges_dilated = binary_dilation(edges_binary, structure=structure).astype(cp.float32)
        edges = cp.where(edges_dilated, cp.maximum(edges, 0.8), edges)

    edges = edges * 0.8
    result = 1.0 - edges
    result = cp.clip(result, 0.2, 1.0)
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


def compute_npr_edges_spatial_block(block, *, edge_sigma=1.0, threshold_low=0.2,
                                    threshold_high=0.5, pixel_size=1.0,
                                    pixel_scale_x=None, pixel_scale_y=None,
                                    radius=4.0, grad_stats_map=None,
                                    _return_grad=False):
    """Edges of the terrain viewed at a given spatial scale (DEM pre-smoothed)."""
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="npr_edges")
    # Global per-radius gradient threshold (seam-free) when provided.  Large radii
    # run the whole coarsened grid as one block, so their per-block threshold is
    # already global -- only the small (full-res, multi-tile) radii need this.
    gs = None
    if grad_stats_map:
        gs = grad_stats_map.get(int(round(float(radius))))
    return compute_npr_edges_block(
        smoothed, edge_sigma=edge_sigma, threshold_low=threshold_low,
        threshold_high=threshold_high, pixel_size=pixel_size,
        grad_stats=gs, _return_grad=_return_grad)


class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR outline algorithm (simplified)."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        edge_sigma = params.get('edge_sigma', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        threshold_low = params.get('threshold_low', 0.2)
        threshold_high = params.get('threshold_high', 0.5)
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), pixel_size)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            # Outlines at multiple scales: pre-smooth the DEM at each radius,
            # detect edges, weighted-combine (large radii via the coarse path).
            is_geo = bool(params.get("is_geographic_dem", False))
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            responses = multiscale_response_fields(
                gpu_arr, [float(r) for r in radii],
                block_fn=compute_npr_edges_spatial_block, radius_kw="radius",
                depth_for_scale=lambda rr: max(3, int(float(rr) * 2 + 1)),
                is_large=lambda rr: int(round(float(rr))) > thr,
                pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
                is_geographic=is_geo, edge_sigma=edge_sigma,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                threshold_low=threshold_low, threshold_high=threshold_high,
                grad_stats_map=params.get("_npr_grad_stats"))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        depth = 3
        if edge_sigma != 1.0:
            depth = max(depth, int(edge_sigma * 4 + 2))

        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            edge_sigma=edge_sigma,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            pixel_size=pixel_size,
        )

    def get_default_params(self) -> dict:
        return {
            'edge_sigma': 1.0,
            'threshold_low': 0.2,
            'threshold_high': 0.5,
            'pixel_size': 1.0,
            'mode': 'local', 'radii': None, 'weights': None,
        }


def _compute_npr_grad_stats(
    src_cog: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
    small_radius_max: float = 600.0,
) -> Optional[dict]:
    """Per-radius GLOBAL gradient (base, range, mean) for npr_edges, from full-res
    stratified tiles.  npr's edge threshold is otherwise computed per block, which
    differs tile-to-tile and seams.  Only the small (full-res, multi-tile) radii
    need this; large radii run the whole coarsened grid as one block and are
    already global.  Returns {round(radius): (base, range, mean)}.

    Backend-neutral (rasterio + cupy only) so both the Dask and tile pipelines
    share one implementation."""
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("npr grad-stats helpers unavailable: %s", exc)
        return None
    radii = (params or {}).get("radii") or []
    small = [float(r) for r in radii if float(r) <= small_radius_max]
    if not small:
        return None
    pixel_size = float(params.get("pixel_size", 1.0))
    edge_sigma = float(params.get("edge_sigma", 1.0))
    tl = float(params.get("threshold_low", 0.2))
    th_ = float(params.get("threshold_high", 0.5))
    low_res = classify_resolution(pixel_size) in ("low", "very_low", "ultra_low")
    try:
        from ._norm_stats import stratified_windows

        margin = int(min(2 * max(small) + 16, max_tile // 4))
        tile = int(min(max_tile, max(2048, 4 * margin)))
        # radius -> list of pooled host arrays.  One GPU tile is resident at a
        # time (the radius loop runs inside the tile loop); the previous layout
        # held every stratified tile on the device simultaneously (~600MB VRAM
        # for nine 4096^2 tiles).
        pools = {float(r): [] for r in small}
        with rasterio.open(src_cog) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            def _dn(a):
                a = a.astype(np.float32, copy=False)
                if nodata is not None and not np.isnan(float(nodata)):
                    a = np.where(np.isclose(a, float(nodata), atol=1e-6), np.nan, a)
                return a

            cov = max(1, max(W, H) // 512)
            ov = _dn(src.read(1, out_shape=(max(1, H // cov), max(1, W // cov)),
                              resampling=Resampling.nearest, out_dtype=np.float32,
                              masked=True).filled(np.nan))
            vmask = np.isfinite(ov)
            if not vmask.any():
                return None
            ys, xs = np.where(vmask)
            by0, by1 = int(ys.min()) * cov, min(H, (int(ys.max()) + 1) * cov)
            bx0, bx1 = int(xs.min()) * cov, min(W, (int(xs.max()) + 1) * cov)
            for wy0, wx0, tw, th2 in stratified_windows(
                    W, H, by0, by1, bx0, bx1, grid=grid, tile=tile):
                a = _dn(src.read(1, window=Window(wx0, wy0, tw, th2),
                                 out_dtype=np.float32, masked=True).filled(np.nan))
                if float(np.isfinite(a).mean()) < min_valid_frac:
                    continue
                g = cp.asarray(a)
                for r in small:
                    grad = compute_npr_edges_spatial_block(
                        g, edge_sigma=edge_sigma, threshold_low=tl, threshold_high=th_,
                        pixel_size=pixel_size, radius=r, _return_grad=True)
                    m = int(min(int(2 * r + 16), grad.shape[0] // 3, grad.shape[1] // 3))
                    if m > 0:
                        grad = grad[m:-m, m:-m]
                    v = grad[~cp.isnan(grad)]
                    if v.size:
                        pools[float(r)].append(cp.asnumpy(v))
                    del grad, v
                del g
                cp.get_default_memory_pool().free_all_blocks()
        out = {}
        for r in small:
            pool = pools[float(r)]
            if not pool:
                continue
            allv = np.concatenate(pool)
            mean = float(np.mean(allv))
            if low_res:
                base, rng = mean, float(np.std(allv)) * 1.5
            else:
                base = float(np.percentile(allv, 50))
                rng = float(np.percentile(allv, 90)) - base
            out[int(round(r))] = (base, rng, mean)
        if not out:
            return None
        logger.info("npr_edges global gradient threshold computed for radii %s",
                    sorted(out.keys()))
        return out
    except Exception as exc:
        logger.warning("Failed to compute npr grad stats: %s", exc)
        return None


__all__ = [
    "compute_npr_edges_block",
    "compute_npr_edges_spatial_block",
    "NPREdgesAlgorithm",
    "_compute_npr_grad_stats",
]
