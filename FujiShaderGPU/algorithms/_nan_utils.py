"""
FujiShaderGPU/algorithms/_nan_utils.py

NaN handling, spatial smoothing, down/up-sampling, and restore helpers.
Module split out from dask_shared.py (Phase 1).
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, zoom

from ._base import Constants
from .common.spatial_mode import auto_spatial_radii, auto_spatial_profile, auto_spatial_weights


def handle_nan_with_gaussian(block: cp.ndarray, sigma: float, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaN-aware Gaussian filtering."""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return gaussian_filter(block, sigma=sigma, mode=mode), nan_mask

    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)

    smoothed_values = gaussian_filter(filled * valid, sigma=sigma, mode=mode)
    smoothed_weights = gaussian_filter(valid, sigma=sigma, mode=mode)
    smoothed = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)

    return smoothed, nan_mask


def handle_nan_with_uniform(block: cp.ndarray, size: int, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaN-aware uniform_filter processing."""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return uniform_filter(block, size=size, mode=mode), nan_mask

    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)

    sum_values = uniform_filter(filled * valid, size=size, mode=mode)
    sum_weights = uniform_filter(valid, size=size, mode=mode)
    mean = cp.where(sum_weights > 0, sum_values / sum_weights, 0)

    return mean, nan_mask


def handle_nan_for_gradient(block: cp.ndarray, scale: float = 1.0,
                          pixel_size: float = 1.0,
                          pixel_scale_x: float = None,
                          pixel_scale_y: float = None) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """NaN-aware gradient computation."""
    nan_mask = cp.isnan(block)
    if bool(nan_mask.any()):
        if bool((~nan_mask).any()):
            local_fill, _ = handle_nan_with_gaussian(block, sigma=1.0, mode="nearest")
            filled = cp.where(nan_mask, local_fill, block)
        else:
            filled = cp.zeros_like(block)
    else:
        filled = block

    # Use metric spacing magnitude only. Sign carries geotransform orientation,
    # which can unintentionally flip illumination direction in shading algorithms.
    step_y = abs(float(pixel_scale_y if pixel_scale_y is not None else pixel_size))
    step_x = abs(float(pixel_scale_x if pixel_scale_x is not None else pixel_size))
    if step_y < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if step_x < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    dy, dx = cp.gradient(filled * scale, step_y, step_x, edge_order=2)
    return dy, dx, nan_mask


def _normalize_spatial_radii(radii: Optional[List[int]], pixel_size: float) -> List[int]:
    """Normalize user-provided radii or auto-derive stable defaults."""
    if radii is None:
        return auto_spatial_radii(None)
    out: List[int] = []
    for r in radii:
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        if rv > 0:
            out.append(rv)
    if not out:
        return auto_spatial_radii(None)
    # Keep user order while dropping duplicates.
    seen = set()
    ordered = []
    for v in out:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _resolve_spatial_radii_weights(
    radii: Optional[List[int]],
    weights: Optional[List[float]],
    pixel_size: float,
    short_side_px: Optional[float] = None,
) -> Tuple[List[int], Optional[List[float]]]:
    """Resolve radii/weights with the DEM-size-aware auto rule.

    Orchestrators (process_dem_tiles / run_pipeline) normally inject explicit
    radii+weights derived from the *full* DEM short side, so the ``radii is None``
    branch here is only a defensive fallback (``short_side_px=None`` -> full
    geometric sequence).  When radii are present but weights are omitted, the
    ``2**n`` weight profile is applied.
    """
    if radii is None:
        auto_radii, auto_weights = auto_spatial_profile(short_side_px)
        if _weight_count_matches(weights, len(auto_radii)):
            user = _clean_normalized_weights(weights)
            if user is not None:
                return auto_radii, user
        return auto_radii, auto_weights

    resolved_radii = _normalize_spatial_radii(radii, pixel_size)
    if _weight_count_matches(weights, len(resolved_radii)):
        user = _clean_normalized_weights(weights)
        if user is not None:
            return resolved_radii, user
    # Weights omitted or invalid: fall back to the 2**n profile (nearer = heavier).
    return resolved_radii, auto_spatial_weights(len(resolved_radii))


def _weight_count_matches(weights, expected: int) -> bool:
    """Accept normal sequences/arrays while rejecting scalars and strings."""
    if weights is None or isinstance(weights, (str, bytes)):
        return False
    try:
        return len(weights) == expected
    except TypeError:
        return False


def _clean_normalized_weights(weights) -> Optional[List[float]]:
    """Sanitize and L1-normalize weights; None if all non-positive/invalid."""
    cleaned: List[float] = []
    for w in weights:
        try:
            fv = float(w)
        except (TypeError, ValueError):
            return None
        cleaned.append(fv if np.isfinite(fv) and fv > 0 else 0.0)
    s = float(sum(cleaned))
    if s <= 0:
        return None
    return [v / s for v in cleaned]


def resolve_block_weights(weights, n: int) -> Optional[cp.ndarray]:
    """Normalize a per-scale weight list to a cupy float32 vector of length ``n``.

    Returns ``None`` (→ caller keeps its default equal/intrinsic weighting) when
    weights are absent, the wrong length, non-finite, or non-positive.  Used by
    the intrinsically multi-scale algorithms (visual_saliency, scale_space_surprise,
    fractal_anomaly) so the unified ``--weights`` influences their scale mixing
    without changing behavior when no weights are supplied.
    """
    if weights is None or n <= 0:
        return None
    try:
        vals = [float(w) for w in weights]
    except (TypeError, ValueError):
        return None
    if len(vals) != n:
        return None
    arr = cp.asarray(vals, dtype=cp.float32)
    arr = cp.where(cp.isfinite(arr) & (arr > 0), arr, cp.float32(0.0))
    s = float(arr.sum())
    if s <= 1e-12:
        return None
    return arr / s


def _combine_multiscale_dask(
    responses: List[da.Array],
    *,
    weights: Optional[List[float]] = None,
    agg: str = "mean",
) -> da.Array:
    """Combine per-radius dask responses with optional weighted mean."""
    if not responses:
        raise ValueError("responses must not be empty")
    agg_norm = str(agg or "mean").lower()
    if agg_norm == "stack":
        # Formal stack contract: band-first (C, H, W), even for a single scale.
        return da.stack(responses, axis=0)
    if len(responses) == 1:
        return responses[0]

    stacked = da.stack(responses, axis=0)
    if agg_norm == "max":
        return da.max(stacked, axis=0)
    if agg_norm == "min":
        return da.min(stacked, axis=0)
    if agg_norm == "sum":
        return da.sum(stacked, axis=0)

    if _weight_count_matches(weights, len(responses)):
        clean = _clean_normalized_weights(weights)
        if clean is not None:
            out = responses[0] * clean[0]
            for i in range(1, len(responses)):
                out = out + responses[i] * clean[i]
            return out
    return da.mean(stacked, axis=0)


# ---------------------------------------------------------------------------
# Large-radius-from-overview helpers (shared by spatial-mode algorithms)
# ---------------------------------------------------------------------------
def large_radius_threshold(gpu_arr: da.Array, fallback: int) -> int:
    """Radii above this are computed from a coarsened copy (no large halo).

    Default = max(256, min_chunk // 16), matching the TopoUSM Fast threshold.
    """
    try:
        min_chunk = min(min(gpu_arr.chunks[0]), min(gpu_arr.chunks[1]))
    except Exception:
        min_chunk = int(fallback)
    return int(max(256, int(min_chunk) // 16))


def coarsen_factor_for_shape(shape, coarse_max: int = 2048) -> int:
    """Power-of-two decimation so the longest side is <= ``coarse_max``."""
    longest = max(int(shape[0]), int(shape[1]))
    if longest <= int(coarse_max):
        return 1
    return 1 << int(np.ceil(np.log2(longest / float(coarse_max))))


def _resolve_scattered(value):
    """Return a concrete value from a distributed Future-like object if needed."""
    if hasattr(value, "result") and callable(getattr(value, "result")):
        return value.result()
    return value


def _scatter_if_client(value):
    """Scatter a bulky concrete value when a distributed client is active."""
    try:
        from distributed import get_client
        return get_client().scatter(value, broadcast=True, hash=False)
    except Exception:
        return value


def _bilinear_sample_coarse(
    coarse: cp.ndarray,
    r0: int, r1: int, c0: int, c1: int,
    full_h: int, full_w: int,
) -> cp.ndarray:
    """Bilinearly sample ``coarse`` at the full-res pixel window [r0:r1, c0:c1].

    The (2, h, w) coordinate array is built by broadcast assignment into a
    preallocated float32 buffer rather than ``meshgrid`` + ``stack``.  The latter
    materialises *three* full h*w arrays in float64 (two grids + the stacked
    copy); on a large chunk (e.g. 8192^2) the stacked array alone is 1 GiB, which
    exhausts the RMM pool.  float32 coordinates stay accurate to well under a
    coarse-grid pixel for raster dimensions up to ~16M (float32 integer-exact
    range), so the sampled result is unchanged while peak memory drops ~3x.
    """
    from cupyx.scipy.ndimage import map_coordinates

    coarse = _resolve_scattered(coarse)
    ch, cw = coarse.shape
    h = int(r1 - r0)
    w = int(c1 - c0)
    rr = (cp.arange(r0, r1, dtype=cp.float32) + cp.float32(0.5)) * cp.float32(ch / float(full_h)) - cp.float32(0.5)
    cc = (cp.arange(c0, c1, dtype=cp.float32) + cp.float32(0.5)) * cp.float32(cw / float(full_w)) - cp.float32(0.5)
    coords = cp.empty((2, h, w), dtype=cp.float32)
    coords[0] = rr[:, None]
    coords[1] = cc[None, :]
    return map_coordinates(coarse, coords, order=1, mode="nearest").astype(cp.float32)


def _upsample_coarse_response_block(block, *, coarse, full_h, full_w,
                                    tile_origin=None, tile_full_shape=None,
                                    block_info=None):
    """Bilinearly sample a small coarse response at this block's global coords.

    On the Dask backend the coarse response spans the same full raster as the
    block, so the block's ``array-location`` is its true global position.  On the
    tile backend the dask array is a single tile, so its location is block-local
    ``(0..win)``: pass ``tile_origin`` (the tile window's global pixel offset) and
    ``tile_full_shape`` (the full raster shape) to shift the sampling onto the
    global overview -- this is what keeps the large-radius field seam-free across
    tiles.  Both ``None`` (Dask) preserves the original behaviour exactly.
    """
    if block_info is not None and block_info.get(0) is not None:
        loc = block_info[0]["array-location"]
        r0, r1 = int(loc[0][0]), int(loc[0][1])
        c0, c1 = int(loc[1][0]), int(loc[1][1])
    else:  # pragma: no cover - non-dask fallback
        r0, c0 = 0, 0
        r1, c1 = block.shape[0], block.shape[1]
    if tile_origin is not None:
        r0 += int(tile_origin[0])
        r1 += int(tile_origin[0])
        c0 += int(tile_origin[1])
        c1 += int(tile_origin[1])
    if tile_full_shape is not None:
        full_h, full_w = int(tile_full_shape[0]), int(tile_full_shape[1])
    return _bilinear_sample_coarse(coarse, r0, r1, c0, c1, full_h, full_w)


def _nanmean_dispatch(a, axis=None, **kwargs):
    """CuPy-backed nanmean tolerant of da.coarsen's numpy meta probe.

    ``da.coarsen`` infers the output meta by calling the reduction once with a
    tiny *numpy* sample array, while the real chunks are CuPy. ``cp.nanmean``
    rejects numpy input outright (TypeError). Promote a numpy probe to CuPy so
    the call succeeds and the coarsened array keeps a CuPy meta consistent with
    its real chunks.
    """
    if isinstance(a, np.ndarray):
        a = cp.asarray(a)
    return cp.nanmean(a, axis=axis, **kwargs)


def coarse_large_radius_response(
    gpu_arr: da.Array,
    *,
    block_fn,
    radius_kw: str,
    radius: float,
    factor: int,
    depth_for_radius,
    pixel_size: float = 1.0,
    pixel_scale_x: Optional[float] = None,
    pixel_scale_y: Optional[float] = None,
    coarse_cache: Optional[dict] = None,
    coarse_dem: Optional[cp.ndarray] = None,
    coarse_decimation: Optional[float] = None,
    tile_origin=None,
    tile_full_shape=None,
    **block_kwargs,
) -> da.Array:
    """One large-radius spatial response computed on a coarsened DEM, upsampled.

    The DEM is da.coarsen-downsampled by ``factor`` (NaN-aware mean), the block
    function runs there with the radius / metric spacing scaled by ``factor``,
    and the small coarse result is bilinearly upsampled to full resolution.
    Intended for projected DEMs (metric pixel scales scale linearly with factor).
    ``coarse_cache`` (a dict) avoids re-coarsening the array for multiple radii.

    ``coarse_dem`` (a concrete CuPy overview read once from the COG, with its own
    ``coarse_decimation``) short-circuits the da.coarsen pass: the block function
    runs on that single overview array directly.  This is the unified, TopoUSM Fast-style
    coarse source -- every algorithm derives its large radii from the same cheap
    decimated overview read instead of a full-resolution da.coarsen per algorithm.
    """
    H, W = int(gpu_arr.shape[0]), int(gpu_arr.shape[1])
    if coarse_dem is not None and coarse_decimation is not None:
        # Unified overview path: one decimated read serves all radii / algorithms.
        fac = float(coarse_decimation)
        r_coarse = max(1, int(round(float(radius) / fac)))
        kw = dict(block_kwargs)
        # A whole-raster coarse response already has global thresholds. Reusing
        # a full-resolution small-radius stats entry under the rounded coarse
        # radius can collide (e.g. radius 320 / factor 8 -> key 40).
        kw.pop("grad_stats_map", None)
        kw[radius_kw] = r_coarse
        kw["pixel_size"] = float(pixel_size) * fac
        if pixel_scale_x is not None:
            kw["pixel_scale_x"] = float(pixel_scale_x) * fac
        if pixel_scale_y is not None:
            kw["pixel_scale_y"] = float(pixel_scale_y) * fac
        # The overview is the whole (small) array -> run the block once, no halo.
        # Scatter the resulting coarse response so the large CuPy array is not
        # serialized into every downstream map_blocks task.
        coarse_resp = _scatter_if_client(block_fn(coarse_dem, **kw).astype(cp.float32))
        upsampled = gpu_arr.map_blocks(
            _upsample_coarse_response_block,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            coarse=coarse_resp,
            full_h=H,
            full_w=W,
            tile_origin=tile_origin,
            tile_full_shape=tile_full_shape,
        )
        return da.where(da.isnan(gpu_arr), cp.float32(cp.nan), upsampled)

    if coarse_cache is not None and "coarse" in coarse_cache:
        coarse = coarse_cache["coarse"]
    else:
        coarse = da.coarsen(_nanmean_dispatch, gpu_arr, {0: factor, 1: factor}, trim_excess=True)
        if coarse_cache is not None:
            # Materialise the (small) coarse DEM once so multiple large radii reuse
            # it instead of re-reading and re-coarsening the full-resolution array
            # for each radius (otherwise N large radii = N full-DEM reads).
            try:
                coarse = coarse.persist()
            except Exception:
                pass
            coarse_cache["coarse"] = coarse

    r_coarse = max(1, int(round(float(radius) / float(factor))))
    kw = dict(block_kwargs)
    kw.pop("grad_stats_map", None)
    kw[radius_kw] = r_coarse
    kw["pixel_size"] = float(pixel_size) * float(factor)
    if pixel_scale_x is not None:
        kw["pixel_scale_x"] = float(pixel_scale_x) * float(factor)
    if pixel_scale_y is not None:
        kw["pixel_scale_y"] = float(pixel_scale_y) * float(factor)

    coarse_resp = coarse.map_overlap(
        block_fn,
        depth=int(depth_for_radius(r_coarse)),
        boundary="reflect",
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        **kw,
    ).compute()

    upsampled = gpu_arr.map_blocks(
        _upsample_coarse_response_block,
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        coarse=coarse_resp,
        full_h=H,
        full_w=W,
        tile_origin=tile_origin,
        tile_full_shape=tile_full_shape,
    )
    # The coarse field was filled (cliff-free) where the DEM is NoData; restore
    # NaN at the true NoData footprint so the large-radius response does not leak
    # finite values into the exterior.
    return da.where(da.isnan(gpu_arr), cp.float32(cp.nan), upsampled)


def multiscale_response_fields(
    gpu_arr: da.Array,
    scales,
    *,
    block_fn,
    depth_for_scale,
    radius_kw: str = "scale",
    is_large=None,
    pixel_size: float = 1.0,
    pixel_scale_x: Optional[float] = None,
    pixel_scale_y: Optional[float] = None,
    is_geographic: bool = False,
    coarse_cache: Optional[dict] = None,
    coarse_dem: Optional[cp.ndarray] = None,
    coarse_decimation: Optional[float] = None,
    tile_origin=None,
    tile_full_shape=None,
    **block_kwargs,
) -> List[da.Array]:
    """Per-scale response fields as dask arrays, large scales via the coarse path.

    Shared by every spatial / multi-scale algorithm.  For each scale,
    ``block_fn(block, <radius_kw>=<scale>, pixel_size=..., ...)`` computes that
    scale's response on a CuPy block.  A scale is "large" when ``is_large(scale)``
    is true (default: ``depth_for_scale(scale) > Constants.MAX_DEPTH``); on a
    projected DEM a large scale is computed on a globally-coarsened copy and
    upsampled (no oversized per-chunk halo), exactly as ``multiscale_terrain``
    does, otherwise it is a bounded ``map_overlap``.  This keeps large ``--radii``
    accurate without the rechunk-merge OOM.

    ``radius_kw`` is the block_fn's radius/scale keyword ("scale" for the intrinsic
    algorithms, "radius" for the spatial-switch algorithms).  ``is_large`` lets the
    switch algorithms keep their chunk-relative ``large_radius_threshold`` instead
    of the MAX_DEPTH rule.  All returned arrays share ``gpu_arr``'s chunking, so a
    downstream ``da.map_blocks/map_overlap(combine, gpu_arr, *fields)`` aligns
    block-wise.
    """
    # Coarsen for large radii regardless of CRS: the coarse path is pixel-based
    # and scales pixel_size / pixel_scale_x / pixel_scale_y independently by the
    # factor, so it stays correct (and anisotropy-preserving) for geographic DEMs
    # too.  Disabling it there forced large radii through a near-chunk halo that
    # exhausts VRAM.  (is_geographic is kept for API compatibility / callers.)
    F = coarsen_factor_for_shape(gpu_arr.shape)
    # Tile backend: gpu_arr is a single (small) tile so coarsen_factor is 1, which
    # would force every large radius through a tile-local halo (seams).  When the
    # orchestrator injected a global overview (coarse_dem) and the tile origin, take
    # the coarse path for large radii regardless of F -- coarse_large_radius_response
    # short-circuits onto that concrete overview anyway.  Dask (tile_origin None) is
    # unchanged: the gate stays F > 1.
    _coarse_ok_tile = (coarse_dem is not None and tile_origin is not None)
    if coarse_cache is None:
        coarse_cache = {}
    # The map_overlap halo must stay below the smallest chunk; a halo >= a chunk
    # makes dask rechunk that field (fewer blocks), so it no longer aligns with
    # gpu_arr in the downstream combine ("shapes do not align").  This is the only
    # cap applied here -- callers control the actual halo via depth_for_scale, and
    # the coarse-vs-full split via is_large -- so each algorithm keeps its exact
    # prior halo behavior.
    min_chunk = min((min(ax) for ax in gpu_arr.chunks), default=1) if hasattr(gpu_arr, "chunks") else 1
    chunk_cap = max(1, int(min_chunk) - 1)
    fields: List[da.Array] = []
    for s in scales:
        sv = float(s)
        d = int(depth_for_scale(sv))
        large = is_large(sv) if is_large is not None else (d > Constants.MAX_DEPTH)
        if large and (F > 1 or _coarse_ok_tile):
            fields.append(coarse_large_radius_response(
                gpu_arr, block_fn=block_fn, radius_kw=radius_kw, radius=sv,
                factor=F,
                depth_for_radius=lambda sc: max(1, int(depth_for_scale(sc))),
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, coarse_cache=coarse_cache,
                coarse_dem=coarse_dem, coarse_decimation=coarse_decimation,
                tile_origin=tile_origin, tile_full_shape=tile_full_shape,
                **block_kwargs))
        else:
            fields.append(gpu_arr.map_overlap(
                block_fn, depth=max(1, min(d, chunk_cap)),
                boundary="reflect", dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                pixel_size=pixel_size,
                pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
                **{radius_kw: sv}, **block_kwargs))
    return fields


def _smooth_for_radius(
    block: cp.ndarray,
    radius: float,
    *,
    pixel_size: float = 1.0,
    algorithm_name: str = "default",
) -> cp.ndarray:
    """NaN-aware gaussian smoothing controlled by spatial radius."""
    r = max(1.0, float(radius))
    if r <= 1.0:
        return block
    factor = _radius_to_downsample_factor(
        r,
        block_shape=block.shape,
        pixel_size=pixel_size,
        algorithm_name=algorithm_name,
    )
    if factor <= 1:
        sigma = max(0.5, r / 2.0)
        smoothed, _ = handle_nan_with_gaussian(block, sigma=sigma, mode="nearest")
        return smoothed

    reduced = _downsample_nan_aware(block, factor)
    sigma_small = max(0.5, (r / factor) / 2.0)
    smoothed_small, _ = handle_nan_with_gaussian(reduced, sigma=sigma_small, mode="nearest")
    return _upsample_to_shape(smoothed_small, block.shape)


def _radius_to_downsample_factor(
    radius: float,
    *,
    block_shape: Optional[Tuple[int, int]] = None,
    pixel_size: float = 1.0,
    algorithm_name: str = "default",
    base_radius: float = 24.0,
    max_factor: int = 16,
) -> int:
    """
    Dynamic downsample factor from radius + workload context.
    Returns power-of-two factors: 1,2,4,8,...
    """
    r = max(1.0, float(radius))
    px = max(1e-3, float(pixel_size) if pixel_size else 1.0)

    algo_factor_map = {
        "topousm_fast": 1.15,
        "hillshade": 1.0,
        "slope": 1.0,
        "specular": 1.4,
        "atmospheric_scattering": 1.05,
        "curvature": 1.1,
        "ambient_occlusion": 1.5,
        "openness": 1.4,
        "multi_light_uncertainty": 1.25,
    }
    algo_factor = float(algo_factor_map.get(str(algorithm_name), 1.0))

    # Deterministic across tile/Dask and ragged edge chunks: do not derive the
    # approximation factor from the current block shape.  A shape-dependent
    # factor made identical radii use different downsample approximations on
    # Dask edge chunks and tile backends.  Keep the calibration anchored to the
    # nominal 1 Mpix workload used by the original heuristic.
    block_factor = 1.0

    # 0.5m should be somewhat more aggressive than 1m.
    resolution_factor = max(1.0, 1.0 / px)

    score = (r / max(1.0, base_radius)) * algo_factor * block_factor * (resolution_factor ** 0.35)
    if score <= 1.0:
        return 1

    # Convert to power-of-two scaling for stable kernels.
    factor = 2 ** int(np.floor(np.log2(score)))
    factor = int(max(1, min(factor, max_factor)))
    return factor


def _downsample_nan_aware(block: cp.ndarray, factor: int) -> cp.ndarray:
    """Downsample by ``factor`` without leaking NoData across the data boundary.

    The previous implementation filled every NoData cell with the *global* block
    mean before decimating.  Near an irregular data boundary that injects a flat
    plateau whose elevation is unrelated to the local terrain, so the subsequent
    spatial operator (AO occlusion, gradient, blur, ...) sees an artificial cliff
    and renders a dark halo just inside the boundary.

    Instead we compute a **valid-weighted mean** (each coarse cell averages only
    its finite contributors, so boundary cells are not diluted by NoData) and
    fill the remaining voids with a smooth, valid-weighted extrapolation -- the
    same low-frequency strategy used by the preprocessing fill.  The result is
    finite and *cliff-free*; the true NoData footprint is reapplied to the final
    output by the pipeline's nodata pass.
    """
    if factor <= 1:
        return block
    f = int(factor)
    h, w = block.shape[:2]
    out_h = max(1, (int(h) + f - 1) // f)
    out_w = max(1, (int(w) + f - 1) // f)

    # True block-mean (area-average) decimation over the finite contributors of
    # each f x f cell.  The previous bilinear ``zoom`` only sampled a 2x2
    # neighbourhood per output pixel, which aliases badly at factors >= 4 and was
    # not the valid-weighted *mean* its docstring promised.  NaN padding keeps the
    # ragged right/bottom edge out of the average.
    work = block.astype(cp.float32, copy=False)
    pad_h = out_h * f - int(h)
    pad_w = out_w * f - int(w)
    if pad_h or pad_w:
        work = cp.pad(work, ((0, pad_h), (0, pad_w)),
                      mode="constant", constant_values=cp.nan)
    cells = work.reshape(out_h, f, out_w, f)
    finite = cp.isfinite(cells)
    cnt = finite.sum(axis=(1, 3), dtype=cp.float32)
    total = cp.where(finite, cells, cp.float32(0)).sum(axis=(1, 3), dtype=cp.float32)
    coarse = cp.where(cnt > 0, total / cp.maximum(cnt, cp.float32(1)),
                      cp.float32(cp.nan)).astype(cp.float32)

    # Fill only thin, well-enclosed coarse voids; preserve NaN over the large
    # exterior NoData (sea / dataset outside).  The previous behaviour
    # extrapolated *every* void -- including the border-connected exterior --
    # falling back to the coarse global mean where the Gaussian support did not
    # reach.  That injected a flat plateau at the mean elevation just outside the
    # data boundary.  The downstream large-radius operators (TopoUSM Fast mean-subtraction,
    # AO occlusion) are NaN-aware and would down-weight a NaN exterior to zero,
    # but a *finite* plateau is not excluded: it leaks into the interior valid
    # pixels and renders a broad halo along the periphery that destroys detail.
    # Keeping the exterior as NaN lets those NaN-aware operators ignore it.
    cnan = cp.isnan(coarse)
    if bool(cnan.any()):
        cvalid = (~cnan).astype(cp.float32)
        sigma = max(1.0, float(min(coarse.shape[:2])) / 64.0)
        sv = gaussian_filter(cp.where(cnan, cp.float32(0), coarse).astype(cp.float32),
                             sigma=sigma, mode="nearest")
        sw = gaussian_filter(cvalid, sigma=sigma, mode="nearest")
        # A void is "enclosed" only when valid terrain dominates its local
        # Gaussian support (sw > 0.5).  The broad exterior NoData has sw ~ 0 and
        # is intentionally left as NaN so it cannot contaminate the boundary.
        enclosed = cnan & (sw > cp.float32(0.5))
        smooth = sv / cp.maximum(sw, cp.float32(1e-6))
        coarse = cp.where(enclosed, smooth, coarse).astype(cp.float32)
    return coarse.astype(cp.float32)


def _upsample_to_shape(block: cp.ndarray, target_shape: Tuple[int, int]) -> cp.ndarray:
    th, tw = int(target_shape[0]), int(target_shape[1])
    h, w = block.shape[:2]
    if h == th and w == tw:
        return block.astype(cp.float32, copy=False)
    sy = th / max(1, h)
    sx = tw / max(1, w)
    nan_mask = cp.isnan(block)
    if not bool(nan_mask.any()):
        out = zoom(block, zoom=(sy, sx), order=1, mode="nearest").astype(cp.float32)
        pad_h, pad_w = max(0, th - out.shape[0]), max(0, tw - out.shape[1])
        if pad_h or pad_w:
            out = cp.pad(out, ((0, pad_h), (0, pad_w)), mode="edge")
        return out[:th, :tw]
    # NaN-aware bilinear upsample: interpolate valid contributors only so the
    # exterior NoData (now preserved as NaN by _downsample_nan_aware) does not
    # bleed a NaN fringe into the interior valid pixels.  Cells whose upsampled
    # valid weight is ~0 (the true exterior) are restored to NaN.
    valid = (~nan_mask).astype(cp.float32)
    filled = cp.where(nan_mask, cp.float32(0), block).astype(cp.float32)
    num = zoom(filled, zoom=(sy, sx), order=1, mode="nearest")
    den = zoom(valid, zoom=(sy, sx), order=1, mode="nearest")
    out = cp.where(den > cp.float32(1e-3), num / cp.maximum(den, cp.float32(1e-6)),
                   cp.float32(cp.nan)).astype(cp.float32)
    pad_h, pad_w = max(0, th - out.shape[0]), max(0, tw - out.shape[1])
    if pad_h or pad_w:
        out = cp.pad(out, ((0, pad_h), (0, pad_w)), mode="edge")
    return out[:th, :tw]


def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """Restore NaN positions."""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result


def pyramid_fill_surface(coarse: cp.ndarray, valid: cp.ndarray) -> cp.ndarray:
    """Smooth, cliff-free void fill via a push-pull (multigrid) image pyramid (GPU).

    Thin CuPy wrapper over the backend-neutral ``_pyramid_fill.pushpull_fill`` so the
    GPU pipeline and the CPU preprocessing fill share one implementation.  This is
    the multiscale replacement for the old single-grid "nearest valid + one
    Gaussian" fill, which injected phantom relief; push-pull instead solves a
    membrane-like (minimal-curvature) interpolation that fills small voids from fine
    levels and large voids from coarse levels without inventing relief.

    ``coarse`` : float32 grid (values at invalid cells are ignored).
    ``valid``  : bool mask of finite/known cells.
    Returns a fully-finite float32 surface; ``valid`` cells are preserved exactly.
    """
    from ._pyramid_fill import pushpull_fill

    return pushpull_fill(coarse, valid, xp=cp, zoom=zoom)


def _hybrid_combine_wrapper(dem, *small_blocks, _scales_order, _small_scales,
                            _large_scales, _large_list, _full_h, _full_w,
                            _combine_fn, _combine_kwargs,
                            _tile_origin=None, _tile_full_shape=None, block_info=None):
    """Assemble the full ordered per-scale field list for one output block and run
    the algorithm's pointwise combine.

    Small scales arrive as already-trimmed, block-aligned Dask fields (positional
    ``small_blocks``).  Large scales are bilinearly sampled from their concrete
    coarse (overview) field at this block's *global* pixel coordinates -- valid
    only under ``map_blocks`` (depth 0), where ``array-location`` is the true
    global core position (under ``map_overlap`` it is the padded-coordinate
    position, so this combine MUST stay depth-0; the per-scale fields already
    carry their own halos).

    On the tile backend ``array-location`` is block-local (single-tile array), so
    ``_tile_origin`` / ``_tile_full_shape`` shift the large-field sampling onto the
    tile's true global position; both ``None`` (Dask) is the original behaviour."""
    if block_info is not None and block_info.get(0) is not None:
        loc = block_info[0]["array-location"]
        r0, r1 = int(loc[0][0]), int(loc[0][1])
        c0, c1 = int(loc[1][0]), int(loc[1][1])
    else:  # pragma: no cover - non-dask fallback
        r0, c0 = 0, 0
        r1, c1 = int(dem.shape[0]), int(dem.shape[1])
    if _tile_origin is not None:
        r0 += int(_tile_origin[0])
        r1 += int(_tile_origin[0])
        c0 += int(_tile_origin[1])
        c1 += int(_tile_origin[1])
    if _tile_full_shape is not None:
        _full_h, _full_w = int(_tile_full_shape[0]), int(_tile_full_shape[1])
    small_map = {float(s): b for s, b in zip(_small_scales, small_blocks)}
    large_map = {float(s): cf for s, cf in zip(_large_scales, _large_list)}
    fields = []
    for s in _scales_order:
        k = float(s)
        if k in small_map:
            fields.append(small_map[k])
        else:
            up = _bilinear_sample_coarse(large_map[k], r0, r1, c0, c1, _full_h, _full_w)
            fields.append(up.astype(cp.float32))
    return _combine_fn(dem, *fields, **_combine_kwargs)


def hybrid_multiscale_response_combine(
    gpu_arr: da.Array,
    scales,
    *,
    small_block_fn,
    combine_fn,
    depth_for_scale,
    large_fields: dict,
    full_shape,
    radius_kw: str = "scale",
    pixel_size: float = 1.0,
    pixel_scale_x: Optional[float] = None,
    pixel_scale_y: Optional[float] = None,
    combine_kwargs: Optional[dict] = None,
    tile_origin=None,
    tile_full_shape=None,
    **block_kwargs,
) -> da.Array:
    """TopoUSM Fast-style hybrid multiscale combine (bounded VRAM + accurate large scales).

    Small scales (those NOT in ``large_fields``) are computed at full resolution as
    bounded-halo ``map_overlap`` fields.  Large scales arrive precomputed as
    concrete coarse fields in ``large_fields`` (``{int(scale): cupy_field}``), each
    produced by running ``small_block_fn`` on a coarsened overview; they are sampled
    per output block by global coordinates and bilinearly upsampled inside a single
    depth-0 ``map_blocks`` combine.  This keeps per-chunk device memory bounded on
    huge streaming rasters (no per-large-scale Dask field whose intermediates
    accumulate) while keeping the large, low-frequency scales accurate (true
    overview response instead of a MAX_DEPTH-truncated halo).

    ``combine_fn(dem_block, *ordered_fields, **combine_kwargs)`` is the algorithm's
    existing per-pixel combine (e.g. roughness->fractal feature, smooths->saliency).
    """
    combine_kwargs = dict(combine_kwargs or {})
    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    min_chunk = min((min(ax) for ax in gpu_arr.chunks), default=1) if hasattr(gpu_arr, "chunks") else 1
    chunk_cap = max(1, int(min_chunk) - 1)
    large_fields = _resolve_scattered(large_fields)
    large_fields = {float(scale): value for scale, value in large_fields.items()}
    large_keys = set(large_fields)
    scales_f = [float(s) for s in scales]
    small_scales = [s for s in scales_f if s not in large_keys]
    large_scales = [s for s in scales_f if s in large_keys]
    small_fields = [
        gpu_arr.map_overlap(
            small_block_fn,
            depth=max(1, min(int(depth_for_scale(s)), chunk_cap)),
            boundary="reflect", dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
            pixel_scale_y=pixel_scale_y, **{radius_kw: s}, **block_kwargs)
        for s in small_scales
    ]
    large_list = [large_fields[s] for s in large_scales]
    return da.map_blocks(
        _hybrid_combine_wrapper, gpu_arr, *small_fields,
        dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
        _scales_order=scales_f, _small_scales=small_scales,
        _large_scales=large_scales, _large_list=large_list,
        _full_h=full_h, _full_w=full_w,
        _combine_fn=combine_fn, _combine_kwargs=combine_kwargs,
        _tile_origin=tile_origin, _tile_full_shape=tile_full_shape)


def read_overview_coarse_dem(src_cog: str, *, sample_max: int = 2048):
    """Read one decimated overview of the DEM as a concrete CuPy array.

    Returns ``(coarse_dem, decimation)`` (NaN at NoData) or ``(None, 1.0)`` on
    failure.  This single cheap read is the unified coarse source shared by every
    algorithm's large-radius path (instead of a full-resolution da.coarsen each)."""
    try:
        import rasterio
        from rasterio.enums import Resampling
    except Exception:
        return None, 1.0
    try:
        with rasterio.open(src_cog) as src:
            source_width, source_height = int(src.width), int(src.height)
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            # Derive both axes from the SAME scale (no per-axis floor): a floor
            # on one axis would make the actual decimation anisotropic on very
            # elongated rasters while callers scale radii/pixel sizes by the
            # single returned ``decimation``.
            sw = max(1, int(round(src.width / scale)))
            sh = max(1, int(round(src.height / scale)))
            sample_ma = src.read(
                1, out_shape=(sh, sw), resampling=Resampling.average,
                out_dtype=np.float32, masked=True)
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)
            nodata = src.nodata
        if nodata is not None and not np.isnan(float(nodata)):
            sample = np.where(
                np.isclose(sample, float(nodata), rtol=0.0, atol=1e-6),
                np.nan, sample).astype(np.float32, copy=False)
        actual_decimation = 0.5 * (
            source_width / float(sw) + source_height / float(sh)
        )
        return cp.asarray(sample, dtype=cp.float32), float(actual_decimation)
    except Exception:
        return None, 1.0


def compute_overview_scale_fields(
    src_cog: str,
    *,
    large_radii,
    block_fn,
    sample_max: int = 2048,
    coarse_dem: Optional[cp.ndarray] = None,
    decimation: Optional[float] = None,
):
    """Per-large-scale response fields from the COG overview (TopoUSM Fast-style fast path).

    Runs ``block_fn(coarse, scale=r/decimation)`` for each large radius on a single
    decimated overview, returning ``({int(radius): cupy_field}, decimation)``.  Pass
    a pre-read ``coarse_dem`` + ``decimation`` to reuse the shared overview read;
    otherwise it reads one itself.  Returns ``({}, 1.0)`` if there are no large
    radii; ``(None, 1.0)`` on failure so callers fall back to the single-block
    path."""
    if not large_radii:
        return {}, 1.0
    if coarse_dem is None or decimation is None:
        coarse_dem, decimation = read_overview_coarse_dem(src_cog, sample_max=sample_max)
    if coarse_dem is None:
        return None, 1.0
    try:
        fields = {}
        for r in large_radii:
            r_c = max(1.0, float(r) / float(decimation))
            fields[int(round(float(r)))] = block_fn(coarse_dem, scale=r_c).astype(cp.float32)
        return fields, float(decimation)
    except Exception:
        return None, 1.0


__all__ = [
    "handle_nan_with_gaussian",
    "handle_nan_with_uniform",
    "handle_nan_for_gradient",
    "_normalize_spatial_radii",
    "_resolve_spatial_radii_weights",
    "_combine_multiscale_dask",
    "_resolve_scattered",
    "_scatter_if_client",
    "_smooth_for_radius",
    "_radius_to_downsample_factor",
    "_downsample_nan_aware",
    "_upsample_to_shape",
    "restore_nan",
    "pyramid_fill_surface",
    "large_radius_threshold",
    "coarsen_factor_for_shape",
    "coarse_large_radius_response",
    "_bilinear_sample_coarse",
    "hybrid_multiscale_response_combine",
    "compute_overview_scale_fields",
    "read_overview_coarse_dem",
]
