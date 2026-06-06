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
from .common.spatial_mode import determine_spatial_radii, determine_spatial_profile


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
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
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
        return determine_spatial_radii(pixel_size=pixel_size)
    out: List[int] = []
    for r in radii:
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        if rv > 0:
            out.append(rv)
    if not out:
        return determine_spatial_radii(pixel_size=pixel_size)
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
) -> Tuple[List[int], Optional[List[float]]]:
    """Resolve radii/weights with YAML presets when user values are omitted."""
    if radii is None:
        auto_radii, auto_weights = determine_spatial_profile(pixel_size=pixel_size)
        return auto_radii, auto_weights if weights is None else weights

    resolved_radii = _normalize_spatial_radii(radii, pixel_size)
    if not isinstance(weights, (list, tuple)) or len(weights) != len(resolved_radii):
        return resolved_radii, None

    cleaned: List[float] = []
    for w in weights:
        try:
            fv = float(w)
        except (TypeError, ValueError):
            return resolved_radii, None
        cleaned.append(fv if np.isfinite(fv) and fv > 0 else 0.0)
    s = float(sum(cleaned))
    if s <= 0:
        return resolved_radii, None
    return resolved_radii, [v / s for v in cleaned]


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
    if len(responses) == 1:
        return responses[0]

    stacked = da.stack(responses, axis=0)
    agg_norm = str(agg or "mean").lower()
    if agg_norm == "stack":
        return stacked
    if agg_norm == "max":
        return da.max(stacked, axis=0)
    if agg_norm == "min":
        return da.min(stacked, axis=0)
    if agg_norm == "sum":
        return da.sum(stacked, axis=0)

    if isinstance(weights, (list, tuple)) and len(weights) == len(responses):
        w = np.asarray(weights, dtype=np.float32)
        if np.isfinite(w).all() and w.sum() > 0:
            w = w / w.sum()
            out = responses[0] * float(w[0])
            for i in range(1, len(responses)):
                out = out + responses[i] * float(w[i])
            return out
    return da.mean(stacked, axis=0)


# ---------------------------------------------------------------------------
# Large-radius-from-overview helpers (shared by spatial-mode algorithms)
# ---------------------------------------------------------------------------
def large_radius_threshold(gpu_arr: da.Array, fallback: int) -> int:
    """Radii above this are computed from a coarsened copy (no large halo).

    Default = max(256, min_chunk // 16), matching the RVI threshold.
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

    ch, cw = coarse.shape
    h = int(r1 - r0)
    w = int(c1 - c0)
    rr = (cp.arange(r0, r1, dtype=cp.float32) + cp.float32(0.5)) * cp.float32(ch / float(full_h)) - cp.float32(0.5)
    cc = (cp.arange(c0, c1, dtype=cp.float32) + cp.float32(0.5)) * cp.float32(cw / float(full_w)) - cp.float32(0.5)
    coords = cp.empty((2, h, w), dtype=cp.float32)
    coords[0] = rr[:, None]
    coords[1] = cc[None, :]
    return map_coordinates(coarse, coords, order=1, mode="nearest").astype(cp.float32)


def _upsample_coarse_response_block(block, *, coarse, full_h, full_w, block_info=None):
    """Bilinearly sample a small coarse response at this block's global coords.

    The coarse response is derived from the *same* array (Dask: the full raster;
    tile: the per-tile padded window), so sampling at block-local coordinates is
    correct and seam-free -- no global offset is needed.
    """
    if block_info is not None and block_info.get(0) is not None:
        loc = block_info[0]["array-location"]
        r0, r1 = int(loc[0][0]), int(loc[0][1])
        c0, c1 = int(loc[1][0]), int(loc[1][1])
    else:  # pragma: no cover - non-dask fallback
        r0, c0 = 0, 0
        r1, c1 = block.shape[0], block.shape[1]
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
    **block_kwargs,
) -> da.Array:
    """One large-radius spatial response computed on a coarsened DEM, upsampled.

    The DEM is da.coarsen-downsampled by ``factor`` (NaN-aware mean), the block
    function runs there with the radius / metric spacing scaled by ``factor``,
    and the small coarse result is bilinearly upsampled to full resolution.
    Intended for projected DEMs (metric pixel scales scale linearly with factor).
    ``coarse_cache`` (a dict) avoids re-coarsening the array for multiple radii.
    """
    H, W = int(gpu_arr.shape[0]), int(gpu_arr.shape[1])
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
        if F > 1 and large:
            fields.append(coarse_large_radius_response(
                gpu_arr, block_fn=block_fn, radius_kw=radius_kw, radius=sv,
                factor=F,
                depth_for_radius=lambda sc: max(1, int(depth_for_scale(sc))),
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, coarse_cache=coarse_cache, **block_kwargs))
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
        "rvi": 1.15,
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

    block_factor = 1.0
    if block_shape is not None and len(block_shape) >= 2:
        h = max(1, int(block_shape[0]))
        w = max(1, int(block_shape[1]))
        block_pixels = float(h * w)
        # Mild scaling by block area to avoid over-aggressive shrink on small chunks.
        block_factor = max(1.0, (block_pixels / 1_000_000.0) ** 0.5)

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
    nan_mask = cp.isnan(block)
    h, w = block.shape[:2]
    out_h = max(1, (int(h) + int(factor) - 1) // int(factor))
    out_w = max(1, (int(w) + int(factor) - 1) // int(factor))
    sy = out_h / max(1, h)
    sx = out_w / max(1, w)

    if not nan_mask.any():
        work = block.astype(cp.float32, copy=False)
        return zoom(work, zoom=(sy, sx), order=1, mode="nearest").astype(cp.float32)

    # Valid-weighted decimation: average finite contributors only.
    valid = (~nan_mask).astype(cp.float32)
    filled0 = cp.where(nan_mask, cp.float32(0), block).astype(cp.float32)
    num = zoom(filled0, zoom=(sy, sx), order=1, mode="nearest")
    den = zoom(valid, zoom=(sy, sx), order=1, mode="nearest")
    coarse = cp.where(den > 1e-6, num / cp.maximum(den, cp.float32(1e-6)),
                      cp.float32(cp.nan)).astype(cp.float32)

    # Fill only thin, well-enclosed coarse voids; preserve NaN over the large
    # exterior NoData (sea / dataset outside).  The previous behaviour
    # extrapolated *every* void -- including the border-connected exterior --
    # falling back to the coarse global mean where the Gaussian support did not
    # reach.  That injected a flat plateau at the mean elevation just outside the
    # data boundary.  The downstream large-radius operators (RVI mean-subtraction,
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
    return out[:th, :tw]


def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """Restore NaN positions."""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result


__all__ = [
    "handle_nan_with_gaussian",
    "handle_nan_with_uniform",
    "handle_nan_for_gradient",
    "_normalize_spatial_radii",
    "_resolve_spatial_radii_weights",
    "_combine_multiscale_dask",
    "_smooth_for_radius",
    "_radius_to_downsample_factor",
    "_downsample_nan_aware",
    "_upsample_to_shape",
    "restore_nan",
    "large_radius_threshold",
    "coarsen_factor_for_shape",
    "coarse_large_radius_response",
    "_bilinear_sample_coarse",
]
