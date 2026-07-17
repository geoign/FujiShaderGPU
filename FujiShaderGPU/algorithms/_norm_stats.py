"""Full-resolution, full-extent output normalization statistics.

Backend-neutral (rasterio + cupy + the algorithms' own block/stat functions) so
both the Dask pipeline and the Windows tile pipeline share one implementation.

Normalized algorithms map a robust statistic of their *full-resolution* output to
display magnitude ~1.0.  Computing that statistic from a decimated overview
under-estimates the scale (high-frequency detail is lost to decimation, and large
radii/kernels become nonsensical on the small grid), which over-amplifies the
output and blows out the integer (uint8/int16) encoding.  This module instead
runs each algorithm's raw block function (normalize off) on several
FULL-RESOLUTION windows stratified across the valid-data extent and pools the
interior pixels, so the magnitude is correct and singular outliers fall into the
unclipped tail rather than setting the scale.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import cupy as cp

logger = logging.getLogger(__name__)

# algorithm -> (impl module, raw block function, stat function).  The block runs
# with normalize=False and the main-pass parameters; the stat function returns the
# algorithm's native (offset, ..., scale) tuple consumed as ``global_stats``.
_NORM_STAT_SPECS = {
    "topousm_fast": ("_impl_topousm_fast", "compute_topousm_fast_efficient_block", "topousm_fast_stat_func"),
    "fractal_anomaly": ("_impl_fractal_anomaly",
                        "compute_fractal_dimension_block", "fractal_stat_func"),
    "scale_space_surprise": ("_impl_experimental",
                            "compute_scale_space_surprise_block",
                            "scale_space_surprise_stat_func"),
    "visual_saliency": ("_impl_visual_saliency",
                        "compute_visual_saliency_block", "visual_saliency_stat_func"),
    "multiscale_terrain": ("_impl_multiscale_terrain",
                          "compute_multiscale_combined_raw", "multiscale_stat_func"),
    # Bounded [0,1] maps concentrated in a narrow high band: data-driven
    # [p1, p99] -> [0, 1] contrast stretch so the integer codes are not wasted.
    "ambient_occlusion": ("_impl_ambient_occlusion",
                         "compute_ambient_occlusion_block",
                         "robust_unsigned_stretch_stat_func"),
    "openness": ("_impl_openness", "compute_openness_vectorized",
                "robust_unsigned_stretch_stat_func"),
    # New (2026-07) algorithms.  The "raw" block output feeding each stat:
    # structure_tensor / scale_drift -> the selected output field itself;
    # frangi -> the Hessian structure energy S (stat = global c);
    # phase_congruency -> the smallest-scale amplitude (stat = noise median);
    # tv_decomposition -> the signed texture v (stat = p90 |v| tanh scale).
    "structure_tensor": ("_impl_structure_tensor",
                        "compute_structure_tensor_block", "st_stretch_stat_func"),
    "frangi": ("_impl_frangi", "compute_frangi_block", "frangi_c_stat_func"),
    "phase_congruency": ("_impl_phase_congruency",
                        "compute_phase_congruency_block", "pc_noise_stat_func"),
    "tv_decomposition": ("_impl_tv_decomposition",
                        "compute_tv_texture_block", "tv_texture_stat_func"),
    "scale_drift": ("_impl_scale_drift",
                   "compute_scale_drift_block", "drift_stretch_stat_func"),
}


def stratified_windows(
    width: int,
    height: int,
    by0: int,
    by1: int,
    bx0: int,
    bx1: int,
    *,
    grid: int = 3,
    tile: int = 4096,
) -> list:
    """Unique full-resolution sample windows stratified over a bounding box.

    Returns ``[(wy0, wx0, win_w, win_h), ...]`` for a ``grid x grid`` layout of
    ``tile``-sized windows centred on the valid-data bounding box
    ``[by0:by1, bx0:bx1]`` (raster size ``width x height``).  Duplicate windows
    are dropped: on rasters smaller than ``grid*tile`` the grid cells collapse
    onto (nearly) the same window, and pooling the same pixels ``grid**2`` times
    is pure waste.  Shared by every stratified global-stats pre-pass
    (display-range / npr gradient / fractal relief / specular roughness)."""
    cell_h = max(1, (int(by1) - int(by0)) // int(grid))
    cell_w = max(1, (int(bx1) - int(bx0)) // int(grid))
    out = []
    seen = set()
    for gy in range(int(grid)):
        for gx in range(int(grid)):
            ccy = int(by0) + gy * cell_h + cell_h // 2
            ccx = int(bx0) + gx * cell_w + cell_w // 2
            wy0 = int(min(max(0, ccy - tile // 2), max(0, int(height) - tile)))
            wx0 = int(min(max(0, ccx - tile // 2), max(0, int(width) - tile)))
            tw, th = min(tile, int(width) - wx0), min(tile, int(height) - wy0)
            key = (wy0, wx0, tw, th)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _norm_stat_max_scale(merged: dict) -> float:
    """Largest user-facing pixel scale/radius among algorithm parameters."""
    vals = []
    for key in ("radii", "scales"):
        v = merged.get(key)
        if isinstance(v, (list, tuple)) and v:
            vals.append(max(float(x) for x in v))
    ks = merged.get("kernel_size")
    if ks:
        vals.append(float(ks))
    # Algorithm-specific implicit supports used by stats block functions.
    if str(merged.get("component", "texture")).lower() == "texture" and merged.get("tv_scale"):
        vals.append(float(merged.get("tv_scale")))
    if merged.get("iterations"):
        vals.append(float(merged.get("iterations")))
    if merged.get("max_distance"):
        vals.append(float(merged.get("max_distance")))
    return max(vals) if vals else 16.0


def _norm_stat_halo_pixels(algorithm: str, merged: dict) -> int:
    """Conservative full-resolution footprint halo for norm-stat sample windows."""
    max_scale = float(_norm_stat_max_scale(merged))
    algo = str(algorithm)
    if algo == "topousm_fast":
        return int(max_scale + 16)
    if algo == "fractal_anomaly":
        return int(2 * max_scale + 16)
    if algo == "visual_saliency":
        return int(5 * max_scale)
    if algo in {"scale_space_surprise", "multiscale_terrain"}:
        return int(4 * max_scale + 4)
    if algo in {"structure_tensor", "frangi"}:
        sigma_d = float(merged.get("derivative_sigma", 1.0) or 1.0)
        return int(2 * max_scale + 4 * sigma_d + 8)
    if algo == "phase_congruency":
        return int(2 * max_scale + 16)
    if algo == "tv_decomposition":
        return int(float(merged.get("iterations", 120) or 120) + 4)
    if algo == "scale_drift":
        return int(4 * max_scale + 104)
    if algo in {"ambient_occlusion", "openness"}:
        return int(max_scale + 16)
    return int(max_scale + 16)


def _norm_stats_unused_for_mode(algorithm: str, merged: dict) -> bool:
    """True when the algorithm/output mode does not consume global_stats."""
    if algorithm == "structure_tensor" and str(merged.get("st_output", "coherence")).lower() == "orientation":
        return True
    if algorithm == "scale_drift" and str(merged.get("drift_output", "magnitude")).lower() == "direction":
        return True
    if algorithm == "tv_decomposition" and str(merged.get("component", "texture")).lower() == "structure":
        return True
    return False


def _compute_norm_stats_tiled(
    src_cog: str,
    algorithm: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
) -> Optional[tuple]:
    """Robust full-resolution, full-extent normalization stats via stratified tiles.

    Reads several full-resolution windows tiled across the valid-data bounding box
    (located from a coarse overview, so off-center footprints are handled), runs
    the algorithm's raw block function on each with the main-pass parameters, pools
    the interior valid pixels, and returns the algorithm's robust ``(offset,
    p99-scale)`` statistics over the pool.  Full resolution keeps scale-sensitive
    magnitudes correct; pooling the whole extent dilutes singular outliers into the
    p99 tail.  Returns ``None`` on failure (caller falls back to window sampling).
    """
    spec = _NORM_STAT_SPECS.get(algorithm)
    if spec is None:
        return None
    try:
        import inspect
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        from .dask_registry import ALGORITHMS
        mod = __import__(f"FujiShaderGPU.algorithms.{spec[0]}", fromlist=[spec[1]])
        block_func = getattr(mod, spec[1])
        stat_func = getattr(mod, spec[2])
        try:
            defaults = ALGORITHMS[algorithm].get_default_params() or {}
        except Exception:
            defaults = {}
        merged = {**defaults, **(params or {})}
        # Tile backend: the topousm_fast large-radius split replaces
        # params["radii"] with the small radii BEFORE this prepass runs and
        # stashes the originals.  The stats must describe the full multiscale
        # output (the Dask backend injects before its split), otherwise the
        # tile display stretch is estimated from a distribution missing the
        # large-radius component and over-amplifies.
        if merged.get("_topousm_fast_full_radii"):
            merged["radii"] = list(merged["_topousm_fast_full_radii"])
            if merged.get("_topousm_fast_full_weights"):
                merged["weights"] = list(merged["_topousm_fast_full_weights"])
        if _norm_stats_unused_for_mode(algorithm, merged):
            logger.info("Skipping %s norm stats: selected output mode does not consume global_stats", algorithm)
            return None
    except Exception as exc:
        logger.warning("Tiled norm-stats helpers unavailable for %s: %s", algorithm, exc)
        return None

    try:
        accepted = set(inspect.signature(block_func).parameters)
        # Drop None-valued params so the block function falls back to its own
        # defaults (e.g. fractal_anomaly's radii default None -> would not iterate).
        kw = {k: merged[k] for k in list(merged) if k in accepted and merged[k] is not None}
        if "normalize" in accepted:
            kw["normalize"] = False
        halo = max(1, int(_norm_stat_halo_pixels(algorithm, merged)))
        # The valid interior after trimming must still contain pixels, so choose
        # a window that comfortably contains the footprint.  For huge radii this
        # may exceed 4096; that is intentional (audit N-3/M-20) and bounded by
        # raster dimensions later.
        margin = int(max(1, min(halo, max_tile)))
        tile = int(max(2048, 4 * margin))

        pooled = []
        with rasterio.open(src_cog) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            def _denodata(a):
                a = a.astype(np.float32, copy=False)
                if nodata is not None and not np.isnan(float(nodata)):
                    a = np.where(np.isclose(a, float(nodata), rtol=0.0, atol=1e-6), np.nan, a)
                return a

            # Coarse overview -> valid-data bounding box.
            cov = max(1, max(W, H) // 512)
            ov = _denodata(src.read(
                1, out_shape=(max(1, H // cov), max(1, W // cov)),
                resampling=Resampling.nearest, out_dtype=np.float32,
                masked=True).filled(np.nan))
            vmask = np.isfinite(ov)
            if not vmask.any():
                return None
            ys, xs = np.where(vmask)
            by0, by1 = int(ys.min()) * cov, min(H, (int(ys.max()) + 1) * cov)
            bx0, bx1 = int(xs.min()) * cov, min(W, (int(xs.max()) + 1) * cov)

            for wy0, wx0, tw, th in stratified_windows(
                    W, H, by0, by1, bx0, bx1, grid=grid, tile=min(tile, max(W, H))):
                a = _denodata(src.read(
                    1, window=Window(wx0, wy0, tw, th),
                    out_dtype=np.float32, masked=True).filled(np.nan))
                if float(np.isfinite(a).mean()) < min_valid_frac:
                    continue
                g = cp.asarray(a)
                raw = block_func(g, **kw)
                m = int(min(margin, raw.shape[0] // 3, raw.shape[1] // 3))
                if m > 0:
                    raw = raw[m:-m, m:-m]
                vals = raw[~cp.isnan(raw)]
                if vals.size:
                    pooled.append(cp.asnumpy(vals))
                del g, raw, vals
                cp.get_default_memory_pool().free_all_blocks()

        if not pooled:
            return None
        pooled_gpu = cp.asarray(np.concatenate(pooled))
        stats = stat_func(pooled_gpu)
        if not stats or not np.isfinite(float(stats[-1])) or float(stats[-1]) <= 1e-9:
            return None
        logger.info(
            "%s global stats from %d full-res tiles (tile=%d, margin=%d, %d px): %s",
            algorithm, len(pooled), tile, margin, int(pooled_gpu.size),
            tuple(round(float(s), 6) for s in stats),
        )
        return stats
    except Exception as exc:
        logger.warning("Failed tiled norm stats for %s: %s", algorithm, exc)
        return None


def inject_global_stats(src_cog: str, algorithm: str, params: dict, *, is_zarr: bool = False) -> dict:
    """Compute and inject every per-algorithm GLOBAL normalization statistic into
    ``params`` (in place), in the correct order and at full resolution.

    Single source of truth shared by the Dask and tile backends so their global
    statistics cannot drift.  All steps are mode-independent (they run for both
    ``local`` and ``spatial``) and seam-free (global, not per-tile).  No-op for
    Zarr inputs.  Order matters:

    1. fractal_anomaly relief (p10/p75) BEFORE the norm-stats pre-pass, so the
       pre-pass feature distribution matches the main pass (correct median
       centering; otherwise the result is biased bright on high-relief DEMs).
    2. generic robust display range for the normalized algorithms (TopoUSM Fast /
       fractal_anomaly / scale_space_surprise / visual_saliency /
       multiscale_terrain / ambient_occlusion / openness) from stratified
       full-resolution tiles.
    3. npr_edges global per-radius gradient threshold.
    4. specular global roughness p95.
    """
    if is_zarr:
        return params

    if (
        algorithm == "fractal_anomaly"
        and params.get("relief_p10") is None
        and params.get("relief_p75") is None
    ):
        from ._impl_fractal_anomaly import _compute_fractal_relief_stats
        _relief = _compute_fractal_relief_stats(src_cog, params)
        if _relief is not None:
            params["relief_p10"], params["relief_p75"] = _relief

    if algorithm in _NORM_STAT_SPECS and "global_stats" not in params:
        _ns = _compute_norm_stats_tiled(src_cog, algorithm, params)
        if _ns is not None:
            params["global_stats"] = _ns

    if algorithm == "npr_edges" and "_npr_grad_stats" not in params:
        from ._impl_npr_edges import _compute_npr_grad_stats
        _ngs = _compute_npr_grad_stats(src_cog, params)
        if _ngs:
            params["_npr_grad_stats"] = _ngs

    if algorithm == "specular" and params.get("roughness_norm_scale") is None:
        from ._impl_specular import _compute_specular_roughness_scale
        _rns = _compute_specular_roughness_scale(src_cog, params)
        if _rns is not None:
            params["roughness_norm_scale"] = _rns

    return params


__all__ = [
    "_NORM_STAT_SPECS", "_norm_stat_max_scale", "_compute_norm_stats_tiled",
    "stratified_windows", "inject_global_stats",
]
