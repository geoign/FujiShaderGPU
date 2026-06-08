"""
FujiShaderGPU/algorithms/_impl_fractal_anomaly.py

Fractal Anomaly algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import logging
from typing import Optional
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter, median_filter

from ._base import DaskAlgorithm, classify_resolution, Constants
from ._nan_utils import (
    handle_nan_with_gaussian, restore_nan, resolve_block_weights,
    hybrid_multiscale_response_combine,
)
from ._global_stats import compute_global_stats
from ._normalization import NORMAL_PERCENTILE

logger = logging.getLogger(__name__)


def fractal_large_scale_predicate(radius) -> bool:
    """A fractal radius is "large" when its roughness halo (~2r + feature margin)
    would exceed MAX_DEPTH, i.e. the single-block path would truncate it.  Such
    radii are taken from the overview via the hybrid coarse path instead."""
    return int(2 * float(radius) + 16) > Constants.MAX_DEPTH


def compute_roughness_multiscale(block, radii, window_mult=3, detrend=True):
    """Compute per-scale roughness maps for fractal-style analysis."""
    nan_mask = cp.isnan(block)
    sigmas = []
    for r in radii:
        sigma = max(0.8, float(r * window_mult) / 6.0)
        if detrend:
            if nan_mask.any():
                trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
                residual = block - trend
                local_energy, _ = handle_nan_with_gaussian(
                    residual ** 2, sigma=sigma, mode='nearest')
            else:
                trend = gaussian_filter(block, sigma=sigma, mode='nearest')
                residual = block - trend
                local_energy = gaussian_filter(residual ** 2, sigma=sigma, mode='nearest')
            rough = cp.sqrt(cp.maximum(local_energy, 1e-8))
        else:
            if nan_mask.any():
                local_mean, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
                local_mean_sq, _ = handle_nan_with_gaussian(
                    block ** 2, sigma=sigma, mode='nearest')
            else:
                local_mean = gaussian_filter(block, sigma=sigma, mode='nearest')
                local_mean_sq = gaussian_filter(block ** 2, sigma=sigma, mode='nearest')
            variance = local_mean_sq - local_mean ** 2
            rough = cp.sqrt(cp.maximum(variance, 0.0))
        sigmas.append(rough)
    return cp.stack(sigmas, axis=-1)


def _fractal_roughness_block(block, *, scale, pixel_size=1.0, pixel_scale_x=None,
                             pixel_scale_y=None, **_ignored):
    """One scale's detrended roughness (matches compute_roughness_multiscale,
    ``window_mult=3, detrend=True``).  Used as the per-scale primitive so large
    radii can be computed on a coarsened overview via the shared coarse path."""
    sigma = max(0.8, float(scale) * 3.0 / 6.0)
    nan_mask = cp.isnan(block)
    if bool(nan_mask.any()):
        trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
        residual = block - trend
        energy, _ = handle_nan_with_gaussian(residual ** 2, sigma=sigma, mode='nearest')
    else:
        trend = gaussian_filter(block, sigma=sigma, mode='nearest')
        residual = block - trend
        energy = gaussian_filter(residual ** 2, sigma=sigma, mode='nearest')
    return cp.sqrt(cp.maximum(energy, 1e-8)).astype(cp.float32)


def compute_fractal_dimension_block(block, *, radii=[4, 8, 16, 32, 64],
                                  normalize=True, mean_global=None, std_global=None,
                                  relief_p10=None, relief_p75=None,
                                  smoothing_sigma=1.2, despeckle_threshold=0.35,
                                  despeckle_alpha_max=0.30, detail_boost=0.35,
                                  weights=None):
    """Compute fractal anomaly from detrended multiscale roughness (full-resolution
    block; used by the global-stats pre-pass).  The main pass computes the same
    feature from per-scale roughness fields (coarse path for large radii) via
    ``_fractal_feature_from_roughness``.

    The unified ``--weights`` (when length-matching ``radii``) replaces the
    default ``sqrt(scale)`` weighting of the log-log roughness regression, so a
    user can emphasize particular scales in the fractal-slope fit.  Absent or
    mismatched weights keep the original ``sqrt(scale)`` behavior.
    """
    sigmas = compute_roughness_multiscale(block, radii, window_mult=3, detrend=True)
    return _fractal_feature_from_roughness(
        block, sigmas, radii=radii, weights=weights, normalize=normalize,
        mean_global=mean_global, std_global=std_global, relief_p10=relief_p10,
        relief_p75=relief_p75, smoothing_sigma=smoothing_sigma,
        despeckle_threshold=despeckle_threshold, despeckle_alpha_max=despeckle_alpha_max,
        detail_boost=detail_boost)


def _fractal_feature_from_roughness(block, sigmas, *, radii, weights=None, normalize=True,
                                    mean_global=None, std_global=None, relief_p10=None,
                                    relief_p75=None, smoothing_sigma=1.2,
                                    despeckle_threshold=0.35, despeckle_alpha_max=0.30,
                                    detail_boost=0.35):
    """Regression -> feature -> despeckle -> normalize half of fractal_anomaly,
    given a precomputed per-scale roughness stack ``sigmas`` (H, W, N)."""
    nan_mask = cp.isnan(block)
    n_sc = int(sigmas.shape[2])
    # Per-scale log-roughness accumulated in place.  The vectorized form built
    # several H x W x N intermediates (y_fit, (log_sigmas - y_fit)**2, ...) which
    # exhaust the RMM pool on large chunks; this computes the identical weighted
    # log-log regression with only H x W accumulators (log_sigma_i recomputed per
    # pass -- cheap on GPU).
    def _lsig(i):
        return cp.log(cp.maximum(sigmas[:, :, i], 1e-5))
    _ls = [float(np.log(max(float(r), 1e-9))) for r in list(radii)[:n_sc]]
    _sw_gpu = resolve_block_weights(weights, n_sc)
    if _sw_gpu is None:
        _swa = np.sqrt(np.asarray([float(r) for r in list(radii)[:n_sc]], dtype=np.float64))
        _sw = (_swa / _swa.sum()).tolist()
    else:
        _sw = [float(x) for x in cp.asnumpy(_sw_gpu)]
    mean_log_scale = float(sum(_ls[i] * _sw[i] for i in range(n_sc)))
    var_log_scale = float(sum(((_ls[i] - mean_log_scale) ** 2) * _sw[i] for i in range(n_sc)))
    mean_log_sigma = cp.zeros(sigmas.shape[:2], dtype=cp.float32)
    for i in range(n_sc):
        mean_log_sigma = mean_log_sigma + _lsig(i) * cp.float32(_sw[i])
    cov = cp.zeros(sigmas.shape[:2], dtype=cp.float32)
    ss_tot = cp.zeros(sigmas.shape[:2], dtype=cp.float32)
    for i in range(n_sc):
        _d = _lsig(i) - mean_log_sigma
        cov = cov + cp.float32(_ls[i] - mean_log_scale) * _d * cp.float32(_sw[i])
        ss_tot = ss_tot + (_d * _d) * cp.float32(_sw[i])
    beta = cov / (var_log_scale + 1e-10)
    ss_res = cp.zeros(sigmas.shape[:2], dtype=cp.float32)
    for i in range(n_sc):
        _yfit = mean_log_sigma + beta * cp.float32(_ls[i] - mean_log_scale)
        _diff = _lsig(i) - _yfit
        ss_res = ss_res + (_diff * _diff) * cp.float32(_sw[i])
    r2 = cp.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0)
    rmse = cp.sqrt(cp.maximum(ss_res, 0.0))
    beta_dev = cp.clip(beta - 1.2, -4.0, 4.0)
    rmse_comp = cp.log1p(cp.maximum(rmse, 0.0))
    roughness = cp.mean(sigmas, axis=2)
    valid_rough = roughness[~nan_mask]
    if (relief_p10 is not None and relief_p75 is not None
            and np.isfinite(relief_p10) and np.isfinite(relief_p75)
            and float(relief_p75) > float(relief_p10)):
        r_p10, r_p75 = float(relief_p10), float(relief_p75)
    elif valid_rough.size > 0:
        r_p10 = float(cp.percentile(valid_rough, 10))
        r_p75 = float(cp.percentile(valid_rough, 75))
    else:
        r_p10, r_p75 = 0.0, 1.0
    relief_conf = cp.clip((roughness - r_p10) / max(r_p75 - r_p10, 1e-6), 0.0, 1.0)
    raw_feature = 0.75 * beta_dev + 0.45 * rmse_comp
    fine_i = 0
    coarse_i = min(2, n_sc - 1)
    fine_ratio = _lsig(fine_i) - _lsig(coarse_i)
    max_i = n_sc - 1
    macro_i = max(max_i - 2, 0)
    macro_ratio = _lsig(max_i) - _lsig(macro_i)
    raw_feature = raw_feature + 0.35 * macro_ratio * relief_conf
    raw_feature = raw_feature + float(detail_boost) * 0.18 * fine_ratio * relief_conf
    smooth = max(0.0, float(smoothing_sigma))
    feat_smooth = raw_feature
    if smooth > 0:
        if nan_mask.any():
            feat_smooth, _ = handle_nan_with_gaussian(raw_feature, sigma=smooth, mode='nearest')
        else:
            feat_smooth = gaussian_filter(raw_feature, sigma=smooth, mode='nearest')
    alpha_r2 = cp.clip((r2 - 0.35) / 0.6, 0.0, 1.0)
    alpha = 0.50 + 0.50 * (alpha_r2 * relief_conf)
    feature_out = alpha * raw_feature + (1.0 - alpha) * feat_smooth
    if normalize and mean_global is not None and std_global is not None:
        if std_global > 1e-6:
            # center via robust median, p99(|centered|) -> +/-1; unclipped tail.
            result = (feature_out - mean_global) / std_global
        else:
            result = cp.zeros_like(feature_out)
    else:
        result = feature_out
    thr = max(0.05, float(despeckle_threshold))
    alpha_max = float(despeckle_alpha_max)
    med = median_filter(result, size=3, mode='nearest')
    thr_map = thr * (0.7 + 1.1 * alpha)
    despeckle_mask = (
        (cp.abs(result - med) > thr_map) & (alpha < alpha_max) & (relief_conf < 0.45))
    result = cp.where(despeckle_mask, med, result)
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


def _fractal_combine_block(block, *roughness, radii, weights=None, normalize=True,
                           mean_global=None, std_global=None, relief_p10=None,
                           relief_p75=None, smoothing_sigma=1.2, despeckle_threshold=0.35,
                           despeckle_alpha_max=0.30, detail_boost=0.35):
    """Stack the per-scale roughness fields (large radii from the coarse path) and
    run the fractal feature/regression.  NaN that the coarse path re-masked into a
    roughness field is floored to 1e-8 (those pixels are the NoData footprint and
    are re-masked at the end)."""
    sig = cp.stack(
        [cp.where(cp.isnan(r), cp.float32(1e-8), r).astype(cp.float32) for r in roughness],
        axis=-1)
    return _fractal_feature_from_roughness(
        block, sig, radii=radii, weights=weights, normalize=normalize,
        mean_global=mean_global, std_global=std_global, relief_p10=relief_p10,
        relief_p75=relief_p75, smoothing_sigma=smoothing_sigma,
        despeckle_threshold=despeckle_threshold, despeckle_alpha_max=despeckle_alpha_max,
        detail_boost=detail_boost)


def fractal_stat_func(data):
    """Compute robust center/scale: p80(abs(centered)) maps to +/-1."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        center = float(cp.median(valid_data))
        abs_dev = cp.abs(valid_data - center)
        scale = float(cp.percentile(abs_dev, NORMAL_PERCENTILE))
        return (center, max(scale, 1e-6))
    return (0.0, 0.5)


class FractalAnomalyAlgorithm(DaskAlgorithm):
    """Fractal anomaly detection -- a FujiShaderGPU-original composite indicator.

    There is no single published "fractal anomaly" algorithm.  The *core* is the
    established fractal-surface method: detrended local roughness sigma(L) is
    measured at several scales and a log-log regression of log(sigma) vs log(scale)
    gives the slope beta, an estimate of the roughness-scaling (Hurst) exponent
    (self-affine terrain follows sigma(L) ~ L^H; surface fractal dimension
    D = 3 - H).  The *anomaly* feature built on top -- deviation of beta from a
    reference exponent, the regression RMSE (departure from a clean power law),
    relief-weighted macro/fine scale ratios, despeckle and blending -- is a
    bespoke FujiShaderGPU construction (its coefficients are heuristic, tuned for
    terrain visualization), not a standard geomorphometric product."""
    def process(self, gpu_arr, **params):
        radii = params.get('radii', None)
        ps = params.get('pixel_size', 1.0)
        sm_sig = float(params.get('smoothing_sigma', 1.2))
        ds_thr = float(params.get('despeckle_threshold', 0.35))
        ds_am = float(params.get('despeckle_alpha_max', 0.30))
        db = float(params.get('detail_boost', 0.35))
        rp10 = params.get('relief_p10', None)
        rp75 = params.get('relief_p75', None)
        weights = params.get('weights', None)
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        if radii is None:
            radii = self._determine_optimal_radii(ps)
        if len(radii) < 5:
            radii = [4, 8, 16, 32, 64]
        max_r = max(radii)
        # Roughness uses a Gaussian with sigma = r/2 (window_mult=3, /6), whose
        # 4-sigma kernel needs ~2r of halo (+16 for the feature smoothing + median).
        # Bound the stats pre-pass at MAX_DEPTH; the main pass computes large radii
        # on a coarsened overview (no oversized halo) instead.
        stats_depth = min(max_r * 2 + 16, Constants.MAX_DEPTH)
        stats = params.get('global_stats', None)
        stats_ok = (isinstance(stats, (tuple, list)) and len(stats) >= 2
                     and float(stats[1]) > 1e-9)
        if not stats_ok:
            nb = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if nb > 1:
                stats = compute_global_stats(
                    gpu_arr, fractal_stat_func, compute_fractal_dimension_block,
                    {'radii': radii, 'normalize': False, 'smoothing_sigma': sm_sig,
                     'despeckle_threshold': ds_thr, 'despeckle_alpha_max': ds_am,
                     'detail_boost': db, 'weights': weights},
                    downsample_factor=params.get('downsample_factor', None),
                    depth=stats_depth, algorithm_name='fractal_anomaly')
            else:
                stats = (0.0, 0.5)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
            stats = (0.0, 0.5)
        if rp10 is None and rp75 is None:
            if isinstance(stats, (tuple, list)) and len(stats) >= 4:
                rp10, rp75 = float(stats[2]), float(stats[3])
        mean_D, std_D = float(stats[0]), float(stats[1])

        # Hybrid coarse path (TopoUSM Fast-style): when the orchestrator supplies the
        # large-radius roughness fields precomputed from the COG overview, compute
        # small radii at full resolution (bounded halo) and sample the large radii
        # from their concrete overview fields inside one depth-0 combine.  This is
        # both accurate for large --radii (true low-frequency roughness, not a
        # MAX_DEPTH-truncated halo) and bounded in VRAM on huge streaming rasters
        # (no per-large-scale Dask field whose GPU intermediates accumulate).
        large_fields = params.get("_fractal_large_fields")
        if large_fields:
            full_shape = params.get("_fractal_full_shape", tuple(int(s) for s in gpu_arr.shape))
            return hybrid_multiscale_response_combine(
                gpu_arr, [float(r) for r in radii],
                small_block_fn=_fractal_roughness_block,
                combine_fn=_fractal_combine_block,
                depth_for_scale=lambda rr: int(2 * float(rr) + 16),
                large_fields=large_fields, full_shape=full_shape,
                tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                radius_kw="scale", pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
                combine_kwargs=dict(
                    radii=radii, weights=weights, normalize=True,
                    mean_global=mean_D, std_global=std_D,
                    relief_p10=rp10, relief_p75=rp75, smoothing_sigma=sm_sig,
                    despeckle_threshold=ds_thr, despeckle_alpha_max=ds_am,
                    detail_boost=db))

        # Single self-contained block per output tile (bounds device memory).  The
        # coarse-overview decomposition into 6 per-scale roughness fields fed a deep
        # dask graph whose GPU intermediates accumulated to tens of GB on large
        # streaming rasters (ALOS/Kyoto).  The halo is capped at MAX_DEPTH (and
        # below the chunk), so very large radii are approximated (truncated halo)
        # instead of OOM; default/small radii are unaffected.
        min_chunk = min((min(ax) for ax in gpu_arr.chunks), default=1) if hasattr(gpu_arr, "chunks") else 1
        depth = max(1, min(2 * max_r + 16, Constants.MAX_DEPTH, int(min_chunk) - 1))
        return gpu_arr.map_overlap(
            compute_fractal_dimension_block, depth=depth,
            boundary='reflect', dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radii=radii, weights=weights, normalize=True,
            mean_global=mean_D, std_global=std_D,
            relief_p10=rp10, relief_p75=rp75, smoothing_sigma=sm_sig,
            despeckle_threshold=ds_thr, despeckle_alpha_max=ds_am, detail_boost=db)

    def _determine_optimal_radii(self, pixel_size):
        """Determine optimal radii based on resolution."""
        rc = classify_resolution(pixel_size)
        if rc == 'ultra_high':
            base = [4, 8, 16, 32, 64, 96]
        elif rc == 'very_high':
            base = [4, 8, 16, 24, 32, 48]
        elif rc == 'high':
            base = [4, 8, 16, 32, 48, 64]
        elif rc == 'medium':
            base = [3, 6, 12, 24, 36, 48]
        elif rc == 'low':
            base = [2, 4, 8, 16, 24, 32]
        else:
            base = [2, 4, 8, 12, 16, 24]
        if len(base) > 6:
            indices = cp.linspace(0, len(base)-1, 6).astype(int).get()
            base = [base[int(i)] for i in indices]
        return base

    def get_default_params(self):
        return {
            'radii': None, 'pixel_size': 1.0, 'downsample_factor': None,
            'smoothing_sigma': 1.2, 'despeckle_threshold': 0.35,
            'despeckle_alpha_max': 0.30, 'detail_boost': 0.35,
        }


def _compute_fractal_relief_stats(
    src_cog: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
) -> Optional[tuple]:
    """Global roughness (relief) p10/p75 for fractal_anomaly, from full-resolution
    stratified tiles.  ``relief_conf`` otherwise uses a per-block roughness
    percentile, which differs tile-to-tile and produces tile-boundary seams; a
    single global pair makes it consistent across the whole raster.  Computed at
    full resolution so the magnitude matches the per-block roughness.

    Backend-neutral (rasterio + cupy) so both the Dask and tile pipelines share
    one copy."""
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        from .dask_registry import ALGORITHMS
    except Exception as exc:
        logger.warning("fractal relief-stats helpers unavailable: %s", exc)
        return None
    try:
        defaults = ALGORITHMS["fractal_anomaly"].get_default_params() or {}
    except Exception:
        defaults = {}
    radii = (params or {}).get("radii") or defaults.get("radii") or [4, 8, 16, 32, 64]
    if len(radii) < 5:
        radii = [4, 8, 16, 32, 64]
    max_r = max(float(r) for r in radii)
    try:
        margin = int(min(2 * max_r + 16, max_tile // 4))
        tile = int(min(max_tile, max(2048, 4 * margin)))
        pooled = []
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
            ch_, cw_ = max(1, (by1 - by0) // grid), max(1, (bx1 - bx0) // grid)
            for gy in range(grid):
                for gx in range(grid):
                    ccy = by0 + gy * ch_ + ch_ // 2
                    ccx = bx0 + gx * cw_ + cw_ // 2
                    wy0 = int(min(max(0, ccy - tile // 2), max(0, H - tile)))
                    wx0 = int(min(max(0, ccx - tile // 2), max(0, W - tile)))
                    tw, th = min(tile, W - wx0), min(tile, H - wy0)
                    a = _dn(src.read(1, window=Window(wx0, wy0, tw, th),
                                     out_dtype=np.float32, masked=True).filled(np.nan))
                    if float(np.isfinite(a).mean()) < min_valid_frac:
                        continue
                    g = cp.asarray(a)
                    sig = compute_roughness_multiscale(g, radii, window_mult=3, detrend=True)
                    rough = cp.mean(sig, axis=2)
                    m = int(min(margin, rough.shape[0] // 3, rough.shape[1] // 3))
                    if m > 0:
                        rough = rough[m:-m, m:-m]
                    vals = rough[~cp.isnan(rough)]
                    if vals.size:
                        pooled.append(cp.asnumpy(vals))
                    del g, sig, rough, vals
                    cp.get_default_memory_pool().free_all_blocks()
        if not pooled:
            return None
        allv = np.concatenate(pooled)
        p10 = float(np.percentile(allv, 10))
        p75 = float(np.percentile(allv, 75))
        if not (np.isfinite(p10) and np.isfinite(p75) and p75 > p10):
            return None
        logger.info("fractal_anomaly global relief: p10=%.4g p75=%.4g (from %d tiles)",
                    p10, p75, len(pooled))
        return (p10, p75)
    except Exception as exc:
        logger.warning("Failed to compute fractal relief stats: %s", exc)
        return None


__all__ = [
    "compute_roughness_multiscale", "compute_fractal_dimension_block",
    "fractal_stat_func", "FractalAnomalyAlgorithm", "_compute_fractal_relief_stats",
]
