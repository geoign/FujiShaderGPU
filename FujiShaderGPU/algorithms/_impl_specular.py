"""
FujiShaderGPU/algorithms/_impl_specular.py

Specular algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter

from ._base import Constants, DaskAlgorithm

logger = logging.getLogger(__name__)
from ._nan_utils import (
    handle_nan_for_gradient, restore_nan,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask, _smooth_for_radius,
    large_radius_threshold, coarsen_factor_for_shape, coarse_large_radius_response,
)


def compute_specular_block(block, *, roughness_scale=50.0, shininess=20.0,
                          pixel_size=1.0, pixel_scale_x=None, pixel_scale_y=None,
                          roughness_norm_scale=None, geographic_mode=False,
                          light_azimuth=Constants.DEFAULT_AZIMUTH,
                          light_altitude=Constants.DEFAULT_ALTITUDE):
    """Specular reflection computation (simplified Cook-Torrance model)."""
    nan_mask = cp.isnan(block)
    dy, dx, nan_mask = handle_nan_for_gradient(
        block, scale=1, pixel_size=pixel_size,
        pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
    )
    sign_x = 1.0 if (pixel_scale_x is None or float(pixel_scale_x) >= 0.0) else -1.0
    sign_y = 1.0 if (pixel_scale_y is None or float(pixel_scale_y) >= 0.0) else -1.0
    dz_d_east = dx * sign_x
    dz_d_north = dy * sign_y
    # Unit surface normal, kept as separate components instead of an (H, W, 3)
    # stack: the stacked-and-normalised array (plus its division temporary)
    # tripled peak VRAM on large chunks -- enough to exhaust the RMM pool on a
    # 12k chunk.  The component form is algebraically identical.
    inv_norm = cp.float32(1.0) / cp.sqrt(
        dz_d_east * dz_d_east + dz_d_north * dz_d_north + cp.float32(1.0)
    )
    n_x = (-dz_d_east) * inv_norm
    n_y = (-dz_d_north) * inv_norm
    n_z = inv_norm
    del dz_d_east, dz_d_north, inv_norm, dx, dy
    kernel_size = max(3, int(roughness_scale))
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        mean_values = uniform_filter(filled * valid, size=kernel_size, mode='constant')
        mean_weights = uniform_filter(valid, size=kernel_size, mode='constant')
        mean_f = cp.where(mean_weights > 0, mean_values / mean_weights, 0)
        sq_values = uniform_filter((filled**2) * valid, size=kernel_size, mode='constant')
        mean_sq_f = cp.where(mean_weights > 0, sq_values / mean_weights, 0)
    else:
        mean_f = uniform_filter(block, size=kernel_size, mode='constant')
        mean_sq_f = uniform_filter(block**2, size=kernel_size, mode='constant')
    roughness = cp.sqrt(cp.maximum(mean_sq_f - mean_f**2, 0))
    roughness_valid = roughness[~nan_mask] if nan_mask.any() else roughness
    if len(roughness_valid) > 0:
        if roughness_norm_scale is not None and float(roughness_norm_scale) > 1e-9:
            denom = float(roughness_norm_scale)
        else:
            p95_local = float(cp.percentile(roughness_valid, 95))
            denom = p95_local if p95_local > 1e-9 else float(cp.max(roughness_valid))
        if denom > 1e-9:
            roughness = roughness / (roughness + denom)
            roughness = cp.clip(roughness, 0.05, 1.0)
        else:
            roughness = cp.full_like(block, 0.5)
    else:
        roughness = cp.full_like(block, 0.5)
    eff_az = float((light_azimuth + 180.0) % 360.0) if geographic_mode else float(light_azimuth)
    light_az_rad = cp.radians(eff_az)
    light_alt_rad = cp.radians(light_altitude)
    light_dir = cp.array([
        cp.sin(light_az_rad) * cp.cos(light_alt_rad),
        cp.cos(light_az_rad) * cp.cos(light_alt_rad),
        cp.sin(light_alt_rad),
    ])
    view_dir = cp.array([0, 0, 1])
    half_vec = (light_dir + view_dir) / cp.linalg.norm(light_dir + view_dir)
    n_dot_h = n_x * half_vec[0] + n_y * half_vec[1] + n_z * half_vec[2]
    n_dot_h = cp.clip(n_dot_h, 0, 1)
    exponent = shininess * (1.0 - roughness * 0.8)
    specular = cp.power(n_dot_h, exponent)
    gloss_boost = 0.95 + 0.70 * (1.0 - roughness)
    specular = cp.clip(specular * gloss_boost, 0.0, 1.0)
    n_dot_v = cp.clip(n_z, 0.0, 1.0)
    f0 = cp.float32(0.06)
    fresnel = f0 + (1.0 - f0) * cp.power(1.0 - n_dot_v, 5.0)
    specular = cp.clip(specular * (0.80 + 0.45 * fresnel), 0.0, 1.0)
    specular = specular / (1.0 + 0.35 * specular)
    n_dot_l = n_x * light_dir[0] + n_y * light_dir[1] + n_z * light_dir[2]
    n_dot_l = cp.clip(n_dot_l, 0, 1)
    diffuse = n_dot_l * 0.28
    result = diffuse * 0.36 + specular * 0.64
    result = cp.clip(result, 0, 1)
    result = cp.power(result, 0.88)
    micro = specular - gaussian_filter(specular, sigma=1.1, mode='nearest')
    result = result + 0.10 * micro * (1.0 - 0.6 * roughness)
    result = cp.clip(result, 0, 1)
    result = 0.5 + 0.5 * cp.tanh((result - 0.5) / 0.82)
    result = 0.04 + 0.92 * result
    if geographic_mode:
        result = 1.0 - result
    result = cp.clip(result, 0, 1)
    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


def compute_specular_spatial_block(
    block, *, roughness_scale=50.0, shininess=20.0, pixel_size=1.0,
    pixel_scale_x=None, pixel_scale_y=None, roughness_norm_scale=None,
    geographic_mode=False, light_azimuth=Constants.DEFAULT_AZIMUTH,
    light_altitude=Constants.DEFAULT_ALTITUDE, radius=4.0,
):
    smoothed = _smooth_for_radius(block, radius, pixel_size=pixel_size, algorithm_name="specular")
    return compute_specular_block(
        smoothed, roughness_scale=roughness_scale, shininess=shininess,
        pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y, roughness_norm_scale=roughness_norm_scale,
        geographic_mode=geographic_mode,
        light_azimuth=light_azimuth, light_altitude=light_altitude,
    )


class SpecularAlgorithm(DaskAlgorithm):
    """Specular reflection algorithm."""
    def process(self, gpu_arr, **params):
        rs = params.get('roughness_scale', 50.0)
        sh = params.get('shininess', 20.0)
        ps = params.get('pixel_size', 1.0)
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        rns = params.get('roughness_norm_scale', None)
        geo = bool(params.get('is_geographic_dem', False))
        laz = params.get('light_azimuth', Constants.DEFAULT_AZIMUTH)
        lal = params.get('light_altitude', Constants.DEFAULT_ALTITUDE)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), ps)
        agg = params.get("agg", "mean")
        if mode == "spatial":
            # NOTE: specular keeps its own per-radius loop (the other spatial
            # algorithms use multiscale_response_fields) because its coarse path
            # also rescales the roughness kernel into the coarse grid
            # (roughness_scale -> rs/F) and uses a roughness-dependent halo
            # (max(rs, 2r+1)); the generic helper passes identical block kwargs to
            # the coarse and full-res branches, so it cannot express that.
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            # Coarsen for large radii on geographic DEMs too (pixel-based; the
            # roughness kernel and metric scales below are scaled by F as well).
            F = coarsen_factor_for_shape(gpu_arr.shape)
            # Unified overview source (shared decimated read): when present the
            # coarse grid uses its decimation, so the roughness kernel must scale by
            # the SAME factor (not F) to stay metric-consistent on that grid.
            _ov_dem = params.get("_overview_coarse_dem")
            _ov_decim = params.get("_overview_decimation")
            _coarse_fac = float(_ov_decim) if (_ov_dem is not None and _ov_decim) else float(F)
            cache = {}
            responses = []
            _coarse_ok_tile = (_ov_dem is not None and params.get("_tile_origin") is not None)
            for radius in radii:
                depth = max(int(rs), int(float(radius) * 2 + 1))
                if int(round(float(radius))) > thr and (F > 1 or _coarse_ok_tile):
                    # Scale the roughness kernel into the coarse grid as well.
                    rs_coarse = max(3.0, float(rs) / _coarse_fac)
                    responses.append(coarse_large_radius_response(
                        gpu_arr, block_fn=compute_specular_spatial_block,
                        radius_kw="radius", radius=float(radius), factor=F,
                        depth_for_radius=lambda rc, _rs=rs_coarse: max(int(_rs), int(rc * 2 + 1)),
                        pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
                        coarse_cache=cache,
                        coarse_dem=_ov_dem, coarse_decimation=_ov_decim,
                        tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                        roughness_scale=rs_coarse, shininess=sh,
                        roughness_norm_scale=rns, geographic_mode=geo,
                        light_azimuth=laz, light_altitude=lal,
                    ))
                else:
                    responses.append(gpu_arr.map_overlap(
                        compute_specular_spatial_block, depth=depth,
                        boundary='reflect', dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        roughness_scale=rs, shininess=sh, pixel_size=ps,
                        pixel_scale_x=psx, pixel_scale_y=psy,
                        roughness_norm_scale=rns, geographic_mode=geo,
                        light_azimuth=laz, light_altitude=lal, radius=float(radius)))
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        return gpu_arr.map_overlap(
            compute_specular_block, depth=int(rs), boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            roughness_scale=rs, shininess=sh, pixel_size=ps,
            pixel_scale_x=psx, pixel_scale_y=psy,
            roughness_norm_scale=rns, geographic_mode=geo,
            light_azimuth=laz, light_altitude=lal)

    def get_default_params(self):
        return {
            'roughness_scale': 20.0, 'shininess': 10.0,
            'light_azimuth': Constants.DEFAULT_AZIMUTH,
            'light_altitude': Constants.DEFAULT_ALTITUDE,
            'pixel_size': 1.0, 'roughness_norm_scale': None,
            'mode': 'local', 'radii': None, 'weights': None,
        }


def _roughness_p95_block(block: cp.ndarray, kernel: int) -> cp.ndarray:
    """Local roughness (std of elevation over a ``kernel`` window), NaN-aware --
    identical to the roughness computed inside ``compute_specular_block`` so the
    pooled p95 matches the per-tile magnitude."""
    nan_mask = cp.isnan(block)
    if bool(nan_mask.any()):
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        mw = uniform_filter(valid, size=kernel, mode='constant')
        mean_f = cp.where(mw > 0, uniform_filter(filled * valid, size=kernel, mode='constant') / mw, 0)
        mean_sq = cp.where(mw > 0, uniform_filter((filled ** 2) * valid, size=kernel, mode='constant') / mw, 0)
    else:
        mean_f = uniform_filter(block, size=kernel, mode='constant')
        mean_sq = uniform_filter(block ** 2, size=kernel, mode='constant')
    return cp.sqrt(cp.maximum(mean_sq - mean_f ** 2, 0))


def _compute_specular_roughness_scale(
    src_cog: str,
    params: dict,
    *,
    elevation_scale: float = 1.0,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
) -> Optional[float]:
    """Global roughness p95 for specular, from full-resolution stratified tiles.

    ``compute_specular_block`` otherwise normalizes roughness by a *per-block* p95
    (``roughness / (roughness + p95)``), which differs tile-to-tile and produces
    tile-boundary seams -- pronounced on large / geographic DEMs where terrain
    roughness varies strongly across the extent.  A single global p95 used as
    ``roughness_norm_scale`` makes the normalization identical for every tile.
    Computed at full resolution (kernel = roughness_scale) so the magnitude
    matches the per-block roughness.  Backend-neutral (rasterio + cupy) so the
    Dask and tile pipelines can share one copy.  Returns ``None`` on failure
    (caller falls back to the per-block p95)."""
    try:
        import rasterio
        from rasterio.windows import Window
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("specular roughness-stats helpers unavailable: %s", exc)
        return None
    kernel = max(3, int(float((params or {}).get("roughness_scale", 50.0))))
    try:
        margin = int(min(kernel, max_tile // 4))
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
                    if elevation_scale != 1.0:
                        # Match the per-block roughness magnitude on backends that
                        # pre-scale elevation (tile path multiplies the DEM by a
                        # latitude-dependent elevation_scale before compute).
                        g = g * cp.float32(elevation_scale)
                    rough = _roughness_p95_block(g, kernel)
                    m = int(min(margin, rough.shape[0] // 3, rough.shape[1] // 3))
                    if m > 0:
                        rough = rough[m:-m, m:-m]
                    nm = cp.isnan(g[m:-m, m:-m]) if m > 0 else cp.isnan(g)
                    vals = rough[~nm]
                    vals = vals[vals > 0]
                    if vals.size:
                        pooled.append(cp.asnumpy(vals))
                    del g, rough, vals
                    cp.get_default_memory_pool().free_all_blocks()
        if not pooled:
            return None
        p95 = float(np.percentile(np.concatenate(pooled), 95))
        if not (np.isfinite(p95) and p95 > 1e-9):
            return None
        logger.info("specular global roughness scale: p95=%.6g (from %d tiles)", p95, len(pooled))
        return p95
    except Exception as exc:
        logger.warning("Failed specular roughness stats: %s", exc)
        return None


__all__ = [
    "compute_specular_block", "compute_specular_spatial_block", "SpecularAlgorithm",
    "_compute_specular_roughness_scale",
]
