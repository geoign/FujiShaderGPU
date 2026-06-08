"""Bridge tile backend algorithms to shared Dask algorithm implementations."""
from __future__ import annotations

import cupy as cp
import numpy as np


class _FallbackToDask(Exception):
    """Internal marker for cases that still need the shared Dask graph."""


def _merged_params(algo, params):
    merged = {}
    try:
        defaults = algo.get_default_params()
        if isinstance(defaults, dict):
            merged.update(defaults)
    except Exception:
        pass
    merged.update(params)
    return merged


def _combine_direct(responses, *, weights=None, agg="mean"):
    if not responses:
        raise ValueError("responses must not be empty")
    if len(responses) == 1:
        return responses[0]

    agg_norm = str(agg or "mean").lower()
    if agg_norm == "stack":
        # The tile writer expects either HxWxC or band-first arrays. Keep this
        # unusual shape on the legacy path until the writer contract is explicit.
        raise _FallbackToDask()
    if agg_norm == "max":
        out = responses[0]
        for item in responses[1:]:
            out = cp.maximum(out, item)
        return out.astype(cp.float32, copy=False)
    if agg_norm == "min":
        out = responses[0]
        for item in responses[1:]:
            out = cp.minimum(out, item)
        return out.astype(cp.float32, copy=False)
    if agg_norm == "sum":
        out = cp.zeros_like(responses[0], dtype=cp.float32)
        for item in responses:
            out = out + item
        return out.astype(cp.float32, copy=False)

    if isinstance(weights, (list, tuple)) and len(weights) == len(responses):
        w = np.asarray(weights, dtype=np.float32)
        if np.isfinite(w).all() and float(w.sum()) > 0:
            w = w / float(w.sum())
            out = responses[0] * cp.float32(w[0])
            for idx in range(1, len(responses)):
                out = out + responses[idx] * cp.float32(w[idx])
            return out.astype(cp.float32, copy=False)

    out = cp.zeros_like(responses[0], dtype=cp.float32)
    inv = cp.float32(1.0 / float(len(responses)))
    for item in responses:
        out = out + item * inv
    return out.astype(cp.float32, copy=False)


def _resolve_radii_weights(params, pixel_size):
    from .._nan_utils import _resolve_spatial_radii_weights

    return _resolve_spatial_radii_weights(
        params.get("radii"), params.get("weights", None), pixel_size,
    )


def _direct_hillshade(block, params):
    from .._base import Constants
    from .._impl_hillshade import compute_hillshade_block, compute_hillshade_spatial_block

    p = params
    azimuth = p.get("azimuth", Constants.DEFAULT_AZIMUTH)
    altitude = p.get("altitude", Constants.DEFAULT_ALTITUDE)
    z_factor = p.get("z_factor", 1.0)
    if z_factor is None:
        z_factor = 1.0
    pixel_size = p.get("pixel_size", 1.0)
    pixel_scale_x = p.get("pixel_scale_x", None)
    pixel_scale_y = p.get("pixel_scale_y", None)
    geographic_mode = bool(p.get("is_geographic_dem", False))
    mode = str(p.get("mode", "local")).lower()
    radii = p.get("radii", [1])
    weights = p.get("weights", None)
    agg = p.get("agg", "mean")
    multiscale = bool(p.get("multiscale", False))

    if mode == "spatial":
        radii, weights = _resolve_radii_weights(p, pixel_size)
        multiscale = True
    else:
        if not isinstance(radii, (list, tuple)) or len(radii) == 0:
            radii = [1]
        radii = [max(1.0, float(r)) for r in radii]
        multiscale = bool(multiscale or len(radii) > 1)

    if mode == "spatial" or (multiscale and len(radii) > 1):
        responses = [
            compute_hillshade_spatial_block(
                block, azimuth=azimuth, altitude=altitude, z_factor=z_factor,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, geographic_mode=geographic_mode,
                radius=float(radius),
            )
            for radius in radii
        ]
        return _combine_direct(responses, weights=weights, agg=agg)

    return compute_hillshade_block(
        block, azimuth=azimuth, altitude=altitude, z_factor=z_factor,
        pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
        pixel_scale_y=pixel_scale_y, geographic_mode=geographic_mode,
    )


def _direct_slope(block, params):
    from .._impl_slope import compute_slope_block, compute_slope_spatial_block

    unit = params.get("unit", "degree")
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = [
            compute_slope_spatial_block(
                block, unit=unit, pixel_size=pixel_size,
                pixel_scale_x=psx, pixel_scale_y=psy, radius=float(radius),
            )
            for radius in radii
        ]
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_slope_block(
        block, unit=unit, pixel_size=pixel_size,
        pixel_scale_x=psx, pixel_scale_y=psy,
    )


def _direct_atmospheric_scattering(block, params):
    from .._impl_atmospheric_scattering import (
        compute_atmospheric_scattering_block,
        compute_atmospheric_scattering_spatial_block,
    )

    ss = params.get("scattering_strength", 0.5)
    intensity = params.get("intensity", None)
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = [
            compute_atmospheric_scattering_spatial_block(
                block, scattering_strength=ss, intensity=intensity,
                pixel_size=pixel_size, pixel_scale_x=psx,
                pixel_scale_y=psy, radius=float(radius),
            )
            for radius in radii
        ]
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_atmospheric_scattering_block(
        block, scattering_strength=ss, intensity=intensity,
        pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
    )


def _direct_curvature(block, params):
    from .._impl_curvature import compute_curvature_block
    from .._nan_utils import _smooth_for_radius

    curvature_type = params.get("curvature_type", "mean")
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = []
        for radius in radii:
            smoothed = _smooth_for_radius(
                block, float(radius), pixel_size=pixel_size,
                algorithm_name="curvature",
            )
            responses.append(
                compute_curvature_block(
                    smoothed, curvature_type=curvature_type,
                    pixel_size=pixel_size, pixel_scale_x=psx,
                    pixel_scale_y=psy,
                )
            )
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_curvature_block(
        block, curvature_type=curvature_type, pixel_size=pixel_size,
        pixel_scale_x=psx, pixel_scale_y=psy,
    )


def _direct_specular(block, params):
    from .._base import Constants
    from .._impl_specular import compute_specular_block, compute_specular_spatial_block

    rs = params.get("roughness_scale", 50.0)
    sh = params.get("shininess", 20.0)
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    rns = params.get("roughness_norm_scale", None)
    geo = bool(params.get("is_geographic_dem", False))
    laz = params.get("light_azimuth", Constants.DEFAULT_AZIMUTH)
    lal = params.get("light_altitude", Constants.DEFAULT_ALTITUDE)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = [
            compute_specular_spatial_block(
                block, roughness_scale=rs, shininess=sh, pixel_size=pixel_size,
                pixel_scale_x=psx, pixel_scale_y=psy,
                roughness_norm_scale=rns, geographic_mode=geo,
                light_azimuth=laz, light_altitude=lal, radius=float(radius),
            )
            for radius in radii
        ]
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_specular_block(
        block, roughness_scale=rs, shininess=sh, pixel_size=pixel_size,
        pixel_scale_x=psx, pixel_scale_y=psy, roughness_norm_scale=rns,
        geographic_mode=geo, light_azimuth=laz, light_altitude=lal,
    )


def _direct_ambient_occlusion(block, params):
    from .._impl_ambient_occlusion import (
        compute_ambient_occlusion_block,
        compute_ambient_occlusion_spatial_block,
    )

    ns = params.get("num_samples", 16)
    radius = params.get("radius", 10.0)
    intensity = params.get("intensity", 1.0)
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = [
            compute_ambient_occlusion_spatial_block(
                block, num_samples=ns,
                radius=float(max(1, int(round(float(r))))),
                intensity=intensity, pixel_size=pixel_size,
                pixel_scale_x=psx, pixel_scale_y=psy,
            )
            for r in radii
        ]
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_ambient_occlusion_block(
        block, num_samples=ns, radius=radius, intensity=intensity,
        pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
    )


def _direct_openness(block, params):
    from .._impl_openness import compute_openness_spatial_block, compute_openness_vectorized

    openness_type = params.get("openness_type", "positive")
    nd = params.get("num_directions", 16)
    max_distance = params.get("max_distance", 50)
    pixel_size = params.get("pixel_size", 1.0)
    psx = params.get("pixel_scale_x", None)
    psy = params.get("pixel_scale_y", None)
    if str(params.get("mode", "local")).lower() == "spatial":
        radii, weights = _resolve_radii_weights(params, pixel_size)
        responses = [
            compute_openness_spatial_block(
                block, openness_type=openness_type, num_directions=nd,
                max_distance=int(max(2, round(float(r)))),
                pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
            )
            for r in radii
        ]
        return _combine_direct(responses, weights=weights, agg=params.get("agg", "mean"))
    return compute_openness_vectorized(
        block, openness_type=openness_type, num_directions=nd,
        max_distance=max_distance, pixel_size=pixel_size,
        pixel_scale_x=psx, pixel_scale_y=psy,
    )


def _direct_topousm_fast(block, params, algo):
    from .._global_stats import apply_global_normalization
    from .._impl_topousm_fast import compute_topousm_fast_efficient_block
    from .._nan_utils import _bilinear_sample_coarse
    from .._normalization import topousm_fast_norm_func, topousm_fast_stat_func

    pixel_size = params.get("pixel_size", 1.0)
    radii = params.get("radii", None)
    weights = params.get("weights", None)
    if radii is None:
        radii = algo._determine_optimal_radii(pixel_size)

    coarse_field = params.get("_topousm_fast_coarse_field", None)
    if coarse_field is not None:
        small_r = params.get("_topousm_fast_small_radii", [])
        small_w = params.get("_topousm_fast_small_weights", None)
        w_large = float(params.get("_topousm_fast_w_large", 0.0))
        off_r, off_c = params.get("_topousm_fast_field_offset", (0, 0))
        full_h, full_w = params.get("_topousm_fast_full_shape", block.shape)
        large_part = cp.float32(w_large) * block - _bilinear_sample_coarse(
            coarse_field,
            int(off_r), int(off_r) + int(block.shape[0]),
            int(off_c), int(off_c) + int(block.shape[1]),
            int(full_h), int(full_w),
        )
        if small_r:
            topousm_fast = compute_topousm_fast_efficient_block(
                block, radii=small_r, weights=small_w, pixel_size=pixel_size,
            ) + large_part
        else:
            topousm_fast = large_part.astype(cp.float32, copy=False)
    else:
        topousm_fast = compute_topousm_fast_efficient_block(
            block, radii=radii, weights=weights, pixel_size=pixel_size,
        )

    stats = params.get("global_stats", None)
    if not (
        isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9
    ):
        stats = topousm_fast_stat_func(topousm_fast)
    return apply_global_normalization(topousm_fast, topousm_fast_norm_func, stats)


def _direct_npr_edges(block, params):
    # Spatial (multi-radius) mode has a coarse-large-radius path; defer to the
    # Dask implementation so the tile backend matches it exactly.
    if str(params.get("mode", "local")).lower() == "spatial":
        raise _FallbackToDask()
    from .._impl_npr_edges import compute_npr_edges_block

    return compute_npr_edges_block(
        block,
        edge_sigma=params.get("edge_sigma", 1.0),
        threshold_low=params.get("threshold_low", 0.1),
        threshold_high=params.get("threshold_high", 0.3),
        pixel_size=params.get("pixel_size", 1.0),
    )


def _direct_multiscale_terrain(block, params):
    # The previous direct reimplementation ignored the unified --radii (read only
    # "scales") and upper-clipped the normalized tail at OVERFLOW_LIMIT, whereas
    # the Dask path resolves scales from radii, leaves the tail unclipped
    # (cp.maximum(...,0)), and has a coarse-large-radius path.  Defer to the Dask
    # implementation so the tile backend matches it exactly.
    raise _FallbackToDask()


def _direct_fractal_anomaly(block, params, algo):
    from .._impl_fractal_anomaly import compute_fractal_dimension_block, fractal_stat_func

    pixel_size = params.get("pixel_size", 1.0)
    radii = params.get("radii", None)
    if radii is None:
        radii = algo._determine_optimal_radii(pixel_size)
    if len(radii) < 5:
        radii = [4, 8, 16, 32, 64]

    sm_sig = float(params.get("smoothing_sigma", 1.2))
    ds_thr = float(params.get("despeckle_threshold", 0.35))
    ds_am = float(params.get("despeckle_alpha_max", 0.30))
    db = float(params.get("detail_boost", 0.35))
    weights = params.get("weights", None)
    stats = params.get("global_stats", None)
    if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
        raw = compute_fractal_dimension_block(
            block, radii=radii, normalize=False, smoothing_sigma=sm_sig,
            despeckle_threshold=ds_thr, despeckle_alpha_max=ds_am,
            detail_boost=db, weights=weights,
        )
        stats = fractal_stat_func(raw)
    rp10 = params.get("relief_p10", None)
    rp75 = params.get("relief_p75", None)
    if rp10 is None and rp75 is None and isinstance(stats, (tuple, list)) and len(stats) >= 4:
        rp10, rp75 = float(stats[2]), float(stats[3])
    return compute_fractal_dimension_block(
        block, radii=radii, normalize=True,
        mean_global=float(stats[0]), std_global=float(stats[1]),
        relief_p10=rp10, relief_p75=rp75, smoothing_sigma=sm_sig,
        despeckle_threshold=ds_thr, despeckle_alpha_max=ds_am,
        detail_boost=db, weights=weights,
    )


def _process_direct(algo, class_name, dem_gpu, params):
    p = _merged_params(algo, params)
    # Spatial-mode multi-radius runs take the shared Dask overview path (large
    # radii sampled from one global overview, tile-origin aware) so the tile
    # backend is seam-free and matches the Linux backend.  TopoUSM Fast and fractal keep
    # their own overview-based direct paths; `local` mode keeps the fast direct
    # paths below (single radius -> no large halo, no seams).
    if str(p.get("mode", "local")).lower() == "spatial" and class_name in {
        "HillshadeAlgorithm", "SlopeAlgorithm", "SpecularAlgorithm",
        "AtmosphericScatteringAlgorithm", "CurvatureAlgorithm",
        "AmbientOcclusionAlgorithm", "OpennessAlgorithm",
        "FractalAnomalyAlgorithm",
    }:
        raise _FallbackToDask()
    if class_name == "HillshadeAlgorithm":
        return _direct_hillshade(dem_gpu, p)
    if class_name == "SlopeAlgorithm":
        return _direct_slope(dem_gpu, p)
    if class_name == "SpecularAlgorithm":
        return _direct_specular(dem_gpu, p)
    if class_name == "AtmosphericScatteringAlgorithm":
        return _direct_atmospheric_scattering(dem_gpu, p)
    if class_name == "CurvatureAlgorithm":
        return _direct_curvature(dem_gpu, p)
    if class_name == "AmbientOcclusionAlgorithm":
        return _direct_ambient_occlusion(dem_gpu, p)
    if class_name == "OpennessAlgorithm":
        return _direct_openness(dem_gpu, p)
    if class_name == "TopoUSMFastAlgorithm":
        return _direct_topousm_fast(dem_gpu, p, algo)
    if class_name == "NPREdgesAlgorithm":
        return _direct_npr_edges(dem_gpu, p)
    # VisualSaliencyAlgorithm intentionally has no direct path: it is always
    # multiscale, so it uses the Dask hybrid overview path (tile-origin aware).
    if class_name == "MultiscaleDaskAlgorithm":
        return _direct_multiscale_terrain(dem_gpu, p)
    if class_name == "FractalAnomalyAlgorithm":
        return _direct_fractal_anomaly(dem_gpu, p, algo)
    raise _FallbackToDask()


class DaskSharedTileAdapter:
    """Run a DaskAlgorithm implementation on a single tile array."""

    dask_algorithm_cls = None

    def __init__(self):
        if self.dask_algorithm_cls is None:
            raise ValueError("dask_algorithm_cls must be set in subclass")
        self._algo = self.dask_algorithm_cls()

    def get_default_params(self):
        return self._algo.get_default_params()

    def process(self, dem_gpu: cp.ndarray, **params):
        try:
            return _process_direct(
                self._algo,
                self.dask_algorithm_cls.__name__,
                dem_gpu,
                params,
            )
        except _FallbackToDask:
            pass

        try:
            import dask.array as da
        except Exception as exc:
            raise RuntimeError(
                "Dask is required for this algorithm on tile backend. "
                "Install with: pip install dask[array]"
            ) from exc

        gpu_da = da.from_array(dem_gpu, chunks=dem_gpu.shape, asarray=False)
        result_da = self._algo.process(gpu_da, **params)
        result = result_da.compute()
        if isinstance(result, cp.ndarray):
            return result
        return cp.asarray(result)
