"""
FujiShaderGPU/core/tile_processor.py
タイルベース地形解析処理のコア実装（Windows/macOS向け）
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from ..core.gpu_memory import gpu_memory_pool
from ..config.system_config import get_gpu_config
from ..core.tile_io import read_tile_window, write_tile_output
from ..core.tile_compute import (
    run_tile_algorithm,
    apply_nodata_mask,
    _normalize_rvi_radii_and_weights,
)
from ..io.raster_info import detect_pixel_size_from_cog, metric_pixel_scales_from_metadata
from ..utils.types import TileResult
from ..utils.scale_analysis import analyze_terrain_scales, _get_default_scales
from ..utils.nodata_handler import _handle_nodata_ultra_fast
from ..io.cog_builder import _build_vrt_and_cog_ultra_fast
from ..io.cog_validator import _validate_cog_for_qgis
import os
import math
import glob
import shutil
import tempfile
from pathlib import Path
import rasterio
import numpy as np
import cupy as cp
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from typing import Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

# デフォルトで利用可能なアルゴリズム（Windows/macOS）
DEFAULT_ALGORITHMS = {
    # Canonical names aligned with Dask registry
    "rvi": "RVIAlgorithm",
    "hillshade": "HillshadeAlgorithm",
    "slope": "SlopeAlgorithm",
    "specular": "SpecularAlgorithm",
    "atmospheric_scattering": "AtmosphericScatteringAlgorithm",
    "multiscale_terrain": "MultiscaleDaskAlgorithm",
    "curvature": "CurvatureAlgorithm",
    "visual_saliency": "VisualSaliencyAlgorithm",
    "npr_edges": "NPREdgesAlgorithm",
    "ambient_occlusion": "AmbientOcclusionAlgorithm",
    "lrm": "LRMAlgorithm",
    "openness": "OpennessAlgorithm",
    "fractal_anomaly": "FractalAnomalyAlgorithm",
    "scale_space_surprise": "ScaleSpaceSurpriseAlgorithm",
    "multi_light_uncertainty": "MultiLightUncertaintyAlgorithm",
}

GLOBAL_STATS_NATIVE_ALGOS = {
    "rvi",
    "multiscale_terrain",
    "visual_saliency",
    "lrm",
    "fractal_anomaly",
}
NO_NORMALIZATION_ALGOS = {"hillshade", "slope", "npr_edges"}
SIGNED_NORMALIZATION_ALGOS = {"rvi", "lrm", "fractal_anomaly"}


def _is_global_stats_required(algorithm: str, mode: str) -> bool:
    mode_norm = str(mode or "local").lower()
    if mode_norm == "spatial":
        return algorithm not in NO_NORMALIZATION_ALGOS
    if mode_norm == "local":
        return algorithm not in NO_NORMALIZATION_ALGOS
    return False


def _normalization_target_range(algorithm: str) -> str:
    """Return output normalization policy: 'none' | 'unit' | 'signed'."""
    if algorithm in NO_NORMALIZATION_ALGOS:
        return "none"
    if algorithm == "scale_space_surprise":
        return "unit_surprise"
    if algorithm in SIGNED_NORMALIZATION_ALGOS:
        return "signed"
    return "unit"


def _required_padding_for_algorithm(
    algorithm: str,
    algo_params: dict,
    sigma: float,
    pixel_size: float,
    target_distances: Optional[List[float]],
) -> int:
    """Minimum halo (pixels) required to avoid tile seam artifacts per algorithm."""
    # Conservative default from sigma-driven filters.
    try:
        base = int(math.ceil(max(float(sigma), 0.0) * 5.0))
    except Exception:
        base = 32

    required = max(32, base)

    if algorithm == "visual_saliency":
        scales = algo_params.get("scales", [2, 4, 8, 16])
        try:
            max_scale = max(float(s) for s in scales)
        except Exception:
            max_scale = 16.0
        required = max(required, int(math.ceil(max_scale * 8.0)))
    elif algorithm == "rvi":
        radii, _ = _normalize_rvi_radii_and_weights(
            target_distances=target_distances,
            weights=algo_params.get("weights"),
            pixel_size=pixel_size,
            manual_radii=algo_params.get("radii"),
            manual_weights=algo_params.get("weights"),
        )
        if radii:
            required = max(required, int(max(radii) * 2 + 1))
    elif algorithm == "multiscale_terrain":
        scales = algo_params.get("scales", [1, 10, 50, 100])
        try:
            max_scale = max(float(s) for s in scales)
        except Exception:
            max_scale = 100.0
        required = max(required, int(min(max_scale * 4.0, 512)))
    elif algorithm == "scale_space_surprise":
        scales = algo_params.get("scales", [1.0, 2.0, 4.0, 8.0, 16.0])
        try:
            max_scale = max(float(s) for s in scales if float(s) > 0)
        except Exception:
            max_scale = 16.0
        # Align with shared algorithm depth: int(max(scales)*3)+1
        required = max(required, int(math.ceil(max_scale * 3.0)) + 1)
    elif algorithm == "fractal_anomaly":
        radii = algo_params.get("radii")
        if not radii:
            try:
                from ..algorithms.dask_shared import FractalAnomalyAlgorithm
                radii = FractalAnomalyAlgorithm()._determine_optimal_radii(float(pixel_size))
            except Exception:
                radii = [4, 8, 16, 32, 64]
        try:
            max_radius = max(int(round(float(r))) for r in radii if float(r) > 0)
        except Exception:
            max_radius = 64
        # Keep tile halo aligned with fractal map_overlap depth=max_radius*3+1.
        required = max(required, int(max_radius * 3 + 1))

    mode = str(algo_params.get("mode", "local")).lower()
    spatial_algorithms = {
        "hillshade",
        "slope",
        "specular",
        "atmospheric_scattering",
        "curvature",
        "ambient_occlusion",
        "openness",
        "multi_light_uncertainty",
    }
    if mode == "spatial" and algorithm in spatial_algorithms:
        radii = algo_params.get("radii")
        if not radii:
            try:
                from ..algorithms.common.spatial_mode import determine_spatial_radii
                radii = determine_spatial_radii(pixel_size=float(pixel_size))
            except Exception:
                radii = [5, 20, 80, 320]
        try:
            max_radius = max(int(round(float(r))) for r in radii if float(r) > 0)
        except Exception:
            max_radius = 32
        # Align with spatial smoothing + local compute depth.
        required = max(required, int(max_radius * 2 + 2))

    # Keep alignment with current tiling preferences.
    return max(32, ((required + 31) // 32) * 32)


def _percentile_minmax_np(data: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Optional[Tuple[float, float]]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return None
    mn = float(np.percentile(valid, pmin))
    mx = float(np.percentile(valid, pmax))
    if not (np.isfinite(mn) and np.isfinite(mx)) or mx <= mn:
        return None
    return (mn, mx)


def _normalize_by_global_stats(
    arr: np.ndarray,
    stats: Union[Tuple[float, float], List[Tuple[float, float]], None],
    target_range: str = "unit",
) -> np.ndarray:
    """Apply robust global normalization for tile consistency."""
    if stats is None:
        return arr

    if arr.ndim == 2 and isinstance(stats, (tuple, list)) and len(stats) == 2 and not isinstance(stats[0], (tuple, list)):
        mn, mx = float(stats[0]), float(stats[1])
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            src = arr.astype(np.float32, copy=False)
            if target_range == "unit_surprise":
                # Match legacy visual feel with a global (tile-stable) curve.
                out = np.clip((src - mn) / (mx - mn), 0.0, 1.0)
                return np.power(out, 1.0 / 2.0).astype(np.float32, copy=False)
            if target_range == "signed":
                center = 0.5 * (mn + mx)
                half = max(abs(mx - center), abs(mn - center), 1e-6)
                out = (src - center) / half
                return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
            out = (src - mn) / (mx - mn)
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        return arr

    if arr.ndim == 2 and isinstance(stats, (tuple, list)) and len(stats) >= 3 and not isinstance(stats[0], (tuple, list)):
        mn, mx = float(stats[0]), float(stats[1])
        gamma_hint = float(stats[2])
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            src = arr.astype(np.float32, copy=False)
            if target_range == "unit_surprise":
                # Keep low-end latitude: avoid hard floor clipping at mn.
                # Use a partial floor subtraction to suppress noise while preserving gradation.
                floor = 0.2 * mn
                denom = max(mx - floor, 1e-6)
                out = np.clip((src - floor) / denom, 0.0, 1.0)
                gamma = max(1e-3, gamma_hint)
                return np.power(out, 1.0 / gamma).astype(np.float32, copy=False)
            out = (src - mn) / (mx - mn)
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        return arr

    if arr.ndim == 2 and isinstance(stats, (tuple, list)) and len(stats) == 1:
        scale = float(stats[0])
        if np.isfinite(scale) and scale > 1e-9:
            src = arr.astype(np.float32, copy=False)
            if target_range == "signed":
                out = np.tanh(src / (2.5 * scale))
                return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
            out = np.tanh(src / (2.5 * scale))
            return np.clip(0.5 * (out + 1.0), 0.0, 1.0).astype(np.float32, copy=False)
        return arr

    if arr.ndim == 3 and isinstance(stats, (tuple, list)) and len(stats) == arr.shape[2]:
        out = arr.astype(np.float32, copy=True)
        for ch in range(arr.shape[2]):
            st = stats[ch]
            if not (isinstance(st, (tuple, list)) and len(st) == 2):
                continue
            mn, mx = float(st[0]), float(st[1])
            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                if target_range == "signed":
                    center = 0.5 * (mn + mx)
                    half = max(abs(mx - center), abs(mn - center), 1e-6)
                    out[:, :, ch] = np.clip((out[:, :, ch] - center) / half, -1.0, 1.0)
                else:
                    out[:, :, ch] = np.clip((out[:, :, ch] - mn) / (mx - mn), 0.0, 1.0)
        return out

    return arr


def _compute_geotiff_tile_profile(
    base_profile: dict,
    core_w: int,
    core_h: int,
    src_crs,
    core_transform: Affine,
    nodata: Optional[float],
    result_core: np.ndarray,
) -> dict:
    """Build a tile profile that supports both single-band and RGB-like outputs."""
    if result_core.ndim == 2:
        band_count = 1
    elif result_core.ndim == 3:
        # Tile algorithms return HxWxC
        band_count = int(result_core.shape[-1])
    else:
        raise ValueError(f"Unsupported result_core ndim: {result_core.ndim}")

    # GeoTIFF tiled blocks must be multiples of 16.
    block_x = min(512, core_w)
    block_y = min(512, core_h)
    block_x = (block_x // 16) * 16
    block_y = (block_y // 16) * 16
    use_tiled = block_x >= 16 and block_y >= 16

    tile_profile = base_profile.copy()
    tile_profile.update({
        "driver": "GTiff",
        "height": core_h,
        "width": core_w,
        "count": band_count,
        "dtype": result_core.dtype,
        "crs": src_crs,
        "transform": core_transform,
        "compress": "ZSTD",
        "zlevel": 1,
        "BIGTIFF": "YES",
        "nodata": nodata,
        "NUM_THREADS": "ALL_CPUS",
    })

    if use_tiled:
        tile_profile["tiled"] = True
        tile_profile["blockxsize"] = block_x
        tile_profile["blockysize"] = block_y
    else:
        tile_profile.pop("blockxsize", None)
        tile_profile.pop("blockysize", None)
        tile_profile["tiled"] = False

    return tile_profile


def _compute_global_hillshade_z_factor(input_cog_path: str, pixel_size: float) -> float:
    """Estimate a stable z-factor from a global overview sample to reduce tile seams."""
    with rasterio.open(input_cog_path, "r") as src:
        sample_max = 2048
        scale = max(src.width / sample_max, src.height / sample_max, 1.0)
        sample_w = max(64, int(src.width / scale))
        sample_h = max(64, int(src.height / scale))
        sample = src.read(
            1,
            out_shape=(sample_h, sample_w),
            resampling=Resampling.average,
            out_dtype=np.float32,
        )
        nodata = src.nodata

    if nodata is not None:
        valid = sample[sample != nodata]
    else:
        valid = sample.reshape(-1)

    if valid.size == 0:
        return 1.0

    p95 = float(np.percentile(valid, 95))
    p5 = float(np.percentile(valid, 5))
    dem_range = max(p95 - p5, 1.0)
    return float(pixel_size * 5.0 / dem_range)


def _compute_global_rvi_stats(
    input_cog_path: str,
    nodata: Optional[float],
    pixel_size: float,
    target_distances: Optional[List[float]],
    weights: Optional[List[float]],
    algo_params: dict,
) -> Optional[Tuple[float]]:
    """Estimate global RVI normalization stats from an overview sample once."""
    try:
        from ..algorithms.dask_shared import compute_rvi_efficient_block, rvi_stat_func
    except Exception as exc:
        logger.warning(f"RVI global stats helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale))
            sample_h = max(128, int(src.height / scale))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        sample_pixel_size = float(pixel_size * scale)
        radii, rvi_weights = _normalize_rvi_radii_and_weights(
            target_distances=target_distances,
            weights=weights,
            pixel_size=sample_pixel_size,
            manual_radii=algo_params.get("radii"),
            manual_weights=algo_params.get("weights"),
        )
        if not radii:
            return None

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        rvi_sample = compute_rvi_efficient_block(sample_gpu, radii=radii, weights=rvi_weights)
        stats = rvi_stat_func(rvi_sample)
        if not stats:
            return None
        std_global = float(stats[0])
        if not np.isfinite(std_global) or std_global <= 0:
            return None
        return (std_global,)
    except Exception as exc:
        logger.warning(f"Failed to compute global RVI stats; fallback to per-run stats: {exc}")
        return None


def _compute_global_multiscale_terrain_stats(
    input_cog_path: str,
    nodata: Optional[float],
    scales: Optional[List[float]],
    weights: Optional[List[float]],
) -> Optional[Tuple[float, float]]:
    """Estimate global normalization stats for multiscale_terrain once."""
    try:
        from cupyx.scipy.ndimage import gaussian_filter
    except Exception as exc:
        logger.warning(f"Multiscale global stats helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        use_scales = list(scales) if scales else [1, 10, 50, 100]
        if not use_scales:
            use_scales = [1, 10, 50, 100]

        if weights is None or len(weights) != len(use_scales):
            use_weights = np.asarray([1.0 / max(s, 1e-6) for s in use_scales], dtype=np.float32)
        else:
            use_weights = np.asarray(weights, dtype=np.float32)
        use_weights = use_weights / max(float(use_weights.sum()), 1e-6)

        block = cp.asarray(sample, dtype=cp.float32)
        nan_mask = cp.isnan(block)
        valid = (~nan_mask).astype(cp.float32)

        combined = cp.zeros_like(block, dtype=cp.float32)
        weight_sum = cp.zeros_like(block, dtype=cp.float32)
        for s, w in zip(use_scales, use_weights):
            sigma = max(float(s), 0.5)
            if nan_mask.any():
                filled = cp.where(nan_mask, 0, block)
                smoothed_values = gaussian_filter(filled * valid, sigma=sigma, mode="nearest")
                smoothed_weights = gaussian_filter(valid, sigma=sigma, mode="nearest")
                denom = cp.where(smoothed_weights > 1e-6, smoothed_weights, 1.0)
                smoothed = smoothed_values / denom
                smoothed = cp.where(smoothed_weights > 1e-6, smoothed, cp.nan)
            else:
                smoothed = gaussian_filter(block, sigma=sigma, mode="nearest")

            detail = block - smoothed
            detail = cp.where(nan_mask, cp.nan, detail)
            valid_detail = ~cp.isnan(detail)
            combined[valid_detail] += detail[valid_detail] * float(w)
            weight_sum[valid_detail] += float(w)

        denom = cp.where(weight_sum > 1e-6, weight_sum, 1.0)
        combined = combined / denom
        combined = cp.where(weight_sum > 1e-6, combined, cp.nan)

        valid_data = combined[~cp.isnan(combined)]
        if valid_data.size == 0:
            return None
        norm_min = float(cp.percentile(valid_data, 5))
        norm_max = float(cp.percentile(valid_data, 95))
        if not np.isfinite(norm_min) or not np.isfinite(norm_max):
            return None
        if norm_max <= norm_min:
            return None
        return (norm_min, norm_max)
    except Exception as exc:
        logger.warning(f"Failed to compute multiscale global stats; fallback to per-tile stats: {exc}")
        return None


def _compute_global_visual_saliency_stats(
    input_cog_path: str,
    nodata: Optional[float],
    pixel_size: float,
    scales: Optional[List[float]],
) -> Optional[Tuple[float, float]]:
    """Estimate global normalization stats for visual_saliency once."""
    try:
        from ..algorithms.dask_shared import compute_visual_saliency_block, visual_saliency_stat_func
    except Exception as exc:
        logger.warning(f"Visual saliency global stats helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.average,
                out_dtype=np.float32,
            )

        if nodata is not None:
            sample = np.where(sample == nodata, np.nan, sample)

        use_scales = list(scales) if scales else [2, 4, 8, 16]
        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        saliency = compute_visual_saliency_block(
            sample_gpu,
            scales=use_scales,
            pixel_size=float(pixel_size * scale_factor),
            normalize=False,
        )
        stats = visual_saliency_stat_func(saliency)
        if not stats:
            return None
        mn, mx = float(stats[0]), float(stats[1])
        if not np.isfinite(mn) or not np.isfinite(mx):
            return None
        if mx <= mn:
            return None
        return (mn, mx)
    except Exception as exc:
        logger.warning(f"Failed to compute visual saliency global stats; fallback to per-tile stats: {exc}")
        return None


def _compute_global_lrm_scale(
    input_cog_path: str,
    nodata: Optional[float],
    pixel_size: float,
    kernel_size: int,
) -> Optional[Tuple[float]]:
    """Estimate global robust LRM scale once."""
    try:
        from ..algorithms.dask_shared import compute_lrm_block, lrm_stat_func
    except Exception as exc:
        logger.warning(f"LRM global stats helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        lrm_raw = compute_lrm_block(
            sample_gpu,
            kernel_size=max(3, int(kernel_size / max(scale_factor, 1.0))),
            pixel_size=float(pixel_size * scale_factor),
            std_global=None,
            normalize=False,
        )
        stats = lrm_stat_func(lrm_raw)
        if not stats:
            return None
        scale = float(stats[0])
        if not np.isfinite(scale) or scale <= 0:
            return None
        return (scale,)
    except Exception as exc:
        logger.warning(f"Failed to compute global LRM stats; fallback to per-run stats: {exc}")
        return None


def _compute_global_fractal_stats(
    input_cog_path: str,
    nodata: Optional[float],
    pixel_size: float,
    radii: Optional[List[int]],
    smoothing_sigma: float = 1.2,
    despeckle_threshold: float = 0.35,
    despeckle_alpha_max: float = 0.30,
    detail_boost: float = 0.35,
) -> Optional[Tuple[float, float, float, float]]:
    """Estimate global fractal stats once for stable tile normalization."""
    try:
        from ..algorithms.dask_shared import (
            compute_fractal_dimension_block,
            compute_roughness_multiscale,
            fractal_stat_func,
            FractalAnomalyAlgorithm,
        )
    except Exception as exc:
        logger.warning(f"Fractal global stats helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        if not radii:
            radii = FractalAnomalyAlgorithm()._determine_optimal_radii(float(pixel_size * scale_factor))

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        fractal_raw = compute_fractal_dimension_block(
            sample_gpu,
            radii=radii,
            normalize=False,
            mean_global=None,
            std_global=None,
            smoothing_sigma=float(smoothing_sigma),
            despeckle_threshold=float(despeckle_threshold),
            despeckle_alpha_max=float(despeckle_alpha_max),
            detail_boost=float(detail_boost),
        )
        stats = fractal_stat_func(fractal_raw)
        if not stats:
            return None
        mean_d, std_d = float(stats[0]), float(stats[1])
        if not np.isfinite(mean_d) or not np.isfinite(std_d) or std_d <= 0:
            return None
        relief_p10, relief_p75 = 0.0, 1.0
        try:
            rough_sigmas = compute_roughness_multiscale(sample_gpu, radii, window_mult=3, detrend=True)
            roughness = cp.mean(rough_sigmas, axis=2)
            valid_rough = roughness[~cp.isnan(roughness)]
            if valid_rough.size > 0:
                relief_p10 = float(cp.percentile(valid_rough, 10))
                relief_p75 = float(cp.percentile(valid_rough, 75))
                if not (
                    np.isfinite(relief_p10)
                    and np.isfinite(relief_p75)
                    and relief_p75 > relief_p10
                ):
                    relief_p10, relief_p75 = 0.0, 1.0
        except Exception:
            relief_p10, relief_p75 = 0.0, 1.0
        return (mean_d, std_d, relief_p10, relief_p75)
    except Exception as exc:
        logger.warning(f"Failed to compute global fractal stats; fallback to per-run stats: {exc}")
        return None


def _compute_global_scale_space_surprise_stats(
    input_cog_path: str,
    nodata: Optional[float],
    scales: Optional[List[float]],
    enhancement: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """Estimate global SSS normalization range once (legacy-style tone curve)."""
    try:
        from ..algorithms.common.kernels import scale_space_surprise as kernel_scale_space_surprise
    except Exception as exc:
        logger.warning(f"Scale-space surprise helpers unavailable: {exc}")
        return None

    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        raw = kernel_scale_space_surprise(
            sample_gpu,
            scales=scales or [1.0, 2.0, 4.0, 8.0, 16.0],
            enhancement=float(enhancement),
            normalize=False,
            nan_mask=cp.isnan(sample_gpu),
        )
        valid = cp.asnumpy(raw[~cp.isnan(raw)])
        if valid.size == 0:
            return None
        p01 = float(np.percentile(valid, 1.0))
        p99 = float(np.percentile(valid, 99.0))
        if not (np.isfinite(p01) and np.isfinite(p99)) or p99 <= p01:
            return None
        gamma = max(1.3, float(enhancement) * 1.0)
        return (p01, p99, gamma)
    except Exception as exc:
        logger.warning(
            f"Failed to compute global scale_space_surprise stats; fallback to generic stats: {exc}"
        )
        return None


def _compute_generic_global_algorithm_stats(
    input_cog_path: str,
    nodata: Optional[float],
    algorithm: str,
    sigma: float,
    multiscale_mode: bool,
    pixel_size: float,
    target_distances: Optional[List[float]],
    weights: Optional[List[float]],
    algo_params: dict,
) -> Optional[Union[Tuple[float, float], List[Tuple[float, float]]]]:
    """Compute robust global output stats by running the algorithm on a global overview sample."""
    try:
        with rasterio.open(input_cog_path, "r") as src:
            sample_max = 2048
            scale_factor = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale_factor))
            sample_h = max(128, int(src.height / scale_factor))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        dem_gpu = cp.asarray(sample, dtype=cp.float32)
        algo_instance = _load_algorithm(algorithm)
        probe_params = dict(algo_params)
        probe_params.pop("global_stats", None)
        probe_result = run_tile_algorithm(
            algo_instance=algo_instance,
            algorithm=algorithm,
            dem_gpu=dem_gpu,
            sigma=sigma,
            multiscale_mode=multiscale_mode,
            target_distances=target_distances,
            weights=weights,
            pixel_size=float(pixel_size * scale_factor),
            algo_params=probe_params,
        )
        result = cp.asnumpy(probe_result)
        if result.ndim == 2:
            return _percentile_minmax_np(result, pmin=1.0, pmax=99.0)
        if result.ndim == 3:
            stats: List[Tuple[float, float]] = []
            for ch in range(result.shape[2]):
                st = _percentile_minmax_np(result[:, :, ch], pmin=1.0, pmax=99.0)
                if st is None:
                    return None
                stats.append(st)
            return stats
        return None
    except Exception as exc:
        logger.warning(
            f"Failed to compute generic global stats for {algorithm}; "
            f"fallback to algorithm-default normalization: {exc}"
        )
        return None


def _format_algorithm_output(
    result_core: np.ndarray,
    algorithm: str,
    algo_params: dict,
    nodata: Optional[float],
) -> Tuple[np.ndarray, Optional[float]]:
    """Normalize dtype/band format per algorithm."""
    if algorithm == "hillshade":
        color_mode = str(algo_params.get("color_mode", "grayscale")).lower()
        arr = result_core
        if color_mode == "grayscale" and arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.rint(arr * 255.0).astype(np.uint8, copy=False)
        return arr, 0

    if algorithm in SIGNED_NORMALIZATION_ALGOS:
        # Signed outputs use 0 as valid signal; avoid nodata=0 collisions.
        return result_core.astype(np.float32, copy=False), np.nan

    return result_core.astype(np.float32, copy=False), nodata


def _apply_output_display_hints(output_cog_path: str, algorithm: str) -> None:
    """Attach lightweight display metadata to improve QGIS default rendering."""
    if algorithm != "rvi":
        return

    try:
        with rasterio.open(output_cog_path, "r+") as ds:
            if ds.count < 1:
                return
            ds.update_tags(
                1,
                STATISTICS_MINIMUM="-1.0",
                STATISTICS_MAXIMUM="1.0",
                STATISTICS_VALID_PERCENT="100",
            )
    except Exception as exc:
        logger.warning(f"Failed to write display hints for {algorithm}: {exc}")


def _infer_nodata_zero_from_border(src: rasterio.io.DatasetReader) -> Optional[float]:
    """Infer nodata=0 when metadata is missing and border is mostly zeros."""
    if src.nodata is not None:
        return src.nodata

    h, w = src.height, src.width
    bw = int(min(64, max(1, h // 20), max(1, w // 20)))
    if bw <= 0:
        return None

    top = src.read(1, window=Window(0, 0, w, bw), out_dtype=np.float32)
    bottom = src.read(1, window=Window(0, h - bw, w, bw), out_dtype=np.float32)
    left = src.read(1, window=Window(0, 0, bw, h), out_dtype=np.float32)
    right = src.read(1, window=Window(w - bw, 0, bw, h), out_dtype=np.float32)
    border = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])

    if border.size == 0:
        return None

    zero_ratio = float(np.count_nonzero(border == 0.0) / border.size)
    nonzero_present = np.any(border != 0.0)
    if zero_ratio >= 0.6 or (zero_ratio >= 0.3 and nonzero_present):
        return 0.0
    return None


def _resolve_writable_tmp_dir(
    requested_tmp_dir: str,
    output_cog_path: str,
    input_cog_path: str,
) -> str:
    """Resolve temp directory to a writable absolute path.

    For default relative `tiles_tmp`, prefer output directory, then input directory,
    and finally fall back to system temp.
    """
    requested = Path(requested_tmp_dir)
    candidates: List[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        # Default value should live next to output to avoid CWD permission issues.
        if requested_tmp_dir == "tiles_tmp":
            output_dir = Path(output_cog_path).resolve().parent
            input_dir = Path(input_cog_path).resolve().parent
            candidates.append(output_dir / requested)
            if input_dir != output_dir:
                candidates.append(input_dir / requested)
        # Respect user-specified relative path under current working directory.
        candidates.append((Path.cwd() / requested).resolve())

    # Last resort: system temp area.
    candidates.append(Path(tempfile.gettempdir()) / "FujiShaderGPU_tiles_tmp")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            if candidate.exists():
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
            candidate.mkdir(parents=True, exist_ok=False)
            logger.info(f"Temporary tile directory: {candidate}")
            return str(candidate)
        except Exception as exc:
            last_error = exc
            logger.warning(f"Cannot prepare tmp dir '{candidate}': {exc}")
            continue

    raise RuntimeError(
        f"Failed to create writable temporary directory. Last error: {last_error}"
    )

def _load_algorithm(name: str):
    """Load algorithm class from `algorithms/tile/<name>.py`."""
    if name in DEFAULT_ALGORITHMS:
        try:
            algorithm_class_name = DEFAULT_ALGORITHMS[name]
            module = import_module(f"..algorithms.tile.{name}", package=__package__)
            algorithm_class = getattr(module, algorithm_class_name, None)
            if algorithm_class is not None:
                return algorithm_class()

        except ImportError as e:
            logger.warning(f"Failed to load algorithm {name}: {e}")

    raise ValueError(f"Algorithm {name} not found or not available on this platform")

def process_single_tile(
    input_cog_path: str,
    tile_info: Tuple[int, int, int, int, int, int, int, int, int, int],
    tmp_tile_dir: str,
    algorithm: str,
    sigma: float,
    nodata: Optional[float],
    src_transform: Affine,
    src_crs,
    profile: dict,
    nodata_threshold: float = 1.0,
    vram_monitor: bool = False,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5,
    target_distances: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    **algo_params
) -> TileResult:
    """
    単一タイル処理（アルゴリズム選択対応版）
    """
    ty, tx, core_x, core_y, core_w, core_h, win_x_off, win_y_off, win_w, win_h = tile_info
    
    try:
        with gpu_memory_pool():
            # メモリマップド読み込み（最適化）
            window = Window(win_x_off, win_y_off, win_w, win_h)
            dem_tile = read_tile_window(input_cog_path, window)
                
            # NoData処理とスキップ判定（最適化）
            mask_nodata = None
            if nodata is not None:
                mask_nodata = (dem_tile == nodata)
                nodata_ratio = np.count_nonzero(mask_nodata) / mask_nodata.size
                
                if nodata_ratio >= nodata_threshold:
                    return TileResult(
                        ty, tx, False,
                        skipped_reason=f"NoDataが{nodata_ratio:.1%}を占める（閾値:{nodata_threshold:.1%}）"
                    )
                
                if nodata_ratio > 0.8:
                    logger.warning(f"タイル({ty}, {tx}) NoData率が高いです: {nodata_ratio:.1%}")

                if algorithm in {"rvi", "fractal_anomaly"}:
                    # RVI uses NaN-aware filters. Keep NoData as NaN to avoid boundary
                    # virtual-fill outliers that can dominate normalization.
                    dem_tile_processed = dem_tile.astype(np.float32, copy=True)
                    dem_tile_processed[mask_nodata] = np.nan
                else:
                    dem_tile_processed = _handle_nodata_ultra_fast(dem_tile, mask_nodata)
            else:
                dem_tile_processed = dem_tile

            # GPU転送（最適化）
            dem_gpu = cp.asarray(dem_tile_processed, dtype=cp.float32)

            # アルゴリズム選択と実行
            algo_instance = _load_algorithm(algorithm)
            if algorithm == "hillshade":
                # Respect geotransform orientation (e.g., dx=1, dy=-1) for light direction.
                algo_params.setdefault("pixel_scale_x", float(src_transform.a))
                algo_params.setdefault("pixel_scale_y", float(src_transform.e))

            result_gpu = run_tile_algorithm(
                algo_instance,
                algorithm,
                dem_gpu,
                sigma,
                multiscale_mode,
                target_distances,
                weights,
                pixel_size,
                algo_params,
            )

            # NoData復元（必要時のみ）
            output_nodata_for_mask = np.nan if algorithm in SIGNED_NORMALIZATION_ALGOS else nodata
            result_gpu = apply_nodata_mask(result_gpu, mask_nodata, output_nodata_for_mask)

            # CPU転送（最適化）
            result_tile = cp.asnumpy(result_gpu)
            if vram_monitor:
                used_gb = cp.get_default_memory_pool().used_bytes() / (1024**3)
                logger.debug(f"Tile ({ty}, {tx}) VRAM used: {used_gb:.2f} GB")
            del dem_gpu, result_gpu

            # コア領域抽出
            core_x_in_win = core_x - win_x_off
            core_y_in_win = core_y - win_y_off
            result_core = result_tile[
                core_y_in_win : core_y_in_win + core_h,
                core_x_in_win : core_x_in_win + core_w,
            ]
            if algo_params.get("_apply_global_stats_post", False):
                core_mask_nodata = None
                if mask_nodata is not None:
                    core_mask_nodata = mask_nodata[
                        core_y_in_win : core_y_in_win + core_h,
                        core_x_in_win : core_x_in_win + core_w,
                    ]
                result_core = _normalize_by_global_stats(
                    result_core,
                    algo_params.get("global_stats"),
                    target_range=algo_params.get("_output_norm_range", "unit"),
                )
                if core_mask_nodata is not None:
                    fill_val = output_nodata_for_mask
                    if fill_val is None:
                        fill_val = np.nan
                    result_core = result_core.astype(np.float32, copy=False)
                    result_core[core_mask_nodata] = np.float32(fill_val)
            result_core, output_nodata = _format_algorithm_output(
                result_core=result_core,
                algorithm=algorithm,
                algo_params=algo_params,
                nodata=nodata,
            )

            # 最適化されたタイルプロファイル
            core_transform = rasterio.windows.transform(
                Window(core_x, core_y, core_w, core_h), src_transform
            )

            tile_profile = _compute_geotiff_tile_profile(
                base_profile=profile,
                core_w=core_w,
                core_h=core_h,
                src_crs=src_crs,
                core_transform=core_transform,
                nodata=output_nodata,
                result_core=result_core,
            )

            tile_filename = os.path.join(
                tmp_tile_dir, f"tile_{ty:03d}_{tx:03d}.tif"
            )

            # 高速書き込み
            write_tile_output(tile_filename, result_core, tile_profile)

            return TileResult(ty, tx, True, tile_filename)
                
    except Exception as e:
        logger.error(f"Tile ({ty}, {tx}) processing failed: {e}")
        return TileResult(ty, tx, False, error_message=str(e))


def process_dem_tiles(
    input_cog_path: str,
    output_cog_path: str,
    tmp_tile_dir: str = "tiles_tmp",
    algorithm: str = "rvi",  # アルゴリズム選択を追加
    tile_size: Optional[int] = None,
    padding: Optional[int] = None,
    sigma: float = 10.0,
    max_workers: Optional[int] = None,
    nodata_threshold: float = 1.0,
    gpu_type: str = "auto",
    multiscale_mode: bool = True,
    pixel_size: Optional[float] = None,
    auto_scale_analysis: bool = True,
    cog_only: bool = False,
    **algo_params  # アルゴリズム固有のパラメータ
):
    """
    タイルベースDEM処理メイン関数（アルゴリズム選択対応版）
    """
    # COG生成のみの場合
    if cog_only:
        resume_cog_generation(
            tmp_tile_dir, 
            output_cog_path, 
            gpu_type, 
            sigma, 
            multiscale_mode, 
            pixel_size or 0.5
        )
        return
    
    logger.info(f"=== DEM→{algorithm.upper()}処理開始 ===")
    
    # ピクセルサイズ検出
    if pixel_size is None:
        pixel_size = detect_pixel_size_from_cog(input_cog_path)

    if algorithm == "hillshade":
        # Stable defaults for tile-consistent hillshade output.
        algo_params.setdefault("color_mode", "grayscale")
        algo_params.setdefault("contrast_enhance", False)
        algo_params.setdefault("z_factor", 1.0)
    
    # スケール分析（RVIの場合のみ）
    if algorithm == "rvi" and multiscale_mode and auto_scale_analysis:
        target_distances, weights = analyze_terrain_scales(input_cog_path, pixel_size)
    elif algorithm == "rvi" and multiscale_mode:
        target_distances, weights = _get_default_scales()
    else:
        target_distances, weights = None, None

    # GPU設定取得
    gpu_config = get_gpu_config(gpu_type, sigma, multiscale_mode, pixel_size, target_distances)
    
    # パラメータ最適化
    if tile_size is None:
        tile_size = gpu_config["tile_size"]
    if padding is None:
        padding = gpu_config["padding"]
    if max_workers is None:
        max_workers = gpu_config["max_workers"]

    required_padding = _required_padding_for_algorithm(
        algorithm=algorithm,
        algo_params=algo_params,
        sigma=sigma,
        pixel_size=float(pixel_size),
        target_distances=target_distances,
    )
    if algorithm == "fractal_anomaly":
        logger.info(
            f"Fractal required halo: {required_padding}px (current padding: {padding}px)"
        )
    if padding < required_padding:
        logger.info(
            f"Padding auto-expanded for {algorithm}: {padding} -> {required_padding} "
            f"(required halo to avoid tile seams)"
        )
        padding = required_padding
    
    logger.info(f"処理設定: {gpu_config['description']}")
    logger.info(f"タイルサイズ: {tile_size}x{tile_size}, パディング: {padding}, ワーカー数: {max_workers}")

    # 一時ディレクトリ準備 (Windows pythonw/IDLE環境を含む書き込み不可CWDに対応)
    tmp_tile_dir = _resolve_writable_tmp_dir(tmp_tile_dir, output_cog_path, input_cog_path)

    try:
        with rasterio.open(input_cog_path, 'r') as src:
            width = src.width
            height = src.height
            profile = src.profile.copy()
            nodata = src.nodata
            if nodata is None:
                inferred = _infer_nodata_zero_from_border(src)
                if inferred is not None:
                    nodata = inferred
                    logger.info("NoData metadata missing; inferred nodata=0 from raster border.")
            src_transform = src.transform
            src_crs = src.crs
            try:
                px_m_x, px_m_y, px_m_mean, is_geo, lat_center = metric_pixel_scales_from_metadata(
                    transform=src_transform,
                    crs=src_crs,
                    bounds=src.bounds,
                )
            except Exception:
                sign_x = 1.0 if float(src_transform.a) >= 0 else -1.0
                sign_y = 1.0 if float(src_transform.e) >= 0 else -1.0
                px_m_x = sign_x * float(pixel_size)
                px_m_y = sign_y * float(pixel_size)
                px_m_mean = float(pixel_size)
                is_geo = False
                lat_center = None

            # Inject anisotropic pixel scales for all algorithms (simple geographic support).
            algo_params.setdefault("pixel_scale_x", float(px_m_x))
            algo_params.setdefault("pixel_scale_y", float(px_m_y))
            if is_geo:
                ratio = abs(px_m_y) / max(abs(px_m_x), 1e-9)
                logger.info(
                    "Geographic DEM approximation enabled: "
                    f"lat={lat_center:.3f}, dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m, dy/dx={ratio:.4f}"
                )
            else:
                logger.info(
                    f"Projected pixel scales: dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m"
                )

            mode = str(algo_params.get("mode", "local")).lower()
            global_stats_required = _is_global_stats_required(algorithm, mode)
            output_norm_range = _normalization_target_range(algorithm)

            if algorithm == "rvi":
                global_rvi_stats = _compute_global_rvi_stats(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    pixel_size=pixel_size,
                    target_distances=target_distances,
                    weights=weights,
                    algo_params=algo_params,
                )
                if global_rvi_stats is not None:
                    algo_params["global_stats"] = global_rvi_stats
                    logger.info(
                        f"RVI global normalization stats fixed for all tiles: "
                        f"abs_p80={global_rvi_stats[0]:.6f}"
                    )
            elif algorithm == "multiscale_terrain":
                global_ms_stats = _compute_global_multiscale_terrain_stats(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    scales=algo_params.get("scales"),
                    weights=algo_params.get("weights"),
                )
                if global_ms_stats is not None:
                    algo_params["global_stats"] = global_ms_stats
                    logger.info(
                        "Multiscale global normalization stats fixed for all tiles: "
                        f"min={global_ms_stats[0]:.6f}, max={global_ms_stats[1]:.6f}"
                    )
            elif algorithm == "visual_saliency":
                global_vs_stats = _compute_global_visual_saliency_stats(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    pixel_size=float(pixel_size),
                    scales=algo_params.get("scales"),
                )
                if global_vs_stats is not None:
                    algo_params["global_stats"] = global_vs_stats
                    logger.info(
                        "Visual saliency global normalization stats fixed for all tiles: "
                        f"min={global_vs_stats[0]:.6f}, max={global_vs_stats[1]:.6f}"
                    )
            elif algorithm == "lrm":
                global_lrm_stats = _compute_global_lrm_scale(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    pixel_size=float(pixel_size),
                    kernel_size=int(algo_params.get("kernel_size", 25)),
                )
                if global_lrm_stats is not None:
                    algo_params["global_stats"] = global_lrm_stats
                    logger.info(
                        "LRM global normalization stats fixed for all tiles: "
                        f"scale={global_lrm_stats[0]:.6f}"
                    )
            elif algorithm == "fractal_anomaly":
                global_fractal_stats = _compute_global_fractal_stats(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    pixel_size=float(pixel_size),
                    radii=algo_params.get("radii"),
                    smoothing_sigma=float(algo_params.get("smoothing_sigma", 1.2)),
                    despeckle_threshold=float(algo_params.get("despeckle_threshold", 0.35)),
                    despeckle_alpha_max=float(algo_params.get("despeckle_alpha_max", 0.30)),
                    detail_boost=float(algo_params.get("detail_boost", 0.35)),
                )
                if global_fractal_stats is not None:
                    algo_params["global_stats"] = global_fractal_stats
                    logger.info(
                        "Fractal global normalization stats fixed for all tiles: "
                        f"mean={global_fractal_stats[0]:.6f}, std={global_fractal_stats[1]:.6f}"
                    )
            elif algorithm == "scale_space_surprise":
                global_sss_stats = _compute_global_scale_space_surprise_stats(
                    input_cog_path=input_cog_path,
                    nodata=nodata,
                    scales=algo_params.get("scales"),
                    enhancement=float(algo_params.get("enhancement", 2.0)),
                )
                if global_sss_stats is not None:
                    algo_params["global_stats"] = global_sss_stats
                    logger.info(
                        "Scale-space surprise global normalization stats fixed for all tiles: "
                        f"min={global_sss_stats[0]:.6f}, max={global_sss_stats[1]:.6f}, "
                        f"gamma={global_sss_stats[2]:.3f}"
                    )

            # Unified fallback: enforce global normalization for all required algorithms.
            # Native global-stats algorithms normalize internally.
            if global_stats_required:
                if "global_stats" not in algo_params:
                    generic_stats = _compute_generic_global_algorithm_stats(
                        input_cog_path=input_cog_path,
                        nodata=nodata,
                        algorithm=algorithm,
                        sigma=sigma,
                        multiscale_mode=multiscale_mode,
                        pixel_size=float(pixel_size),
                        target_distances=target_distances,
                        weights=weights,
                        algo_params=algo_params,
                    )
                    if generic_stats is not None:
                        algo_params["global_stats"] = generic_stats
                        logger.info(f"{algorithm} global normalization stats fixed for all tiles.")

                # Algorithms that already apply global stats internally should not be normalized twice.
                algo_params["_apply_global_stats_post"] = (
                    algorithm not in GLOBAL_STATS_NATIVE_ALGOS
                    and isinstance(algo_params.get("global_stats"), (tuple, list))
                )
            else:
                algo_params["_apply_global_stats_post"] = False
            algo_params["_output_norm_range"] = output_norm_range

            n_tiles_x = math.ceil(width / tile_size)
            n_tiles_y = math.ceil(height / tile_size)
            total_tiles = n_tiles_x * n_tiles_y

            logger.info(f"処理タイル数: {n_tiles_x} x {n_tiles_y} = {total_tiles}")

            # タイル情報事前計算
            tile_infos = []
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    core_x = tx * tile_size
                    core_y = ty * tile_size
                    core_w = min(tile_size, width - core_x)
                    core_h = min(tile_size, height - core_y)

                    win_x_off = max(core_x - padding, 0)
                    win_y_off = max(core_y - padding, 0)
                    win_x_end = min(core_x + core_w + padding, width)
                    win_y_end = min(core_y + core_h + padding, height)
                    win_w = win_x_end - win_x_off
                    win_h = win_y_end - win_y_off

                    tile_info = (ty, tx, core_x, core_y, core_w, core_h, 
                               win_x_off, win_y_off, win_w, win_h)
                    tile_infos.append(tile_info)

            # 並列処理
            processed_tiles = []
            skipped_tiles = []
            error_tiles = []
            completed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tile = {
                    executor.submit(
                        process_single_tile,
                        input_cog_path, tile_info, tmp_tile_dir, algorithm, sigma,
                        nodata, src_transform, src_crs, profile,
                        nodata_threshold, gpu_config.get("vram_monitor", False),
                        multiscale_mode, pixel_size, target_distances, weights,
                        **algo_params
                    ): tile_info for tile_info in tile_infos
                }

                for future in as_completed(future_to_tile):
                    result = future.result()
                    completed_count += 1
                    progress = completed_count / total_tiles * 100
                    
                    if result.success:
                        processed_tiles.append(result)
                        if completed_count % 10 == 0:
                            logger.info(f"[OK] 処理完了: {completed_count}/{total_tiles} ({progress:.1f}%)")
                    elif result.skipped_reason:
                        skipped_tiles.append(result)
                    else:
                        error_tiles.append(result)

            logger.info(f"処理結果: 成功{len(processed_tiles)}, スキップ{len(skipped_tiles)}, エラー{len(error_tiles)}")

            if error_tiles:
                error_details = "\n".join([f"タイル({t.tile_y}, {t.tile_x}): {t.error_message}" 
                                         for t in error_tiles[:3]])
                raise RuntimeError(f"タイル処理エラー:\n{error_details}")

            if not processed_tiles:
                raise ValueError("処理されたタイルがありません")

        # COG生成
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        
        # COG品質検証
        _validate_cog_for_qgis(output_cog_path)
        _apply_output_display_hints(output_cog_path, algorithm)

    except Exception as e:
        if os.path.exists(tmp_tile_dir):
            logger.error(f"エラーが発生しました ({e})。タイルディレクトリを保持します: {tmp_tile_dir}")
            logger.info("COG生成のみ実行するには: --cog-only オプションを使用してください")
        raise
    
    logger.info("=== 処理完了 ===")
    logger.info("[INFO] 生成されたCOGは最適化済みです")


def resume_cog_generation(
    tmp_tile_dir: str,
    output_cog_path: str,
    gpu_type: str = "auto",
    sigma: float = 10.0,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5
):
    """
    既存のタイルからCOG生成のみを実行する関数
    """
    logger.info("=== タイルからCOG生成再開 ===")
    
    # タイル存在確認
    if not os.path.exists(tmp_tile_dir):
        raise ValueError(f"タイルディレクトリが存在しません: {tmp_tile_dir}")
    
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))
    if not tile_files:
        raise ValueError(f"タイルファイルが見つかりません: {tmp_tile_dir}")
    
    logger.info(f"発見されたタイル数: {len(tile_files)}")
    
    # 最初のタイルから基本情報を取得
    sample_tile = tile_files[0]
    try:
        with rasterio.open(sample_tile) as src:
            logger.info(f"タイル例: {os.path.basename(sample_tile)}")
            logger.info(f"  サイズ: {src.width} x {src.height}")
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  データ型: {src.dtypes[0]}")
    except Exception as e:
        logger.warning(f"タイル情報取得警告: {e}")
    
    # GPU設定取得（元の処理設定を考慮）
    try:
        target_distances, weights = _get_default_scales()
        gpu_config = get_gpu_config(gpu_type, sigma, multiscale_mode, pixel_size, target_distances)
    except Exception as e:
        logger.warning(f"GPU設定警告: {e}, デフォルト設定を使用")
        gpu_config = get_gpu_config(gpu_type)
    
    # COG生成実行
    try:
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        _validate_cog_for_qgis(output_cog_path)
        logger.info("[OK] COG生成完了")
        
        # 成功時のクリーンアップ提案
        logger.info("\n[TIP] COG生成が完了しました。")
        logger.info("一時タイルディレクトリを削除しますか？")
        logger.info(f"削除コマンド: rm -rf {tmp_tile_dir}")
        
    except Exception as e:
        logger.error(f"COG生成エラー: {e}")
        logger.error(f"タイルディレクトリは保持されています: {tmp_tile_dir}")
        raise
