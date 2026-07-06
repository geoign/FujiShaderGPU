"""
FujiShaderGPU/core/tile_processor.py
Core implementation of tile-based terrain analysis (for Windows/macOS).
"""
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from importlib import import_module
from ..core.gpu_memory import gpu_memory_pool
from ..config.system_config import get_gpu_config
from ..config.auto_tune import compute_max_workers, ALGORITHM_COMPLEXITY
from ..core.tile_io import read_tile_window, write_tile_output
from ..core.tile_compute import (
    run_tile_algorithm,
    apply_nodata_mask,
    _normalize_topousm_fast_radii_and_weights,
)
from ..io.raster_info import detect_pixel_size_from_cog, metric_pixel_scales_from_metadata
from ..utils.types import TileResult
from ..io.cog_builder import _build_vrt_and_cog_ultra_fast
from ..io.cog_validator import _validate_cog_for_qgis
from ..io.output_encoding import (
    SUPPORTED_OUTPUT_DTYPES,
    resolve_output_range,
    quantize_params,
    quantize_array,
)
from ..algorithms._base import Constants
from ..algorithms._norm_stats import inject_global_stats
# Spatial radii/weights are auto-derived from the DEM short side via the shared
# rule (single source of truth in algorithms.common.spatial_mode).
from ..algorithms.common.spatial_mode import (
    RADII_DRIVEN_ALGOS as AUTO_SPATIAL_RADII_ALGOS,
    MULTISCALE_REQUIRED_ALGOS,
)
from ..utils.paths import safe_abspath
import os
import math
import glob
import tempfile
import uuid
from pathlib import Path
import rasterio
import numpy as np
import cupy as cp
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Algorithms available by default (Windows/macOS)
DEFAULT_ALGORITHMS = {
    # Canonical names aligned with Dask registry
    "topousm_fast": "TopoUSMFastAlgorithm",
    "hillshade": "HillshadeAlgorithm",
    "slope": "SlopeAlgorithm",
    "specular": "SpecularAlgorithm",
    "atmospheric_scattering": "AtmosphericScatteringAlgorithm",
    "multiscale_terrain": "MultiscaleDaskAlgorithm",
    "blur": "BlurAlgorithm",
    "curvature": "CurvatureAlgorithm",
    "visual_saliency": "VisualSaliencyAlgorithm",
    "npr_edges": "NPREdgesAlgorithm",
    "ambient_occlusion": "AmbientOcclusionAlgorithm",
    "openness": "OpennessAlgorithm",
    "fractal_anomaly": "FractalAnomalyAlgorithm",
    "scale_space_surprise": "ScaleSpaceSurpriseAlgorithm",
    "multi_light_uncertainty": "MultiLightUncertaintyAlgorithm",
    "structure_tensor": "StructureTensorAlgorithm",
    "frangi": "FrangiAlgorithm",
    "lic": "LICAlgorithm",
    "phase_congruency": "PhaseCongruencyAlgorithm",
    "tv_decomposition": "TVDecompositionAlgorithm",
    "scale_drift": "ScaleDriftAlgorithm",
}

# NOTE: output normalization is owned entirely by the algorithms themselves
# (internal normalization with globally injected stats via inject_global_stats),
# identically on both backends.  The tile pipeline no longer applies any
# post-normalization of its own (the old tile-only generic p1-p99 stretch made
# Windows output diverge from the Dask backend).


def _sanitize_spatial_radii_weights_for_tile(
    algorithm: str,
    radii: Optional[List[float]],
    weights: Optional[List[float]],
    tile_size: int,
) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[str]]:
    """
    Clamp spatial radii to tile-safe values to avoid runaway padding/depth and OOM.
    Returns (radii, weights, warning_message).
    """
    if not isinstance(radii, (list, tuple)) or len(radii) == 0:
        return None, weights, None

    # Algorithms with a lightweight per-radius path (Gaussian smoothing) only
    # need integer sanitisation; downstream `_resolve_spatial_radii_weights`
    # handles de-duplication and weights.
    if algorithm not in {"ambient_occlusion", "openness"}:
        out_no_cap: List[int] = []
        for v in radii:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv <= 0:
                continue
            out_no_cap.append(max(1, int(round(fv))))
        if not out_no_cap:
            return None, weights, None
        return out_no_cap, weights, None

    # AO / openness previously capped radii (256 / 512 px) to bound per-tile
    # padding and VRAM.  That cap is removed so user-specified radii are honoured
    # as-is; the VRAM fit check in process_dem_tiles warns when the resulting
    # tile window is likely to exceed available VRAM.  We still de-duplicate
    # radii and realign weights so the algorithm receives a clean profile.
    out_r: List[int] = []
    out_w: List[float] = []
    for i, rv in enumerate(radii):
        try:
            r = int(round(float(rv)))
        except (TypeError, ValueError):
            continue
        if r <= 0:
            continue
        if r < 2:
            r = 2
        out_r.append(r)
        if isinstance(weights, (list, tuple)) and i < len(weights):
            try:
                out_w.append(float(weights[i]))
            except (TypeError, ValueError):
                out_w.append(0.0)

    if not out_r:
        return None, None, None

    # De-duplicate while preserving order.
    dedup_idx = []
    seen = set()
    for idx, r in enumerate(out_r):
        if r not in seen:
            seen.add(r)
            dedup_idx.append(idx)
    dedup_r = [out_r[i] for i in dedup_idx]

    dedup_w: Optional[List[float]] = None
    if len(out_w) == len(out_r):
        w2 = [max(0.0, out_w[i]) for i in dedup_idx]
        s = float(sum(w2))
        if s > 0:
            dedup_w = [w / s for w in w2]

    warn = None
    try:
        src_r = [int(round(float(v))) for v in radii]
        if src_r != dedup_r:
            warn = (
                f"Spatial radii de-duplicated for {algorithm}: "
                f"{list(radii)} -> {dedup_r}"
            )
    except Exception:
        pass
    return dedup_r, dedup_w, warn


def _nodata_is_nan(nodata: Optional[float]) -> bool:
    if nodata is None:
        return False
    try:
        return bool(np.isnan(float(nodata)))
    except Exception:
        return False


def _build_nodata_mask(data: np.ndarray, nodata: Optional[float]) -> Optional[np.ndarray]:
    """Build nodata mask supporting both numeric nodata and nodata=NaN."""
    if nodata is None:
        return None
    if _nodata_is_nan(nodata):
        return np.isnan(data)
    return np.isclose(data, float(nodata), rtol=0.0, atol=1e-6)


def _replace_nodata_with_nan(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Return a copy-like array where nodata cells are converted to NaN."""
    mask = _build_nodata_mask(data, nodata)
    if mask is None:
        return data
    return np.where(mask, np.nan, data)


def _required_padding_for_algorithm(
    algorithm: str,
    algo_params: dict,
    sigma: float,
    pixel_size: float,
    target_distances: Optional[List[float]],
    tile_size: int = 1024,
) -> int:
    """Minimum halo (pixels) required to avoid tile seam artifacts per algorithm.

    When the unified overview path is active (a projected spatial run with explicit
    radii, see process_dem_tiles), large radii are taken from a single global
    overview, so the per-tile halo only has to cover the SMALL radii (those below
    the overview split threshold).  This both removes the seams the old `scales`-
    default sizing caused for radii-driven runs and shrinks the halo for the
    already-correct spatial algorithms.  Otherwise the halo covers the full radii.
    """
    # Conservative default from sigma-driven filters.
    try:
        base = int(math.ceil(max(float(sigma), 0.0) * 5.0))
    except Exception:
        base = 32

    required = max(32, base)

    _MAX_DEPTH = int(Constants.MAX_DEPTH)
    _mode = str(algo_params.get("mode", "spatial")).lower()
    _is_geo = bool(algo_params.get("is_geographic_dem", False))
    # Overview path engaged for these runs (mirrors the injection gate); large
    # radii then come from the overview and need no per-tile halo.
    _overview_active = (
        _mode == "spatial" and bool(algo_params.get("radii"))
        and not _is_geo and algorithm != "topousm_fast"
    )
    # large_radius_threshold for the spatial-switch algorithms (matches
    # _nan_utils.large_radius_threshold's floor; the tile is one chunk).
    _thr_switch = max(256, int(tile_size) // 16)

    def _unified_radii(default):
        rs = algo_params.get("radii")
        if not rs:
            return [float(s) for s in default]
        try:
            return [float(r) for r in rs if float(r) > 0] or [float(s) for s in default]
        except Exception:
            return [float(s) for s in default]

    if algorithm == "visual_saliency":
        scales = _unified_radii([2, 4, 8, 16])
        # Saliency uses Gaussian center-surround at sigma up to max_scale (5*sigma
        # halo).  With the overview active only the small scales (5*scale<=MAX_DEPTH)
        # use the halo; larger scales come from the overview fields.
        small = [s for s in scales if int(s * 5) <= _MAX_DEPTH] or [min(scales)]
        max_scale = max(small) if _overview_active else max(scales)
        required = max(required, int(math.ceil(max_scale * 5.0)))
    elif algorithm == "topousm_fast":
        radii, _ = _normalize_topousm_fast_radii_and_weights(
            target_distances=target_distances,
            weights=algo_params.get("weights"),
            pixel_size=pixel_size,
            manual_radii=algo_params.get("radii"),
            manual_weights=algo_params.get("weights"),
        )
        if radii:
            # TopoUSM Fast uses a uniform (box) mean of radius R, which needs exactly R
            # pixels of real neighbour context -- not 2R.  R + small margin keeps
            # adjacent tile cores identical (seam-free) while halving the per-tile
            # halo read.  Matches the Dask map_overlap depth (max_radius + 16).
            required = max(required, int(max(radii) + 16))
    elif algorithm == "multiscale_terrain":
        scales = _unified_radii([1, 10, 50, 100])
        # Detail = block - gaussian(sigma=scale); 4*scale halo.  Overview active:
        # only small scales (4*scale<=MAX_DEPTH) use the halo.
        small = [s for s in scales if int(4 * s) <= _MAX_DEPTH] or [min(scales)]
        max_scale = max(small) if _overview_active else max(scales)
        required = max(required, int(min(max_scale * 4.0, 512)))
    elif algorithm == "scale_space_surprise":
        scales = _unified_radii([1.0, 2.0, 4.0, 8.0, 16.0])
        # The surprise kernel blurs at sigma up to max_scale; gaussian_filter
        # truncates at 4 sigma, so 4*max_scale + 1 of halo is needed for a seam-free
        # tile core.  Overview active: only small scales use the halo.
        small = [s for s in scales if int(math.ceil(s * 4.0)) + 1 <= _MAX_DEPTH] or [min(scales)]
        max_scale = max(small) if _overview_active else max(scales)
        required = max(required, int(math.ceil(max_scale * 4.0)) + 1)
    elif algorithm == "fractal_anomaly":
        radii = algo_params.get("radii")
        if not radii:
            try:
                from ..algorithms.dask_shared import FractalAnomalyAlgorithm
                radii = FractalAnomalyAlgorithm()._determine_optimal_radii(float(pixel_size))
            except Exception:
                radii = [4, 8, 16, 32, 64]
        try:
            radii_f = [float(r) for r in radii if float(r) > 0]
            if _overview_active:
                # Large radii come from the overview hybrid; halo only covers the
                # small (full-res) radii, split by the same predicate the compute
                # uses.  This is what stops the 5184px effective tile / 1-worker
                # blow-up on big rasters.
                try:
                    from ..algorithms._impl_fractal_anomaly import fractal_large_scale_predicate as _frpred
                    small = [r for r in radii_f if not _frpred(r)] or [min(radii_f)]
                except Exception:
                    small = [r for r in radii_f if int(round(r)) <= _thr_switch] or [min(radii_f)]
                max_radius = int(round(max(small)))
            else:
                max_radius = max(int(round(r)) for r in radii_f)
        except Exception:
            max_radius = 64
        # Aligned with the fractal map_overlap depth (2r + 16): roughness uses a
        # Gaussian of sigma = r/2 whose 4-sigma kernel needs ~2r of halo; +16
        # covers feature smoothing and the size-3 median.
        required = max(required, int(max_radius * 2 + 16))
    elif algorithm == "lic":
        # Streamline integration: length steps of ~1 px each way + vector-field
        # smoothing support.
        from ..algorithms._impl_lic import LIC_MAX_LENGTH
        length = min(int(algo_params.get("length", 20) or 20), LIC_MAX_LENGTH)
        flow_sigma = float(algo_params.get("flow_sigma", 1.5) or 1.5)
        required = max(required, int(length + 4 * flow_sigma + 4))
    elif algorithm == "phase_congruency":
        # FFT wraparound decays over ~lambda; 2*lambda_max of halo makes the
        # tile core seam-free.  Wavelengths are clamped to PC_MAX_WAVELENGTH.
        from ..algorithms._impl_phase_congruency import resolve_pc_wavelengths
        scales = resolve_pc_wavelengths(algo_params.get("radii"))
        required = max(required, int(2 * max(scales)) + 16)
    elif algorithm == "tv_decomposition":
        # Primal-dual TV information travels ~1 px/iteration.
        from ..algorithms._impl_tv_decomposition import TV_MAX_ITERATIONS
        iters = min(int(algo_params.get("iterations", 120) or 120),
                    TV_MAX_ITERATIONS)
        required = max(required, int(iters + 4))
    elif algorithm == "scale_drift":
        # Gaussian levels (4*sigma) for the small scales + the Lucas-Kanade
        # window / divergence halo of the combine stage.  Large scales come
        # from the overview fields (MAX_DEPTH split, mirroring the compute).
        from ..algorithms._impl_scale_drift import DRIFT_WINDOW_CAP
        scales = _unified_radii([2.0, 4.0, 8.0, 16.0, 32.0])
        small = [s for s in scales
                 if int(max(1, round(s * 4))) + 1 <= _MAX_DEPTH] or [min(scales)]
        required = max(
            required,
            int(math.ceil(max(small) * 4.0)) + int(4 * DRIFT_WINDOW_CAP) + 8)

    spatial_algorithms = {
        "hillshade",
        "slope",
        "specular",
        "atmospheric_scattering",
        "curvature",
        "ambient_occlusion",
        "openness",
        "multi_light_uncertainty",
        "npr_edges",
        # sigma = radius/2 Gaussian support -> the same 2R + margin rule.
        "structure_tensor",
        "frangi",
    }
    if _mode == "spatial" and algorithm in spatial_algorithms:
        radii = algo_params.get("radii")
        if not radii:
            try:
                from ..algorithms.common.spatial_mode import auto_spatial_radii
                radii = auto_spatial_radii(None)
            except Exception:
                radii = [2, 8, 32, 128]
        try:
            radii_f = [float(r) for r in radii if float(r) > 0]
            if _overview_active:
                # Large radii (> threshold) come from the overview; size the halo
                # from the largest SMALL radius only.
                small = [r for r in radii_f if int(round(r)) <= _thr_switch] or [min(radii_f)]
                max_radius = int(round(max(small)))
            else:
                max_radius = max(int(round(r)) for r in radii_f)
        except Exception:
            max_radius = 32
        if algorithm in {"ambient_occlusion", "openness"}:
            # Directional-search algorithms only sample up to R pixels away, so
            # they need ~R of halo (the Dask map_overlap depth is R+1), not the
            # 2R required by Gaussian radius-smoothing algorithms.  This halves
            # the per-tile halo read for AO/openness in spatial mode.
            required = max(required, int(max_radius + 16))
        else:
            # Gaussian radius smoothing uses sigma = r/2, whose 4-sigma kernel
            # needs ~2R of halo; keep 2R + a couple pixels for the local compute.
            required = max(required, int(max_radius * 2 + 2))

    # Keep alignment with current tiling preferences.
    return max(32, ((required + 31) // 32) * 32)


def _estimate_scale_count(
    algorithm: str,
    algo_params: dict,
    pixel_size: float,
    target_distances: Optional[List[float]],
) -> int:
    """Estimate multi-scale fan-out count for rough cost warning."""
    try:
        if algorithm == "topousm_fast":
            radii, _ = _normalize_topousm_fast_radii_and_weights(
                target_distances=target_distances,
                weights=algo_params.get("weights"),
                pixel_size=pixel_size,
                manual_radii=algo_params.get("radii"),
                manual_weights=algo_params.get("weights"),
            )
            return max(1, len(radii))

        if algorithm in {"multiscale_terrain", "scale_space_surprise"}:
            scales = algo_params.get("scales")
            if isinstance(scales, (list, tuple)) and len(scales) > 0:
                return len(scales)
            return 4 if algorithm == "multiscale_terrain" else 5

        if algorithm in {"visual_saliency", "fractal_anomaly"}:
            scales = algo_params.get("scales") if algorithm == "visual_saliency" else algo_params.get("radii")
            if isinstance(scales, (list, tuple)) and len(scales) > 0:
                return len(scales)
            return 4 if algorithm == "visual_saliency" else 5

        mode = str(algo_params.get("mode", "spatial")).lower()
        if mode == "spatial":
            radii = algo_params.get("radii")
            if not radii:
                try:
                    from ..algorithms.common.spatial_mode import auto_spatial_radii
                    radii = auto_spatial_radii(None)
                except Exception:
                    radii = [2, 8, 32, 128]
            return max(1, len(radii))
    except Exception:
        pass
    return 1


def _warn_if_compute_cost_high(
    *,
    algorithm: str,
    width: int,
    height: int,
    tile_size: int,
    padding: int,
    pixel_size: float,
    algo_params: dict,
    target_distances: Optional[List[float]],
    gpu_config: dict,
) -> None:
    """
    Emit a heuristic cost warning when workload is likely too high for interactive runs.
    This does not change behavior; it only warns with actionable guidance.
    """
    px_size = float(pixel_size) if pixel_size and pixel_size > 0 else 1.0
    scale_count = _estimate_scale_count(
        algorithm=algorithm,
        algo_params=algo_params,
        pixel_size=px_size,
        target_distances=target_distances,
    )

    effective_span = float(tile_size + 2 * padding)
    halo_factor = max(1.0, (effective_span * effective_span) / float(tile_size * tile_size))
    total_pixels = float(width) * float(height)

    algo_factor = float(ALGORITHM_COMPLEXITY.get(algorithm, 1.0))

    adjusted_gpix = (total_pixels * scale_count * halo_factor * algo_factor) / 1e9
    resolution_factor_vs_1m = 1.0 / max(px_size * px_size, 1e-6)

    try:
        vram_gb = float(gpu_config.get("system_info", {}).get("vram_gb", 0.0))
    except Exception:
        vram_gb = 0.0

    # VRAM-aware thresholds (scale continuously)
    high_threshold = max(10.0, vram_gb * 1.5)
    medium_threshold = max(5.0, vram_gb * 0.75)

    if adjusted_gpix >= high_threshold:
        logger.warning(
            "[COST] Estimated heavy workload: "
            f"{adjusted_gpix:.1f} Gpix-equivalent "
            f"(algo={algorithm}, scales={scale_count}, halo={halo_factor:.2f}x, "
            f"pixel_size={px_size:.3f}m, vs1m={resolution_factor_vs_1m:.2f}x). "
            "Consider larger pixel size, local mode, smaller radii, or smaller tile/padding."
        )
    elif adjusted_gpix >= medium_threshold:
        logger.info(
            "[COST] Estimated workload is high: "
            f"{adjusted_gpix:.1f} Gpix-equivalent "
            f"(algo={algorithm}, scales={scale_count}, halo={halo_factor:.2f}x, "
            f"pixel_size={px_size:.3f}m, vs1m={resolution_factor_vs_1m:.2f}x)."
        )


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


def _compute_topousm_fast_overview_coarse_field_tile(
    input_cog_path: str,
    *,
    large_radii: List[int],
    large_weights: List[float],
    nodata: Optional[float],
    sample_max: int = 2048,
):
    """Large-radius TopoUSM Fast coarse field (Sum w*mean) from the COG overview, for tiles.

    A single global field (shared by every tile) keeps the large-radius TopoUSM Fast
    contribution seam-free while letting each tile read only a small halo for the
    small radii.  Returns a CuPy 2D field, or ``None`` on any failure (caller
    then keeps the full-resolution radii path).
    """
    if not large_radii:
        return None
    try:
        from ..algorithms._impl_topousm_fast import compute_topousm_fast_large_coarse_field
    except Exception as exc:
        logger.warning(f"TopoUSM Fast overview coarse-field helper unavailable: {exc}")
        return None
    try:
        with rasterio.open(input_cog_path, "r") as src:
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            # Both axes from the SAME scale (no per-axis floor) so the actual
            # decimation stays isotropic on elongated rasters.
            sample_w = max(1, int(round(src.width / scale)))
            sample_h = max(1, int(round(src.height / scale)))
            sample_ma = src.read(
                1, out_shape=(sample_h, sample_w), resampling=Resampling.average,
                out_dtype=np.float32, masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)
            if nodata is None:
                nodata = src.nodata
        if nodata is not None:
            sample = _replace_nodata_with_nan(sample, nodata)
        coarse_dem = cp.asarray(sample, dtype=cp.float32)
        field = compute_topousm_fast_large_coarse_field(
            coarse_dem, large_radii=large_radii, large_weights=large_weights,
            decimation=float(scale),
        )
        logger.info(
            "TopoUSM Fast large-radius overview field (tile): decimation=%.1fx, large_radii=%s",
            scale, list(large_radii),
        )
        return field
    except Exception as exc:
        logger.warning(f"Failed to compute tile TopoUSM Fast overview coarse field: {exc}")
        return None



def _format_algorithm_output(
    result_core: np.ndarray,
    algorithm: str,
) -> Tuple[np.ndarray, float]:
    """Normalize dtype/band format per algorithm (all outputs float32).

    Every float tile output uses **NaN** as NoData -- the same policy as the
    Dask backend (``output_nodata_for_dtype``) and the ``prepare`` command.
    Keeping a numeric input sentinel (e.g. -9999) here was fragile: hillshade's
    [0, 1] clip turned a -9999 fill into a *valid* black 0.0 pixel.
    """
    arr = result_core.astype(np.float32, copy=False)
    if algorithm == "hillshade":
        # Keep hillshade as single-band float output.
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = np.clip(arr, 0.0, 1.0)
    return arr, np.nan


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


def _warn_implicit_nodata_candidates(
    src: rasterio.io.DatasetReader,
    threshold_ratio: float = 0.01,
) -> None:
    """Warn when likely implicit nodata values exist but nodata metadata is absent."""
    try:
        sample_w = int(min(2048, src.width))
        sample_h = int(min(2048, src.height))
        sample = src.read(
            1,
            out_shape=(sample_h, sample_w),
            out_dtype=np.float32,
            masked=False,
            resampling=Resampling.nearest,
        )
        if sample.size == 0:
            return
        # Explicit NaN in data already works as nodata signal in many paths.
        if np.isnan(sample).any():
            return

        candidates = [0.0, -9999.0, 9999.0, -32768.0, 32768.0]
        hits = []
        total = float(sample.size)
        for v in candidates:
            mask = np.isclose(sample, v, rtol=0.0, atol=1e-6)
            ratio = float(np.count_nonzero(mask) / total)
            if ratio >= threshold_ratio:
                hits.append((v, ratio))

        if hits:
            detail = ", ".join([f"{int(v) if v.is_integer() else v}:{r*100:.2f}%" for v, r in hits])
            logger.warning(
                "[NoData] NoData metadata is undefined, but possible implicit nodata values were detected "
                f"(>= {threshold_ratio*100:.1f}%): {detail}. "
                "Consider specifying --nodata explicitly."
            )
    except Exception as exc:
        logger.debug(f"Implicit nodata candidate scan skipped: {exc}")


def _is_dangerous_tmp_path(path: Path) -> bool:
    """True if *path* must never be used as a scratch tile directory.

    The current working directory (``--tmp-dir .``), the user home, and any
    filesystem root / drive anchor (e.g. ``C:\\`` or ``/``) are never legitimate
    dedicated temp dirs.  Earlier this function unconditionally ``rmtree``-d the
    resolved tmp dir, so pointing ``--tmp-dir`` at one of these (or any directory
    that already held data) could destroy the user's files; refuse them outright.
    """
    p = safe_abspath(path)
    if p == p.parent:  # filesystem root or drive anchor (C:\, /)
        return True
    sentinels: List[Path] = []
    for getter in (Path.cwd, Path.home):
        try:
            sentinels.append(safe_abspath(getter()))
        except Exception:
            pass
    return p in sentinels


def _resolve_writable_tmp_dir(
    requested_tmp_dir: str,
    output_cog_path: str,
    input_cog_path: str,
) -> str:
    """Resolve temp directory to a writable absolute path.

    For default relative `tiles_tmp`, prefer output directory, then input directory,
    and finally fall back to system temp.

    Never deletes pre-existing content: dangerous targets (cwd/home/root) are
    refused, and an existing *non-empty* directory is preserved by creating a fresh
    unique run subdirectory inside it instead of wiping it.
    """
    requested = Path(requested_tmp_dir)
    candidates: List[Path] = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        # Default value should live next to output to avoid CWD permission issues.
        if requested_tmp_dir == "tiles_tmp":
            output_dir = safe_abspath(output_cog_path).parent
            input_dir = safe_abspath(input_cog_path).parent
            candidates.append(output_dir / requested)
            if input_dir != output_dir:
                candidates.append(input_dir / requested)
        # Respect user-specified relative path under current working directory.
        candidates.append(safe_abspath(Path.cwd() / requested))

    # Last resort: system temp area.
    candidates.append(Path(tempfile.gettempdir()) / "FujiShaderGPU_tiles_tmp")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        candidate = safe_abspath(candidate)
        if _is_dangerous_tmp_path(candidate):
            logger.warning(
                "Refusing to use '%s' as a temp tile directory: cwd/home/root are "
                "never used as scratch space (their contents would be at risk).",
                candidate,
            )
            continue
        try:
            target = candidate
            if candidate.exists():
                if not candidate.is_dir():
                    # A file already occupies the path; never delete a user file --
                    # fall through to the next candidate (e.g. system temp).
                    logger.warning(
                        "Tmp dir candidate '%s' exists and is not a directory; skipping.",
                        candidate,
                    )
                    continue
                if any(candidate.iterdir()):
                    # Existing, non-empty directory: do NOT wipe it. Use a fresh,
                    # unique run subdirectory so pre-existing content is preserved.
                    target = candidate / f"fsg_{uuid.uuid4().hex[:12]}"
            target.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporary tile directory: {target}")
            return str(target)
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
    topousm_fast_weights: Optional[List[float]] = None,
    **algo_params
) -> TileResult:
    """
    Single-tile processing (with algorithm selection).
    """
    ty, tx, core_x, core_y, core_w, core_h, win_x_off, win_y_off, win_w, win_h = tile_info

    # Output quantization params (popped up front so they are not passed to the algorithm)
    _quantize_qp = algo_params.pop("_quantize_qp", None)
    _quantize_dtype = algo_params.pop("_quantize_dtype", None)

    try:
        with gpu_memory_pool(release=False):
            # Memory-mapped read (optimized)
            window = Window(win_x_off, win_y_off, win_w, win_h)
            dem_tile = read_tile_window(input_cog_path, window)
                
            # NoData handling and skip decision (optimized)
            mask_nodata = None
            if nodata is not None:
                mask_nodata = _build_nodata_mask(dem_tile, nodata)
            elif np.isnan(dem_tile).any():
                mask_nodata = np.isnan(dem_tile)

            # NoData void filling is owned by the preprocessing command
            # (`python -m FujiShaderGPU.prepare`); tiles only mask NoData here.
            if mask_nodata is not None:
                nodata_ratio = np.count_nonzero(mask_nodata) / mask_nodata.size

                if nodata_ratio >= nodata_threshold:
                    return TileResult(
                        ty, tx, False,
                        skipped_reason=(
                            f"NoData covers {nodata_ratio:.1%} "
                            f"(threshold: {nodata_threshold:.1%})"
                        ),
                    )

                if nodata_ratio > 0.8:
                    # Avoid flooding IDLE socket with thousands of warnings on huge rasters.
                    logger.debug(f"Tile({ty}, {tx}) has high NoData ratio: {nodata_ratio:.1%}")

                # Keep NoData as NaN for *all* algorithms so the shared NaN-aware
                # kernels handle the boundary (no virtual-fill cliff -> no dark
                # halo).  This matches the Dask-CUDA backend, which always feeds
                # NaN, and apply_nodata_mask restores the NoData footprint below.
                dem_tile_processed = dem_tile.astype(np.float32, copy=True)
                if mask_nodata is not None:
                    dem_tile_processed[mask_nodata] = np.nan
            else:
                dem_tile_processed = dem_tile

            # GPU transfer (optimized)
            dem_gpu = cp.asarray(dem_tile_processed, dtype=cp.float32)

            # Algorithm selection and execution
            algo_instance = _load_algorithm(algorithm)
            # Build per-tile params so geographic DEMs can use local latitude scaling.
            tile_algo_params = dict(algo_params)
            # TopoUSM Fast overview large-radius path: tell the algorithm this tile's global
            # origin so the shared coarse field is sampled at the correct position.
            if "_topousm_fast_coarse_field" in tile_algo_params:
                tile_algo_params["_topousm_fast_field_offset"] = (int(win_y_off), int(win_x_off))
            # Unified overview path (all other spatial algorithms): this tile's
            # padded-window global origin lets the shared coarse-sampling helpers
            # (_nan_utils) read the one global overview field at the correct
            # position, so large radii are seam-free across tiles.  _tile_full_shape
            # is injected globally (= raster H,W) alongside the overview field.
            if any(k in tile_algo_params for k in (
                "_overview_coarse_dem", "_vs_large_fields",
                "_sss_large_fields", "_fractal_large_fields",
            )):
                tile_algo_params["_tile_origin"] = (int(win_y_off), int(win_x_off))
            if src_crs is not None and getattr(src_crs, "is_geographic", False):
                try:
                    # Use core tile center latitude for local meter conversion;
                    # pixel_scale_x/y carry REAL signed meters per pixel (the same
                    # convention as the Dask backend and the projected path), so
                    # the DEM itself is never rescaled and elevation-based outputs
                    # keep their physical magnitude against the shared stats.
                    core_window = Window(core_x, core_y, core_w, core_h)
                    b_left, b_bottom, b_right, b_top = rasterio.windows.bounds(core_window, src_transform)
                    lat_center_tile = 0.5 * (float(b_bottom) + float(b_top))
                    meters_per_degree_lat = 111_320.0
                    meters_per_degree_lon = meters_per_degree_lat * max(
                        1e-6, abs(math.cos(math.radians(lat_center_tile)))
                    )
                    sx_deg = float(src_transform.a)
                    sy_deg = float(src_transform.e)
                    px_m_x = math.copysign(abs(sx_deg) * meters_per_degree_lon, sx_deg if sx_deg != 0 else 1.0)
                    px_m_y = math.copysign(abs(sy_deg) * meters_per_degree_lat, sy_deg if sy_deg != 0 else -1.0)

                    tile_algo_params["pixel_scale_x"] = float(px_m_x)
                    tile_algo_params["pixel_scale_y"] = float(px_m_y)
                except Exception:
                    # Fallback to globally prepared params if local conversion fails.
                    pass

            result_gpu = run_tile_algorithm(
                algo_instance,
                algorithm,
                dem_gpu,
                sigma,
                multiscale_mode,
                target_distances,
                topousm_fast_weights,
                pixel_size,
                tile_algo_params,
            )

            # Restore NoData as NaN -- the uniform float NoData policy (matches
            # the Dask backend; a numeric fill like -9999 would survive into
            # post-processing, e.g. hillshade's [0,1] clip turned it into a
            # valid black pixel).
            result_gpu = apply_nodata_mask(result_gpu, mask_nodata, np.nan)

            # Crop on GPU before PCIe transfer; the padded halo is discarded.
            core_x_in_win = core_x - win_x_off
            core_y_in_win = core_y - win_y_off
            if result_gpu.ndim == 3:
                result_core_gpu = result_gpu[
                    core_y_in_win : core_y_in_win + core_h,
                    core_x_in_win : core_x_in_win + core_w,
                    :,
                ]
            else:
                result_core_gpu = result_gpu[
                    core_y_in_win : core_y_in_win + core_h,
                    core_x_in_win : core_x_in_win + core_w,
                ]

            # CPU transfer (optimized)
            result_core = cp.asnumpy(result_core_gpu)
            if vram_monitor:
                used_gb = cp.get_default_memory_pool().used_bytes() / (1024**3)
                logger.debug(f"Tile ({ty}, {tx}) VRAM used: {used_gb:.2f} GB")
            del dem_gpu, result_gpu, result_core_gpu
            result_core, output_nodata = _format_algorithm_output(
                result_core=result_core,
                algorithm=algorithm,
            )

            # Output dtype quantization (int16/uint8): NaN (NoData) -> 0, valid values to [DN range].
            if _quantize_qp is not None and _quantize_dtype is not None:
                result_core = quantize_array(result_core, _quantize_qp, _quantize_dtype)
                output_nodata = 0.0

            # Optimized tile profile
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

            # Fast write
            write_tile_output(tile_filename, result_core, tile_profile)

            return TileResult(ty, tx, True, tile_filename)
                
    except Exception as e:
        logger.error(f"Tile ({ty}, {tx}) processing failed: {e}")
        return TileResult(ty, tx, False, error_message=str(e))


def process_dem_tiles(
    input_cog_path: str,
    output_cog_path: str,
    tmp_tile_dir: str = "tiles_tmp",
    algorithm: str = "topousm_fast",  # added algorithm selection
    tile_size: Optional[int] = None,
    padding: Optional[int] = None,
    sigma: float = 10.0,
    max_workers: Optional[int] = None,
    nodata_threshold: float = 1.0,
    multiscale_mode: bool = True,
    pixel_size: Optional[float] = None,
    cog_only: bool = False,
    nodata_override: Optional[float] = None,
    cog_backend: str = "internal",
    gdal_bin_dir: Optional[str] = None,
    **algo_params  # algorithm-specific parameters
):
    """
    Main tile-based DEM processing function (with algorithm selection).
    """
    # COG-generation-only case
    if cog_only:
        resume_cog_generation(
            tmp_tile_dir,
            output_cog_path,
            sigma,
            multiscale_mode,
            pixel_size or 0.5,
            cog_backend,
            gdal_bin_dir,
        )
        return
    
    logger.info(f"=== DEM -> {algorithm.upper()} processing start ===")

    # Output encoding: float32 (default)/int16/uint8. Not passed to the algorithm.
    output_dtype = str(algo_params.pop("output_dtype", "float32") or "float32").lower()
    output_range = algo_params.pop("output_range", None)
    if output_dtype not in SUPPORTED_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output_dtype={output_dtype!r}. Choose from {SUPPORTED_OUTPUT_DTYPES}."
        )
    if output_dtype in ("int16", "uint8"):
        _vr = resolve_output_range(algorithm, params=algo_params, override=output_range)
        if _vr is None:
            logger.warning(
                "No fixed output range for %s (unit=%s); writing float32 instead of %s.",
                algorithm, algo_params.get("unit", ""), output_dtype,
            )
        else:
            _qp = quantize_params(float(_vr[0]), float(_vr[1]), output_dtype)
            algo_params["_quantize_qp"] = _qp
            algo_params["_quantize_dtype"] = output_dtype
            # Visualization integer outputs are plain DN rasters (NoData=0); the
            # DN<->value mapping is logged but NOT embedded as GDAL scale/offset
            # (QGIS would auto-unscale the band) -- same policy as the Dask backend.
            logger.info(
                "Output dtype=%s (%s): range [%.6g, %.6g] -> DN [%d, %d], NoData=0 "
                "(DN<->value mapping: value = %.6g*DN + %.6g; not embedded as scale/offset)",
                output_dtype, "signed" if _qp["signed"] else "unsigned",
                _vr[0], _vr[1], _qp["dn_min"], _qp["dn_max"], _qp["scale"], _qp["offset"],
            )

    # Pixel-size detection
    if pixel_size is None:
        pixel_size = detect_pixel_size_from_cog(input_cog_path)

    if algorithm == "hillshade":
        # Stable defaults for tile-consistent hillshade output.
        algo_params.setdefault("contrast_enhance", False)
        algo_params.setdefault("z_factor", 1.0)

    # Resolve radii/weights ONCE from the full-raster dimensions and inject them
    # into algo_params so every downstream path (padding/cost, global stats, the
    # per-tile compute) uses the same explicit values.
    #   * --mode local  -> radii=[1], weights=[1.0] (single-pixel "simplest" run);
    #                      explicit --radii are ignored (with a warning).  For
    #                      topousm_fast, --single-scale is treated as local too.
    #   * --mode spatial -> geometric radii truncated by the DEM short side + a
    #                       2**n weight profile (auto when --radii is omitted).
    # The intrinsically multi-scale algorithms (fractal_anomaly / scale_space_
    # surprise / visual_saliency) are undefined at one scale: --mode local falls
    # back to the spatial default with a warning.
    _mode_now = str(algo_params.get("mode", "spatial")).lower()
    _is_local = _mode_now == "local" or (algorithm == "topousm_fast" and not multiscale_mode)
    if _is_local and algorithm in MULTISCALE_REQUIRED_ALGOS:
        logger.warning(
            "%s requires multiple scales; --mode local is not supported -- "
            "using the spatial default instead.", algorithm,
        )
        algo_params["mode"] = "spatial"
        _mode_now = "spatial"
        _is_local = False

    from ..algorithms.common.spatial_mode import (
        auto_spatial_profile, LOCAL_RADII, LOCAL_WEIGHTS,
    )
    if _is_local:
        if algo_params.get("radii") or algo_params.get("scales"):
            logger.warning(
                "--mode local ignores explicit radii/scales; forcing radii=%s.",
                LOCAL_RADII,
            )
        algo_params["radii"] = list(LOCAL_RADII)
        algo_params["weights"] = list(LOCAL_WEIGHTS)
        logger.info("Local mode: radii=%s, weights=%s", LOCAL_RADII, LOCAL_WEIGHTS)
    elif (
        algorithm in AUTO_SPATIAL_RADII_ALGOS
        and not algo_params.get("radii")
        and _mode_now == "spatial"
    ):
        try:
            with rasterio.open(input_cog_path) as _src:
                _short_side = min(int(_src.width), int(_src.height))
            _auto_r, _auto_w = auto_spatial_profile(_short_side)
            algo_params["radii"] = _auto_r
            if not algo_params.get("weights"):
                algo_params["weights"] = _auto_w
            logger.info(
                "Auto spatial radii (short_side=%d px): radii=%s, weights=%s",
                _short_side, _auto_r, [round(w, 3) for w in _auto_w],
            )
        except Exception as exc:
            logger.warning("Auto spatial radii determination failed: %s", exc)

    # Radii now live in algo_params (explicit pixel radii); the per-algorithm
    # padding/cost/normalization paths read them directly.
    target_distances, weights = None, None

    # Get GPU configuration
    gpu_config = get_gpu_config(
        sigma=sigma,
        multiscale_mode=multiscale_mode,
        pixel_size=pixel_size,
        target_distances=target_distances,
        algorithm=algorithm,
    )
    user_padding_provided = padding is not None
    
    # Parameter optimization
    if tile_size is None:
        tile_size = gpu_config["tile_size"]
    if padding is None:
        padding = gpu_config["padding"]
    if max_workers is None:
        max_workers = gpu_config["max_workers"]

    mode_norm = str(algo_params.get("mode", "spatial")).lower()
    spatial_algorithms = {
        "hillshade",
        "slope",
        "specular",
        "atmospheric_scattering",
        "curvature",
        "ambient_occlusion",
        "openness",
        "multi_light_uncertainty",
        "structure_tensor",
        "frangi",
    }
    if mode_norm == "spatial" and algorithm in spatial_algorithms:
        # Radii are normally injected above from the DEM short side; this is a
        # defensive fallback (full geometric sequence) if that was skipped.
        if algo_params.get("radii", None) is None:
            try:
                from ..algorithms.common.spatial_mode import auto_spatial_profile
                auto_radii, auto_weights = auto_spatial_profile(None)
                algo_params["radii"] = auto_radii
                if algo_params.get("weights", None) is None and auto_weights:
                    algo_params["weights"] = auto_weights
            except Exception:
                pass

        adj_r, adj_w, adj_warn = _sanitize_spatial_radii_weights_for_tile(
            algorithm=algorithm,
            radii=algo_params.get("radii"),
            weights=algo_params.get("weights"),
            tile_size=int(tile_size),
        )
        if adj_r is not None:
            algo_params["radii"] = adj_r
            if adj_w is not None:
                algo_params["weights"] = adj_w
            elif "weights" in algo_params and algo_params.get("weights") is not None:
                # Fall back to uniform inside algorithm when weights are invalid after adjustment.
                algo_params["weights"] = None
        if adj_warn:
            logger.warning(adj_warn)

    # TopoUSM Fast large-radius-from-overview fast path (tile).  Split radii at a
    # tile-size-aware threshold; the large radii are taken from a single global
    # overview-derived coarse field (seam-free, no large per-tile halo), so the
    # tile padding only needs to cover the small radii.  Projected-only: on
    # geographic DEMs each tile uses local-latitude metric pixel scales, which a
    # single global field cannot match, so the optimization is skipped there.
    if algorithm == "topousm_fast":
        try:
            with rasterio.open(input_cog_path, "r") as _src:
                _sx, _sy, _pxm, _is_geo, _lat = metric_pixel_scales_from_metadata(
                    transform=_src.transform, crs=_src.crs, bounds=_src.bounds,
                )
                _full_w_px, _full_h_px = int(_src.width), int(_src.height)
            if not _is_geo:
                from ..algorithms._impl_topousm_fast import (
                    topousm_fast_default_large_radius_threshold,
                    split_radii_by_threshold,
                )
                _full_r, _full_w = _normalize_topousm_fast_radii_and_weights(
                    target_distances=target_distances,
                    weights=weights,
                    pixel_size=float(pixel_size),
                    manual_radii=algo_params.get("radii"),
                    manual_weights=algo_params.get("weights"),
                )
                if _full_r:
                    _thr = topousm_fast_default_large_radius_threshold(int(tile_size))
                    _sr, _sw, _lr, _lw = split_radii_by_threshold(_full_r, _full_w, _thr)
                    if _lr:
                        _field = _compute_topousm_fast_overview_coarse_field_tile(
                            input_cog_path, large_radii=_lr, large_weights=_lw,
                            nodata=nodata_override,
                        )
                        if _field is not None:
                            # Keep the full radii for the global normalization stat,
                            # but use small radii for padding + per-tile compute.
                            algo_params["_topousm_fast_full_radii"] = list(_full_r)
                            algo_params["_topousm_fast_full_weights"] = list(_full_w)
                            algo_params["radii"] = _sr
                            algo_params["weights"] = _sw
                            algo_params["_topousm_fast_coarse_field"] = _field
                            algo_params["_topousm_fast_small_radii"] = _sr
                            algo_params["_topousm_fast_small_weights"] = _sw
                            algo_params["_topousm_fast_w_large"] = float(sum(_lw))
                            algo_params["_topousm_fast_full_shape"] = (_full_h_px, _full_w_px)
                            logger.info(
                                "TopoUSM Fast overview large-radius path (tile): small=%s, large=%s "
                                "(threshold=%dpx)",
                                _sr, _lr, _thr,
                            )
        except Exception as exc:
            logger.warning(
                "TopoUSM Fast tile overview large-radius path unavailable; using full radii: %s",
                exc,
            )

    # Detect geographic CRS before sizing padding: the unified overview path (and
    # its reduced small-radius halo) is projected-only, so padding must know this
    # up front (the full is_geographic_dem flag is set later, during stats setup).
    if "is_geographic_dem" not in algo_params:
        try:
            with rasterio.open(input_cog_path) as _crs_src:
                algo_params["is_geographic_dem"] = bool(
                    getattr(_crs_src.crs, "is_geographic", False))
        except Exception:
            algo_params["is_geographic_dem"] = False

    required_padding = _required_padding_for_algorithm(
        algorithm=algorithm,
        algo_params=algo_params,
        sigma=sigma,
        pixel_size=float(pixel_size),
        target_distances=target_distances,
        tile_size=int(tile_size),
    )
    if algorithm == "fractal_anomaly":
        logger.info(
            f"Fractal required halo: {required_padding}px (current padding: {padding}px)"
        )
    if not user_padding_provided:
        if padding != required_padding:
            logger.info(
                f"Padding auto-adjusted for {algorithm}: {padding} -> {required_padding} "
                f"(algorithm-required halo)"
            )
        padding = required_padding
    elif padding < required_padding:
        logger.info(
            f"Padding auto-expanded for {algorithm}: {padding} -> {required_padding} "
            f"(required halo to avoid tile seams)"
        )
        padding = required_padding
    
    # Avoid GPU starvation / apparent stalls on very large effective windows.
    # Effective window per tile includes halo on both sides.
    effective_span = int(tile_size + 2 * padding)
    try:
        gpu_detected = bool(gpu_config.get("system_info", {}).get("gpu_detected", False))
    except Exception:
        gpu_detected = True
    if gpu_detected:
        try:
            _vram = float(gpu_config.get("system_info", {}).get("vram_gb", 12.0))
        except Exception:
            _vram = 12.0
        recommended_workers = compute_max_workers(
            _vram,
            effective_span=effective_span,
            cpu_count=max_workers,
            algorithm=algorithm,
        )
        if recommended_workers < max_workers:
            logger.info(
                "Worker auto-throttled: "
                f"{max_workers} -> {recommended_workers} "
                f"(tile={effective_span}px, VRAM={_vram:.0f}GB)"
            )
            max_workers = recommended_workers

        # VRAM fit check.  The Windows/tile backend loads the whole padded tile
        # window onto the GPU, so large radii / halos can push a single tile past
        # available VRAM.  The TopoUSM Fast radius cap was intentionally removed, so warn
        # explicitly (rather than silently clamping) when an OOM is likely.
        _complexity = float(ALGORITHM_COMPLEXITY.get(algorithm, 1.0))
        est_tile_vram_gb = (effective_span ** 2 * 4.0 * 15.0 * _complexity) / (1024 ** 3)
        usable_vram_gb = max(0.1, _vram * 0.70)
        try:
            _radii_for_msg = algo_params.get("radii")
            _max_radius_for_msg = (
                int(max(int(round(float(r))) for r in _radii_for_msg if float(r) > 0))
                if isinstance(_radii_for_msg, (list, tuple)) and _radii_for_msg
                else None
            )
        except Exception:
            _max_radius_for_msg = None

        if est_tile_vram_gb > usable_vram_gb * 0.6:
            # Suggest a core tile size that would bring one tile within budget.
            _budget_span = int(math.sqrt(
                usable_vram_gb * (1024 ** 3) / (4.0 * 15.0 * _complexity)
            ))
            _suggested_tile = _budget_span - 2 * int(padding)
            severity = "high" if est_tile_vram_gb > usable_vram_gb else "moderate"
            crash_clause = (
                "There is a high chance of crashing mid-run with CUDA out of memory."
                if est_tile_vram_gb > usable_vram_gb
                else "VRAM is tight; workers may be auto-reduced and, depending on the environment, "
                     "out of memory may occur."
            )
            radius_clause = (
                f"The maximum radius {_max_radius_for_msg}px is dominant."
                if _max_radius_for_msg is not None
                else "Large spatial-scale settings (--radii / scales) are the cause."
            )
            if _suggested_tile >= 512:
                tile_advice = f"lower --tile-size to about {_suggested_tile}px or less"
            else:
                tile_advice = (
                    "The halo (padding) exceeds the tile body, so shrinking --tile-size alone is not enough. "
                    "Reduce the maximum radius itself"
                )
            logger.warning(
                "[VRAM] Per-tile GPU memory demand is a %s risk against available VRAM.\n"
                "  - Tile window: %dx%d px (core %d px + halo %d px x 2). %s\n"
                "  - Estimated peak VRAM/tile: ~%.1f GB vs. ~%.1f GB available"
                " (70%% of the detected %.0f GB VRAM).\n"
                "  - Cause: the Windows (tile) backend loads the padded tile window "
                "onto the GPU in full. The spatial-scale (--radii / scales) upper cap is "
                "disabled, so the given values are used as-is and the halo (padding) grows with scale, "
                "expanding the tile window quadratically.\n"
                "  - Expected impact: %s\n"
                "  - Remedies: (1) %s, (2) reduce the largest --radii, "
                "(3) use a GPU with more VRAM, (4) if very large radii are required, "
                "run on the Linux Dask-CUDA backend (chunk-distributed, strong for large radii).",
                severity,
                effective_span, effective_span, int(tile_size), int(padding), radius_clause,
                est_tile_vram_gb, usable_vram_gb, _vram,
                crash_clause,
                tile_advice,
            )

    logger.info(f"Processing profile: {gpu_config['description']}")
    logger.info(
        f"Tile size: {tile_size}x{tile_size}, padding: {padding}, "
        f"effective: {effective_span}x{effective_span}, workers: {max_workers}"
    )

    # Prepare temp directory (handles read-only CWDs incl. Windows pythonw/IDLE)
    tmp_tile_dir = _resolve_writable_tmp_dir(tmp_tile_dir, output_cog_path, input_cog_path)

    try:
        with rasterio.open(input_cog_path, 'r') as src:
            width = src.width
            height = src.height
            profile = src.profile.copy()
            # FujiShaderGPU expects an overview-bearing COG; warn (do not fail) and
            # point to the preprocessing command when overviews are missing.
            try:
                if src.count >= 1 and not src.overviews(1):
                    logger.warning(
                        "[OVERVIEW] Input has no overviews; global stats/sampling reads "
                        "will be slow. Pre-process the input into an overview-bearing COG:\n"
                        "    python -m FujiShaderGPU.prepare %s prepared_cog.tif\n"
                        "then run on 'prepared_cog.tif'.",
                        input_cog_path,
                    )
            except Exception:
                pass
            requested_mode = str(algo_params.get("mode", "spatial")).lower()
            user_radii_specified = ("radii" in algo_params) and (algo_params.get("radii") is not None)
            user_weights_specified = ("weights" in algo_params) and (algo_params.get("weights") is not None)
            # Spatial preset may include large radii (up to 512px). For small rasters,
            # force local mode unless the user explicitly provided radii/weights.
            if (
                requested_mode == "spatial"
                and not user_radii_specified
                and not user_weights_specified
                and min(int(width), int(height)) <= 1024
            ):
                algo_params["mode"] = "local"
                logger.warning(
                    "[MODE] Input DEM is small (<=1024px on one side) and no --radii/--weights were provided. "
                    "Falling back from --mode spatial to --mode local to avoid oversized auto-radius presets."
                )
            nodata = nodata_override if nodata_override is not None else src.nodata
            if nodata_override is not None:
                if _nodata_is_nan(nodata_override):
                    logger.info("NoData override applied from CLI: NaN")
                else:
                    logger.info(f"NoData override applied from CLI: {float(nodata_override):g}")
            if nodata is None:
                inferred = _infer_nodata_zero_from_border(src)
                if inferred is not None:
                    nodata = inferred
                    logger.info("NoData metadata missing; inferred nodata=0 from raster border.")
                else:
                    _warn_implicit_nodata_candidates(src, threshold_ratio=0.01)
            src_transform = src.transform
            src_crs = src.crs
            try:
                px_m_x, px_m_y, _px_m_mean, is_geo, lat_center = metric_pixel_scales_from_metadata(
                    transform=src_transform,
                    crs=src_crs,
                    bounds=src.bounds,
                )
            except Exception:
                sign_x = 1.0 if float(src_transform.a) >= 0 else -1.0
                sign_y = 1.0 if float(src_transform.e) >= 0 else -1.0
                px_m_x = sign_x * float(pixel_size)
                px_m_y = sign_y * float(pixel_size)
                is_geo = False
                lat_center = None

            # Inject anisotropic pixel scales for all algorithms.  REAL signed
            # meters per pixel on both projected and geographic DEMs -- the same
            # convention as the Dask backend.  The DEM itself is never rescaled
            # (no elevation_scale): metric handling lives entirely in these
            # scales, so elevation-based outputs keep their physical magnitude
            # and the shared normalization stats (computed on raw elevation)
            # stay correct on geographic DEMs.
            algo_params["pixel_scale_x"] = float(px_m_x)
            algo_params["pixel_scale_y"] = float(px_m_y)
            algo_params["is_geographic_dem"] = bool(is_geo)
            if is_geo:
                ratio = abs(px_m_y) / max(abs(px_m_x), 1e-9)
                logger.info(
                    "Geographic DEM approximation enabled: "
                    f"lat={lat_center:.3f}, dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m, "
                    f"dy/dx={ratio:.4f}"
                )
            else:
                logger.info(
                    f"Projected pixel scales: dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m"
                )

            mode = str(algo_params.get("mode", "spatial")).lower()

            # Shared global-statistics injection -- single source of truth with the
            # dask backend (FujiShaderGPU/algorithms/_norm_stats.inject_global_stats).
            # Computes, in order, full-resolution, mode-independent and seam-free:
            # fractal relief -> robust display range (TopoUSM Fast / multiscale_terrain /
            # visual_saliency / scale_space_surprise / ambient_occlusion / openness /
            # fractal_anomaly) -> npr_edges gradient -> specular roughness p95.
            # Output normalization itself is owned by the algorithms (identical on
            # both backends); the tile pipeline applies no post-normalization.
            inject_global_stats(input_cog_path, algorithm, algo_params, is_zarr=False)

            # ----------------------------------------------------------------
            # Unified overview coarse source for the tile backend (mirrors the
            # Dask orchestrator in core/dask_processor.py).  Read ONE decimated
            # overview of the DEM and share it across every spatial algorithm's
            # large-radius path, so large radii come from a single global field
            # (seam-free, bounded halo) instead of a per-tile coarsen.  The per-tile
            # global origin is injected in process_single_tile (_tile_origin).
            # Projected DEMs only: a single global field is incompatible with the
            # per-tile latitude metre scaling used on geographic DEMs.  TopoUSM Fast has its
            # own split above (bespoke direct path).
            if (
                mode == "spatial"
                and algo_params.get("radii")
                and algorithm != "topousm_fast"
                and not bool(algo_params.get("is_geographic_dem", False))
            ):
                try:
                    from ..algorithms._nan_utils import read_overview_coarse_dem
                    _ov_dem, _ov_decim = read_overview_coarse_dem(input_cog_path)
                    if _ov_dem is not None:
                        algo_params["_overview_coarse_dem"] = _ov_dem
                        algo_params["_overview_decimation"] = _ov_decim
                        algo_params["_tile_full_shape"] = (int(height), int(width))
                        logger.info(
                            "Unified overview coarse source (tile): %dx%d (decimation=%.1fx)",
                            int(_ov_dem.shape[1]), int(_ov_dem.shape[0]), float(_ov_decim),
                        )
                        # Hybrid algorithms also need per-large-scale overview fields.
                        _HYBRID_PFX = {
                            "visual_saliency": "_vs", "scale_space_surprise": "_sss",
                            "fractal_anomaly": "_fractal",
                        }
                        if algorithm in _HYBRID_PFX:
                            from ..algorithms._nan_utils import compute_overview_scale_fields
                            _radii = [float(r) for r in (algo_params.get("radii") or [])]
                            _pfx = _HYBRID_PFX[algorithm]
                            if algorithm == "visual_saliency":
                                from ..algorithms._impl_visual_saliency import (
                                    vs_large_scale_predicate as _pred,
                                    _vs_smooth_block as _bfn,
                                )
                                _radii = [max(0.5, float(s)) for s in _radii]
                                if len(_radii) < 4:
                                    _radii = [2.0, 4.0, 8.0, 16.0]
                            elif algorithm == "fractal_anomaly":
                                from ..algorithms._impl_fractal_anomaly import (
                                    fractal_large_scale_predicate as _pred,
                                    _fractal_roughness_block as _bfn,
                                )
                                if len(_radii) < 5:
                                    _radii = [4.0, 8.0, 16.0, 32.0, 64.0]
                            else:  # scale_space_surprise
                                from ..algorithms._impl_experimental import (
                                    sss_large_scale_predicate as _pred,
                                    _sss_smooth_block as _bfn,
                                )
                                _radii = sorted(s for s in _radii if s > 0)
                            _large = [r for r in _radii if _pred(r)]
                            if _large:
                                _fields, _ = compute_overview_scale_fields(
                                    input_cog_path, large_radii=_large, block_fn=_bfn,
                                    coarse_dem=_ov_dem, decimation=_ov_decim,
                                )
                                if _fields:
                                    algo_params[f"{_pfx}_large_fields"] = _fields
                                    algo_params[f"{_pfx}_full_shape"] = (int(height), int(width))
                        # npr_edges gradient + fractal relief are now injected
                        # unconditionally (and in the correct order) by
                        # inject_global_stats above, so they are not recomputed here.
                except Exception as exc:
                    logger.warning("Unified overview coarse source (tile) unavailable: %s", exc)

            n_tiles_x = math.ceil(width / tile_size)
            n_tiles_y = math.ceil(height / tile_size)
            total_tiles = n_tiles_x * n_tiles_y

            _warn_if_compute_cost_high(
                algorithm=algorithm,
                width=width,
                height=height,
                tile_size=tile_size,
                padding=padding,
                pixel_size=float(pixel_size),
                algo_params=algo_params,
                target_distances=target_distances,
                gpu_config=gpu_config,
            )

            logger.info(f"Tile count: {n_tiles_x} x {n_tiles_y} = {total_tiles}")

            # Precompute tile info
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

            # Parallel processing
            processed_tiles = []
            skipped_tiles = []
            error_tiles = []
            completed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Save memory: stream-submit tasks instead of creating all Futures at once
                pending = set()
                tile_iter = iter(tile_infos)
                max_pending = max_workers * 2  # max look-ahead submissions

                def _submit_next():
                    """Add a task to the pending queue; return True on success."""
                    info = next(tile_iter, None)
                    if info is None:
                        return False
                    fut = executor.submit(
                        process_single_tile,
                        input_cog_path, info, tmp_tile_dir, algorithm, sigma,
                        nodata, src_transform, src_crs, profile,
                        nodata_threshold, gpu_config.get("vram_monitor", False),
                        multiscale_mode, pixel_size, target_distances, weights,
                        **algo_params
                    )
                    pending.add(fut)
                    return True

                # Initial batch submission
                for _ in range(min(max_pending, total_tiles)):
                    if not _submit_next():
                        break

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        completed_count += 1
                        progress = completed_count / total_tiles * 100

                        if result.success:
                            processed_tiles.append(result)
                        elif result.skipped_reason:
                            skipped_tiles.append(result)
                        else:
                            error_tiles.append(result)

                        if completed_count % 10 == 0:
                            logger.info(
                                f"[OK] Progress: {completed_count}/{total_tiles} ({progress:.1f}%) "
                                f"[success={len(processed_tiles)}, skipped={len(skipped_tiles)}, error={len(error_tiles)}]"
                            )

                        # Refill with new tasks as ones complete
                        _submit_next()

            logger.info(f"Results: success={len(processed_tiles)}, skipped={len(skipped_tiles)}, error={len(error_tiles)}")

            if error_tiles:
                error_details = "\n".join([f"Tile ({t.tile_y}, {t.tile_x}): {t.error_message}" 
                                         for t in error_tiles[:3]])
                raise RuntimeError(f"Tile processing error:\n{error_details}")

            if not processed_tiles:
                raise ValueError("No tiles were processed")

        # COG generation
        _build_vrt_and_cog_ultra_fast(
            tmp_tile_dir,
            output_cog_path,
            gpu_config,
            backend=cog_backend,
            gdal_bin_dir=gdal_bin_dir,
        )
        
        # COG quality validation.  No post-hoc metadata edits: updating a COG
        # in place (display hints / scale-offset) breaks its layout guarantee
        # and GDAL 3.8+ refuses the update outright.
        _validate_cog_for_qgis(output_cog_path)

    except Exception as e:
        if os.path.exists(tmp_tile_dir):
            logger.error(f"An error occurred ({e}). Keeping the tile directory: {tmp_tile_dir}")
            logger.info("To run COG generation only: use the --cog-only option")
        raise
    
    logger.info("=== Processing complete ===")
    logger.info("[INFO] The generated COG is optimized")


def resume_cog_generation(
    tmp_tile_dir: str,
    output_cog_path: str,
    sigma: float = 10.0,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5,
    cog_backend: str = "internal",
    gdal_bin_dir: Optional[str] = None,
):
    """
    Function that only generates a COG from existing tiles.
    """
    logger.info("=== Resuming COG generation from tiles ===")
    
    # Check tile existence
    if not os.path.exists(tmp_tile_dir):
        raise ValueError(f"Tile directory does not exist: {tmp_tile_dir}")
    
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))
    if not tile_files:
        raise ValueError(f"No tile files found: {tmp_tile_dir}")
    
    logger.info(f"Tiles found: {len(tile_files)}")
    
    # Get basic info from the first tile
    sample_tile = tile_files[0]
    try:
        with rasterio.open(sample_tile) as src:
            logger.info(f"Example tile: {os.path.basename(sample_tile)}")
            logger.info(f"  Size: {src.width} x {src.height}")
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  Data type: {src.dtypes[0]}")
    except Exception as e:
        logger.warning(f"Tile info retrieval warning: {e}")
    
    # Get GPU configuration (considering the original processing settings).
    # COG-resume only sizes padding; no per-radius compute happens here.
    try:
        gpu_config = get_gpu_config(
            sigma=sigma, multiscale_mode=multiscale_mode,
            pixel_size=pixel_size, target_distances=None,
        )
    except Exception as e:
        logger.warning(f"GPU configuration warning: {e}; using default settings")
        gpu_config = get_gpu_config()
    
    # Run COG generation
    try:
        _build_vrt_and_cog_ultra_fast(
            tmp_tile_dir,
            output_cog_path,
            gpu_config,
            backend=cog_backend,
            gdal_bin_dir=gdal_bin_dir,
        )
        _validate_cog_for_qgis(output_cog_path)
        logger.info("[OK] COG generation complete")
        
        # Cleanup suggestion on success
        logger.info("\n[TIP] COG generation completed.")
        logger.info("Delete the temporary tile directory?")
        logger.info(f"Delete command: rm -rf {tmp_tile_dir}")
        
    except Exception as e:
        logger.error(f"COG generation error: {e}")
        logger.error(f"The tile directory is kept: {tmp_tile_dir}")
        raise
