"""
FujiShaderGPU/utils/scale_analysis.py
Terrain scale auto-analysis utilities.
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Tuple, List
import logging

# scipy is for the CPU fallback (optional)
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scipy is unavailable. Falling back to some CPU paths."
    )


def _analyze_scale_variances_scipy_fast(
    dem_2d: np.ndarray,
    candidate_distances: List[float],
    pixel_size: float,
) -> List[float]:
    """
    Fast Scipy CPU analysis (processed while preserving 2D spatial structure).
    """
    if not SCIPY_AVAILABLE:
        return [1.0] * len(candidate_distances)

    # Temporarily fill NaN regions with the mean before filtering
    nan_mask = np.isnan(dem_2d)
    if nan_mask.any():
        fill_val = float(np.nanmean(dem_2d))
        dem_work = np.where(nan_mask, fill_val, dem_2d)
    else:
        dem_work = dem_2d

    variances = []
    for distance in candidate_distances:
        sigma = max(0.5, distance / pixel_size)
        blurred = gaussian_filter(dem_work, sigma=sigma, mode="nearest", truncate=4.0)
        # Take the difference from the source; keep NaN where the input is NaN
        rvi = dem_2d - blurred
        variance = float(np.nanvar(rvi))
        variances.append(variance)

    return variances


def analyze_terrain_scales(
    input_cog_path: str,
    pixel_size: float,
    sample_size: int = 8192,
) -> Tuple[List[float], List[float]]:
    """
    Terrain scale analysis (ultra-fast version).
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Ultra-fast terrain scale analysis start ===")

    try:
        with rasterio.open(input_cog_path) as src:
            width = src.width
            height = src.height

            # Larger sample for better accuracy
            sample_x = max(0, (width - sample_size) // 2)
            sample_y = max(0, (height - sample_size) // 2)
            sample_w = min(sample_size, width - sample_x)
            sample_h = min(sample_size, height - sample_y)

            logger.debug(
                "Sample window: x=%d, y=%d, w=%d, h=%d",
                sample_x, sample_y, sample_w, sample_h,
            )

            window = Window(sample_x, sample_y, sample_w, sample_h)
            dem_sample = src.read(1, window=window, out_dtype=np.float32)

            # NoData handling: replace with NaN to preserve spatial structure (avoid collapsing to 1D)
            nodata = src.nodata
            if nodata is not None:
                valid_mask = dem_sample != nodata
                if np.sum(valid_mask) < dem_sample.size * 0.5:
                    return _get_default_scales()
                dem_sample = np.where(valid_mask, dem_sample, np.nan).astype(
                    np.float32
                )

            # Extended candidate scales
            candidate_distances = [
                pixel_size * 2, pixel_size * 4, pixel_size * 8,
                pixel_size * 16, pixel_size * 32, pixel_size * 64,
                pixel_size * 128, pixel_size * 256, pixel_size * 512
            ]

            # Fast GPU analysis (pass the 2D data as-is)
            scale_variances = _analyze_scale_variances_ultra_fast(
                dem_sample, candidate_distances, pixel_size
            )
            optimal_scales, optimal_weights = _select_optimal_scales_enhanced(
                candidate_distances, scale_variances
            )

            logger.info(
                "[OK] Scale analysis complete: %d scales selected", len(optimal_scales)
            )
            return optimal_scales, optimal_weights

    except Exception as e:
        logger.info("Terrain analysis error: %s", e)
        return _get_default_scales()


def _analyze_scale_variances_ultra_fast(
    dem_2d: np.ndarray,
    candidate_distances: List[float],
    pixel_size: float,
) -> List[float]:
    """
    Ultra-fast scale analysis via GPU batch processing.
    Assumes the input is a 2D array (preserving spatial structure).
    """
    logger = logging.getLogger(__name__)

    try:
        # Lazy CuPy import (allow module loading even without a GPU)
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_ndimage

        dem_gpu = cp.asarray(dem_2d, dtype=cp.float32)

        # Temporarily fill NaN regions with the mean before filtering
        nan_mask = cp.isnan(dem_gpu)
        if nan_mask.any():
            fill_val = cp.nanmean(dem_gpu)
            dem_filled = cp.where(nan_mask, fill_val, dem_gpu)
        else:
            dem_filled = dem_gpu

        variances = []
        sigma_values = [max(0.5, dist / pixel_size) for dist in candidate_distances]

        for sigma in sigma_values:
            blurred = cpx_ndimage.gaussian_filter(
                dem_filled, sigma=sigma, mode="nearest", truncate=4.0
            )
            rvi = dem_gpu - blurred
            # Compute variance excluding NaN locations
            variance = float(cp.nanvar(rvi))
            variances.append(variance)
            del blurred, rvi

        del dem_gpu, dem_filled
        return variances

    except Exception as e:
        logger.info("GPU analysis failed; falling back to fast scipy CPU: %s", e)
        return _analyze_scale_variances_scipy_fast(dem_2d, candidate_distances, pixel_size)


def _select_optimal_scales_enhanced(
    candidate_distances: List[float],
    variances: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Improved optimal-scale selection.
    """
    variances = np.array(variances)
    if np.max(variances) > 0:
        variances = variances / np.max(variances)

    # More refined selection algorithm
    n_scales = min(5, len(candidate_distances))  # at most 5 scales

    if len(variances) >= n_scales:
        # Select the top scales by variance
        top_indices = np.argsort(variances)[-n_scales:]
        top_indices = np.sort(top_indices)
    else:
        top_indices = np.arange(len(variances))

    optimal_distances = [candidate_distances[i] for i in top_indices]
    optimal_variances = [variances[i] for i in top_indices]

    # Exponential weighting (favoring small scales)
    if np.sum(optimal_variances) > 0:
        weights_raw = np.array(optimal_variances)
        # Add weighting inversely proportional to distance
        distance_weights = 1.0 / np.array(optimal_distances)
        combined_weights = weights_raw * distance_weights
        optimal_weights = (combined_weights / np.sum(combined_weights)).tolist()
    else:
        optimal_weights = [1.0 / len(optimal_distances)] * len(optimal_distances)

    return optimal_distances, optimal_weights


def _get_default_scales() -> Tuple[List[float], List[float]]:
    """
    Improved default scales.
    """
    default_distances = [2.5, 10.0, 40.0, 160.0, 320.0]
    default_weights = [0.4, 0.25, 0.2, 0.1, 0.05]
    return default_distances, default_weights
