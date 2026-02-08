"""NoData preprocessing helpers for tile-based computation."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy is not available; using fallback CPU NoData filling."
    )


def _handle_nodata_ultra_fast(dem_tile: np.ndarray, mask_nodata: np.ndarray) -> np.ndarray:
    """Fill NoData temporarily for neighborhood filters, then mask is restored later.

    Strategy:
    1) nearest-neighbor virtual fill for masked cells
    2) weighted smoothing only on the virtual area for boundary stability
    3) fallback to valid-cell mean when scipy is unavailable
    """
    dem_processed = dem_tile.copy()

    if mask_nodata is None or not np.any(mask_nodata):
        return dem_processed

    valid_data = dem_tile[~mask_nodata]
    if valid_data.size == 0:
        dem_processed[mask_nodata] = 0.0
        return dem_processed

    fallback_value = float(np.mean(valid_data))

    if SCIPY_AVAILABLE:
        try:
            # Fill NoData cells by nearest valid neighbors.
            nearest_idx = distance_transform_edt(
                mask_nodata, return_distances=False, return_indices=True
            )
            dem_processed[mask_nodata] = dem_tile[tuple(nearest_idx[:, mask_nodata])]

            # Smooth virtual region with valid-data weighted blur to avoid sharp boundary artifacts.
            valid_mask = (~mask_nodata).astype(np.float32)
            nodata_ratio = float(np.count_nonzero(mask_nodata) / mask_nodata.size)
            sigma = float(min(8.0, max(1.5, 2.0 + nodata_ratio * 8.0)))

            smooth_values = gaussian_filter(
                dem_processed * valid_mask, sigma=sigma, mode="nearest"
            )
            smooth_weights = gaussian_filter(valid_mask, sigma=sigma, mode="nearest")
            smooth_filled = np.full_like(dem_processed, fallback_value, dtype=np.float32)
            np.divide(
                smooth_values,
                smooth_weights,
                out=smooth_filled,
                where=smooth_weights > 1e-6,
            )
            dem_processed[mask_nodata] = smooth_filled[mask_nodata]
            return dem_processed
        except Exception as exc:
            logger.debug("NoData virtual fill fallback triggered: %s", exc)

    dem_processed[mask_nodata] = fallback_value
    return dem_processed
