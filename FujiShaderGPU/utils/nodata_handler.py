"""NoData preprocessing helpers for tile-based computation."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import (
        binary_propagation,
        distance_transform_edt,
        gaussian_filter,
        label,
        zoom,
    )

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


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    """Return masked cells connected to the current block edge."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    seeds = np.zeros_like(mask, dtype=bool)
    seeds[0, :] = mask[0, :]
    seeds[-1, :] = mask[-1, :]
    seeds[:, 0] = mask[:, 0]
    seeds[:, -1] = mask[:, -1]
    if not np.any(seeds):
        return np.zeros_like(mask, dtype=bool)
    return binary_propagation(seeds, mask=mask).astype(bool, copy=False)


def _fill_sparse_holes(
    dem: np.ndarray,
    fill_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Interpolate sparse enclosed holes by nearest valid DEM plus light smoothing."""
    out = dem.astype(np.float32, copy=True)
    valid_data = out[valid_mask]
    if valid_data.size == 0:
        return out

    nearest_idx = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True
    )
    out[fill_mask] = out[tuple(nearest_idx[:, fill_mask])]

    local_ratio = float(np.count_nonzero(fill_mask) / max(1, fill_mask.size))
    sigma = float(min(6.0, max(1.0, 1.5 + local_ratio * 12.0)))
    weights = valid_mask.astype(np.float32)
    smooth_values = gaussian_filter(
        np.where(valid_mask, out, 0.0).astype(np.float32) * weights,
        sigma=sigma,
        mode="nearest",
    )
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    smooth = out.copy()
    np.divide(smooth_values, smooth_weights, out=smooth, where=smooth_weights > 1e-6)
    out[fill_mask] = smooth[fill_mask]
    return out


def _fill_dense_holes(
    dem: np.ndarray,
    fill_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Fill many enclosed holes from a smoothed low-resolution DEM surface."""
    out = dem.astype(np.float32, copy=True)
    valid_data = out[valid_mask]
    if valid_data.size == 0:
        return out

    fallback_value = float(np.mean(valid_data))
    work = np.where(valid_mask, out, fallback_value).astype(np.float32, copy=False)
    weights = valid_mask.astype(np.float32)

    max_side = max(work.shape)
    factor = int(max(2, min(16, 2 ** int(np.floor(np.log2(max(2, max_side // 512)))))))
    small_shape = (
        max(1, int(np.ceil(work.shape[0] / factor))),
        max(1, int(np.ceil(work.shape[1] / factor))),
    )
    zoom_y = small_shape[0] / max(1, work.shape[0])
    zoom_x = small_shape[1] / max(1, work.shape[1])

    small_values = zoom(work * weights, (zoom_y, zoom_x), order=1, mode="nearest")
    small_weights = zoom(weights, (zoom_y, zoom_x), order=1, mode="nearest")
    sigma = float(max(1.0, min(8.0, factor * 0.75)))
    small_values = gaussian_filter(small_values, sigma=sigma, mode="nearest")
    small_weights = gaussian_filter(small_weights, sigma=sigma, mode="nearest")
    small_surface = np.full_like(small_values, fallback_value, dtype=np.float32)
    np.divide(
        small_values,
        small_weights,
        out=small_surface,
        where=small_weights > 1e-6,
    )

    up_y = work.shape[0] / max(1, small_surface.shape[0])
    up_x = work.shape[1] / max(1, small_surface.shape[1])
    surface = zoom(small_surface, (up_y, up_x), order=1, mode="nearest")
    surface = surface[: work.shape[0], : work.shape[1]].astype(np.float32, copy=False)
    out[fill_mask] = surface[fill_mask]
    return out


def fill_enclosed_nodata_holes(
    dem_tile: np.ndarray,
    mask_nodata: np.ndarray,
    *,
    max_holes_for_interpolation: int = 256,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fill NoData holes that are not connected to the current block edge.

    Returns ``(filled_dem, remaining_nodata_mask, filled_hole_count)``.
    Cells connected to the current tile/chunk edge are preserved as NoData.
    This is conservative for chunked processing: it avoids filling oceans or
    DEM-exterior regions that pass through a block edge.
    """
    if mask_nodata is None or not np.any(mask_nodata):
        return dem_tile, mask_nodata, 0

    if not SCIPY_AVAILABLE:
        logger.warning("scipy is not available; enclosed NoData hole filling skipped.")
        return dem_tile, mask_nodata, 0

    mask = np.asarray(mask_nodata, dtype=bool)
    if mask.ndim != 2:
        return dem_tile, mask, 0

    edge_connected = _edge_connected_mask(mask)
    fill_mask = mask & ~edge_connected
    if not np.any(fill_mask):
        return dem_tile, mask, 0

    _labels, hole_count = label(fill_mask)
    valid_mask = ~mask
    if hole_count <= int(max_holes_for_interpolation):
        filled = _fill_sparse_holes(dem_tile, fill_mask, valid_mask)
    else:
        filled = _fill_dense_holes(dem_tile, fill_mask, valid_mask)

    remaining_mask = edge_connected
    return filled.astype(np.float32, copy=False), remaining_mask, int(hole_count)
