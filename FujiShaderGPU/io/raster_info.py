"""Raster metadata helpers."""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import rasterio

logger = logging.getLogger(__name__)


def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """(meters/degree longitude, meters/degree latitude) at a latitude.

    WGS84 series expansion, the single conversion used everywhere (CLI
    logging, Dask metadata detection, per-tile scaling) so metre-radius
    conversions cannot drift between backends.  The simple
    ``111320 * cos(lat)`` form deviated from this by 0.1-0.7% per axis.
    """
    lat = math.radians(float(lat_deg))
    m_lat = (
        111132.92
        - 559.82 * math.cos(2.0 * lat)
        + 1.175 * math.cos(4.0 * lat)
        - 0.0023 * math.cos(6.0 * lat)
    )
    m_lon = (
        111412.84 * math.cos(lat)
        - 93.5 * math.cos(3.0 * lat)
        + 0.118 * math.cos(5.0 * lat)
    )
    # The longitude scale collapses at the poles; keep it positive so signed
    # per-axis pixel scales never flip or hit zero.
    return max(1e-6, float(m_lon)), float(m_lat)


def metric_pixel_scales_from_metadata(
    *,
    transform,
    crs,
    bounds,
) -> Tuple[float, float, float, bool, Optional[float]]:
    """Return signed x/y pixel scales in meters and mean absolute pixel size.

    Returns:
    - pixel_scale_x_m_signed
    - pixel_scale_y_m_signed
    - pixel_size_mean_m
    - is_geographic
    - center_latitude (None for projected CRS)
    """
    sx = float(transform.a)
    sy = float(transform.e)
    abs_x = abs(sx)
    abs_y = abs(sy)

    if crs is None:
        if hasattr(bounds, "left"):
            left, bottom, right, top = (
                float(bounds.left), float(bounds.bottom),
                float(bounds.right), float(bounds.top),
            )
        else:
            left, bottom, right, top = map(float, bounds)
        plausible_lonlat_extent = (
            -180.0 <= left <= 180.0 and -180.0 <= right <= 180.0
            and -90.0 <= bottom <= 90.0 and -90.0 <= top <= 90.0
        )
        plausible_degree_pixels = 0.0 < abs_x <= 1.0 and 0.0 < abs_y <= 1.0
        if plausible_lonlat_extent and plausible_degree_pixels:
            raise ValueError(
                "Raster has no CRS and its extent/pixel size look geographic; "
                "cannot safely interpret degree-sized pixels as meters. Assign a CRS first."
            )
        logger.warning(
            "Raster has no CRS; treating coordinate units as meters because the "
            "metadata does not look like longitude/latitude."
        )

    if crs and getattr(crs, "is_geographic", False):
        # bounds may be a rasterio BoundingBox (.bottom/.top) or a plain
        # (left, bottom, right, top) tuple, which rioxarray's .rio.bounds()
        # returns; support both so geographic detection works on the dask path.
        if hasattr(bounds, "bottom"):
            _bottom, _top = float(bounds.bottom), float(bounds.top)
        else:
            _bottom, _top = float(bounds[1]), float(bounds[3])
        lat_center = 0.5 * (_bottom + _top)
        meters_per_degree_lon, meters_per_degree_lat = meters_per_degree(lat_center)
        scale_x = math.copysign(abs_x * meters_per_degree_lon, sx if sx != 0 else 1.0)
        scale_y = math.copysign(abs_y * meters_per_degree_lat, sy if sy != 0 else -1.0)
        mean_m = 0.5 * (abs(scale_x) + abs(scale_y))
        return float(scale_x), float(scale_y), float(mean_m), True, float(lat_center)

    scale_x = sx if sx != 0 else 1.0
    scale_y = sy if sy != 0 else -1.0
    mean_m = 0.5 * (abs(scale_x) + abs(scale_y))
    return float(scale_x), float(scale_y), float(mean_m), False, None


def detect_pixel_size_from_cog(input_cog_path: str) -> float:
    """Detect representative pixel size in meters from COG metadata."""
    try:
        with rasterio.open(input_cog_path) as src:
            scale_x, scale_y, pixel_size, is_geo, lat_center = metric_pixel_scales_from_metadata(
                transform=src.transform,
                crs=src.crs,
                bounds=src.bounds,
            )
            if is_geo:
                logger.info("Geographic CRS: center latitude %.3f deg", lat_center)
                logger.info("Converted meters: %.3fm x %.3fm", abs(scale_x), abs(scale_y))
            else:
                logger.info("Projected CRS: %.3fm x %.3fm", abs(scale_x), abs(scale_y))
            logger.info("Auto pixel size: %.3fm", pixel_size)
            return float(pixel_size)
    except Exception as exc:
        raise ValueError(
            f"Could not determine a safe metric pixel size for {input_cog_path}: {exc}"
        ) from exc
