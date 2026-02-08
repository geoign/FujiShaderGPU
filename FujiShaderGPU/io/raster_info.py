"""Raster metadata helpers."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import rasterio


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

    if crs and getattr(crs, "is_geographic", False):
        lat_center = 0.5 * (float(bounds.bottom) + float(bounds.top))
        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = meters_per_degree_lat * max(
            1e-6, abs(math.cos(math.radians(lat_center)))
        )
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
                print(f"Geographic CRS: center latitude {lat_center:.3f} deg")
                print(f"Converted meters: {abs(scale_x):.3f}m x {abs(scale_y):.3f}m")
            else:
                print(f"Projected CRS: {abs(scale_x):.3f}m x {abs(scale_y):.3f}m")
            print(f"Auto pixel size: {pixel_size:.3f}m")
            return float(pixel_size)
    except Exception as e:
        print(f"Pixel size detection error: {e}")
        return 0.5