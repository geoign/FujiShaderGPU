"""
FujiShaderGPU/algorithms/_impl_blur.py

Blur algorithm: NaN-aware Gaussian smoothing of the DEM.

Unlike the other algorithms this one is *not* a normalized visualization -- it
returns the **raw smoothed elevation** (same units as the input).  That makes it
usable both as a standalone "soften the terrain" output and as the smoothing
primitive behind the preprocessing void fill (see ``io.dem_preprocess``).

The smoothing core (``handle_nan_with_gaussian``) and the large-radius overview
path (``coarse_large_radius_response``) are shared verbatim with RVI /
multiscale_terrain, so blur inherits the same NaN handling, tiling, and
overview-accelerated large-radius behaviour without forking that plumbing.
"""
from __future__ import annotations
import logging

import cupy as cp
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    handle_nan_with_gaussian,
    restore_nan,
    coarsen_factor_for_shape,
    coarse_large_radius_response,
)

logger = logging.getLogger(__name__)


def smooth_block(block: cp.ndarray, *, scale: float, **_ignored) -> cp.ndarray:
    """NaN-aware Gaussian smooth of one block (raw elevation, NaN preserved).

    ``scale`` is the Gaussian sigma in pixels.  NoData (NaN) cells stay NaN; valid
    cells receive the valid-weighted Gaussian average so the data boundary is not
    diluted by NoData.  ``**_ignored`` swallows ``pixel_size`` / ``pixel_scale_*``
    that the shared overview path forwards but a pixel-space blur does not need.
    """
    smoothed, nan_mask = handle_nan_with_gaussian(
        block, sigma=max(float(scale), 0.5), mode="nearest")
    return restore_nan(smoothed.astype(cp.float32), nan_mask)


def _resolve_radius(params: dict) -> float:
    """Single blur sigma (px).  The unified ``--radii`` first value wins (the CLI
    always sets ``radius`` from its default, so radii must take precedence to
    override it); otherwise ``radius``; otherwise the default."""
    r = None
    radii = params.get("radii") or None
    if radii:
        r = radii[0]
    if r is None:
        r = params.get("radius", None)
    try:
        rv = float(r) if r is not None else 16.0
    except (TypeError, ValueError):
        rv = 16.0
    return max(0.5, rv)


class BlurAlgorithm(DaskAlgorithm):
    """Gaussian blur of the DEM (raw smoothed elevation, no normalization)."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radius = _resolve_radius(params)
        pixel_size = float(params.get("pixel_size", 1.0))
        psx = params.get("pixel_scale_x", None)
        psy = params.get("pixel_scale_y", None)

        # Large radii are low-frequency: compute the smooth on a coarsened copy (or
        # the injected global overview on the tile backend) and bilinearly upsample,
        # exactly as multiscale_terrain handles its smooth term -- no oversized halo.
        F = coarsen_factor_for_shape(gpu_arr.shape)
        _coarse_ok_tile = (
            params.get("_overview_coarse_dem") is not None
            and params.get("_tile_origin") is not None)

        if int(4 * radius) > Constants.MAX_DEPTH and (F > 1 or _coarse_ok_tile):
            smooth = coarse_large_radius_response(
                gpu_arr, block_fn=smooth_block, radius_kw="scale",
                radius=float(radius), factor=F,
                depth_for_radius=lambda sc: min(int(4 * sc) + 1, Constants.MAX_DEPTH),
                pixel_size=pixel_size, pixel_scale_x=psx, pixel_scale_y=psy,
                coarse_cache={},
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"),
                tile_origin=params.get("_tile_origin"),
                tile_full_shape=params.get("_tile_full_shape"),
            )
            # coarse_large_radius_response already restores NaN at the NoData
            # footprint, so the smoothed elevation is returned as-is.
            return smooth.astype(cp.float32)

        depth = max(1, min(int(4 * radius), Constants.MAX_DEPTH))
        return gpu_arr.map_overlap(
            smooth_block, depth=depth, boundary="reflect",
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            scale=float(radius))

    def get_default_params(self) -> dict:
        return {
            "radius": 16.0,
            "mode": "local",
            "radii": None,
            "weights": None,
        }


__all__ = ["BlurAlgorithm", "smooth_block"]
