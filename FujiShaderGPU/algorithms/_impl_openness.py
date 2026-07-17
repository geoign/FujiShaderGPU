"""
FujiShaderGPU/algorithms/_impl_openness.py

Openness algorithm implementation.
Module split out from dask_shared.py (Phase 2).
"""
from __future__ import annotations
import cupy as cp
import numpy as np
import dask.array as da

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    restore_nan,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
    large_radius_threshold, multiscale_response_fields,
)
# robust_unsigned_stretch_stat_func is re-exported here (not used directly in this
# module) so the global-stats pre-pass can resolve it via getattr on this module --
# see _norm_stats._NORM_STAT_SPECS, the same convention _impl_topousm_fast uses for
# topousm_fast_stat_func. Without it the getattr raised AttributeError, the stats
# were silently skipped, and the [p1,p99]->[0,1] display stretch never ran.
from ._global_stats import (  # noqa: F401  (re-export for _NORM_STAT_SPECS)
    apply_display_stretch_dask,
    robust_unsigned_stretch_stat_func,
)


def compute_openness_vectorized(block: cp.ndarray, *,
                              openness_type: str = 'positive',
                              num_directions: int = 16,
                              max_distance: int = 50,
                              pixel_size: float = 1.0,
                              pixel_scale_x: float = None,
                              pixel_scale_y: float = None) -> cp.ndarray:
    """Topographic openness (Yokoyama et al., 2002), vectorized over the grid.

    For each of ``num_directions`` azimuths, the elevation angle to every sample
    along the ray (out to ``max_distance``) is ``beta = arctan(dz / d)``.  Per
    azimuth we keep the extreme horizon angle, then average the corresponding
    angle over all azimuths -- this directional **mean** is the defining feature
    of openness (a single steepest neighbour does not dominate the result):

    * positive openness = mean over azimuths of the zenith angle ``90deg - max(beta)``
      (large on convexities: ridges, peaks, spurs);
    * negative openness = mean over azimuths of the nadir angle ``90deg + min(beta)``
      (large on concavities: valleys, channels, pits).

    The raw mean angle is then rescaled to ``[0, 1]`` (divide by 90deg, clip) and
    gamma-corrected for visualization -- the output is a display-ready relief
    image, not the quantitative openness angle in degrees.
    """
    h, w = block.shape
    nan_mask = cp.isnan(block)

    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Per-azimuth horizon extreme initial value: -90deg so cp.maximum captures the
    # steepest rise (positive), +90deg so cp.minimum captures the steepest fall.
    init_val = -cp.pi/2 if openness_type == 'positive' else cp.pi/2

    # Cast to int BEFORE dedup so e.g. max_distance=5 yields [1,2,3,4,5] instead
    # of [1,1,2,2,3,3,4,4,5] (duplicate ray samples = pure wasted shifts).
    distances = np.unique((np.linspace(0.1, 1.0, 10) * max_distance).astype(int))
    distances = distances[distances > 0]

    _sx = abs(float(pixel_scale_x)) if pixel_scale_x is not None else float(pixel_size)
    _sy = abs(float(pixel_scale_y)) if pixel_scale_y is not None else float(pixel_size)
    if _sx < 1e-9:
        _sx = float(pixel_size) if pixel_size else 1.0
    if _sy < 1e-9:
        _sy = float(pixel_size) if pixel_size else 1.0

    # Pad ONCE with the maximum offset and take shifted views by slicing.  The
    # previous per-(direction, distance) cp.pad allocated and copied a padded
    # block for every sample (up to num_directions * len(distances) times).
    D = int(max(distances)) if distances.size else 0
    if D > 0:
        padded_all = cp.pad(cp.where(nan_mask, 0.0, block), D, mode='edge')
        padded_valid = cp.pad(~nan_mask, D, mode='constant', constant_values=False)

    # Accumulate the per-azimuth angle (zenith for positive, nadir for negative)
    # then divide by the number of azimuths that contributed a valid sample.
    angle_sum = cp.zeros((h, w), dtype=cp.float32)
    dir_count = cp.zeros((h, w), dtype=cp.float32)

    for d in range(num_directions):
        direction = directions[d]
        dir_ext = cp.full((h, w), init_val, dtype=cp.float32)
        dir_valid = cp.zeros((h, w), dtype=cp.bool_)

        for r in distances:
            offset_x = int(round(float(r) * float(direction[0])))
            offset_y = int(round(float(r) * float(direction[1])))

            if offset_x == 0 and offset_y == 0:
                continue

            shifted = padded_all[D + offset_y:D + offset_y + h,
                                 D + offset_x:D + offset_x + w]
            shifted_valid = padded_valid[D + offset_y:D + offset_y + h,
                                         D + offset_x:D + offset_x + w]

            phys_dx = float(offset_x) * _sx
            phys_dy = float(offset_y) * _sy
            phys_dist = max(float(np.hypot(phys_dx, phys_dy)), 1e-9)
            angle = cp.arctan((shifted - block) / phys_dist)

            valid = shifted_valid & ~nan_mask
            if openness_type == 'positive':
                dir_ext = cp.where(valid, cp.maximum(dir_ext, angle), dir_ext)
            else:
                dir_ext = cp.where(valid, cp.minimum(dir_ext, angle), dir_ext)
            dir_valid |= valid

        # Zenith (positive) / nadir (negative) angle for this azimuth.
        dir_angle = (cp.pi/2 - dir_ext if openness_type == 'positive'
                     else cp.pi/2 + dir_ext)
        angle_sum += cp.where(dir_valid, dir_angle, 0.0)
        dir_count += dir_valid.astype(cp.float32)

    # Directional mean (Yokoyama openness), then display normalization + gamma.
    openness = angle_sum / cp.maximum(dir_count, 1.0)
    openness = cp.clip(openness / (cp.pi/2), 0, 1)

    result = cp.power(openness, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


def compute_openness_spatial_block(
    block: cp.ndarray,
    *,
    openness_type: str = 'positive',
    num_directions: int = 16,
    max_distance: int = 50,
    pixel_size: float = 1.0,
    pixel_scale_x: float = None,
    pixel_scale_y: float = None,
) -> cp.ndarray:
    ds_factor = _radius_to_downsample_factor(
        float(max_distance), block_shape=block.shape,
        pixel_size=pixel_size, algorithm_name="openness",
    )
    if ds_factor <= 1:
        return compute_openness_vectorized(
            block, openness_type=openness_type, num_directions=num_directions,
            max_distance=max_distance, pixel_size=pixel_size,
            pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
        )
    small = _downsample_nan_aware(block, ds_factor)
    ds_psx = float(abs(float(pixel_scale_x)) * ds_factor) if pixel_scale_x is not None else None
    ds_psy = float(abs(float(pixel_scale_y)) * ds_factor) if pixel_scale_y is not None else None
    result_small = compute_openness_vectorized(
        small, openness_type=openness_type, num_directions=num_directions,
        max_distance=max(2, int(round(float(max_distance) / float(ds_factor)))),
        pixel_size=float(pixel_size) * float(ds_factor),
        pixel_scale_x=ds_psx, pixel_scale_y=ds_psy,
    )
    return _upsample_to_shape(result_small, block.shape)


class OpennessAlgorithm(DaskAlgorithm):
    """Topographic openness (Yokoyama et al., 2002).

    Positive openness (``openness_type='positive'``) highlights convex relief
    (ridges/peaks); negative openness highlights concave relief (valleys/pits).
    Each pixel is the directional mean of the per-azimuth horizon angle within
    ``max_distance``, rescaled to [0,1] + gamma for display."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        openness_type = params.get('openness_type', 'positive')
        num_directions = params.get('num_directions', 16)
        pixel_size = params.get('pixel_size', 1.0)
        pixel_scale_x = params.get('pixel_scale_x', None)
        pixel_scale_y = params.get('pixel_scale_y', None)
        mode = str(params.get("mode", "local")).lower()
        radii, weights = _resolve_spatial_radii_weights(
            params.get("radii"), params.get("weights", None), pixel_size,
        )
        agg = params.get("agg", "mean")

        if mode == "spatial":
            is_geo = bool(params.get("is_geographic_dem", False))
            thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
            # Large radius from a coarsened DEM (no large per-chunk halo).
            responses = multiscale_response_fields(
                gpu_arr, [float(int(max(2, round(float(r))))) for r in radii],
                block_fn=compute_openness_spatial_block, radius_kw="max_distance",
                depth_for_scale=lambda md: int(md) + 1,
                is_large=lambda md: int(md) > thr,
                pixel_size=pixel_size, pixel_scale_x=pixel_scale_x,
                pixel_scale_y=pixel_scale_y, is_geographic=is_geo,
                coarse_dem=params.get("_overview_coarse_dem"),
                coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
                openness_type=openness_type, num_directions=num_directions)
            result = _combine_multiscale_dask(responses, weights=weights, agg=agg)
        else:
            result = gpu_arr.map_overlap(
                compute_openness_vectorized,
                depth=max_distance+1,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                openness_type=openness_type, num_directions=num_directions,
                max_distance=max_distance, pixel_size=pixel_size,
                pixel_scale_x=pixel_scale_x, pixel_scale_y=pixel_scale_y,
            )
        # Data-driven [p1, p99] -> [0, 1] contrast stretch (openness concentrates
        # in a narrow high band).  No-op unless 'global_stats' was injected.
        return apply_display_stretch_dask(result, params.get("global_stats"))

    def get_default_params(self) -> dict:
        return {
            'openness_type': 'positive',
            'num_directions': 16,
            'max_distance': 50,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }


__all__ = [
    "compute_openness_vectorized",
    "compute_openness_spatial_block",
    "OpennessAlgorithm",
]
