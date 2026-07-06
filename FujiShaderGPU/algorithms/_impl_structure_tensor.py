"""
FujiShaderGPU/algorithms/_impl_structure_tensor.py

Structure Tensor Fabric (terrain fabric orientation / anisotropy).

The 2x2 structure tensor ``J_rho = G_rho * (grad z . grad z^T)`` (Bigun &
Granlund 1987) is eigen-analysed per pixel.  The eigenvalue spread yields the
anisotropy strength (coherence, Weickert 1999) and the dominant eigenvector the
local *strike* of linear terrain fabric (lineaments, fault traces, glacial
striations, dune crests).

Orientation is pi-periodic, so all cross-scale/pixel averaging happens in the
double-angle vector representation ``(u, v) = C * (cos 2theta, sin 2theta)``
of the STRIKE direction: vectors add linearly, and scales that disagree in
direction cancel -- the combined magnitude is a direction-consistency-weighted
coherence.  Output modes (all single-band float32):

* ``coherence``   |(U, V)| robust-stretched to [0, 1] (default)
* ``orientation`` strike angle, [0, 1) == [0, 180) degrees (image frame)
* ``fabric``      0.5 + 0.5 * (U cos 2a + V sin 2a): directional lighting of
                  the fabric relative to the compass azimuth ``a`` (--azimuth)

This module also hosts the shared Gaussian-derivative helper reused by the
Frangi and Scale-Drift implementations.
"""
from __future__ import annotations
import logging
import math

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import DaskAlgorithm
from ._nan_utils import (
    restore_nan, _resolve_spatial_radii_weights, _combine_multiscale_dask,
    large_radius_threshold, multiscale_response_fields,
)
from ._normalization import NORMAL_PERCENTILE

logger = logging.getLogger(__name__)


def nan_filled(block: cp.ndarray):
    """(filled, nan_mask): NaN -> per-block nanmean (0 when all-NaN)."""
    nan_mask = cp.isnan(block)
    if not bool(nan_mask.any()):
        return block.astype(cp.float32, copy=False), nan_mask
    fill = cp.nanmean(block)
    fill = cp.where(cp.isfinite(fill), fill, cp.float32(0.0))
    return cp.where(nan_mask, fill, block).astype(cp.float32), nan_mask


def gaussian_gradients(filled: cp.ndarray, sigma_d: float):
    """Gaussian-derivative gradient (pixel units): returns (gx, gy).

    ``gy`` is the derivative along rows (image y, increasing downward /
    southward), ``gx`` along columns (eastward).  Shared by the structure
    tensor, Frangi, and scale-drift implementations.
    """
    s = max(0.5, float(sigma_d))
    gy = gaussian_filter(filled, sigma=s, order=(1, 0), mode='nearest')
    gx = gaussian_filter(filled, sigma=s, order=(0, 1), mode='nearest')
    return gx, gy


def _strike_uv(filled: cp.ndarray, radius: float, sigma_d: float):
    """Double-angle strike vector (u, v) = C*(cos 2t, sin 2t) at one radius.

    The integration scale follows the repo convention sigma = radius/2 (see
    ``_smooth_for_radius``).  The gradient eigenvector of the tensor points
    ACROSS a lineament; the strike is its perpendicular, i.e. the double-angle
    vector is negated.
    """
    gx, gy = gaussian_gradients(filled, sigma_d)
    sigma_i = max(1.0, float(radius) / 2.0)
    jxx = gaussian_filter(gx * gx, sigma=sigma_i, mode='nearest')
    jyy = gaussian_filter(gy * gy, sigma=sigma_i, mode='nearest')
    jxy = gaussian_filter(gx * gy, sigma=sigma_i, mode='nearest')
    diff = jxx - jyy
    spread = cp.sqrt(diff * diff + 4.0 * jxy * jxy)  # lambda1 - lambda2
    trace = jxx + jyy
    eps = cp.float32(1e-12)
    coh = (spread / (trace + eps)) ** 2  # Weickert coherence in [0, 1]
    inv = cp.where(spread > eps, 1.0 / (spread + eps), cp.float32(0.0))
    cos2t_g = diff * inv       # gradient-direction double angle
    sin2t_g = 2.0 * jxy * inv
    # Strike = gradient direction + 90 deg -> double angle rotates by 180 deg.
    u = (-coh * cos2t_g).astype(cp.float32)
    v = (-coh * sin2t_g).astype(cp.float32)
    return u, v


def st_component_block(block, *, radius, component='u', derivative_sigma=1.0,
                       pixel_size=1.0, pixel_scale_x=None, pixel_scale_y=None,
                       **_ignored):
    """One radius' strike-vector component field (NaN-aware, for map_overlap)."""
    filled, nan_mask = nan_filled(block)
    u, v = _strike_uv(filled, radius, derivative_sigma)
    out = u if str(component) == 'u' else v
    return restore_nan(out, nan_mask)


def _st_output_from_uv(u: cp.ndarray, v: cp.ndarray, *, st_output: str,
                       azimuth: float, normalize: bool, norm_lo: float,
                       norm_scale: float) -> cp.ndarray:
    """Final single-band output from the combined strike vector (U, V)."""
    mode = str(st_output or 'coherence').lower()
    if mode == 'orientation':
        theta = 0.5 * cp.arctan2(v, u)  # [-pi/2, pi/2)
        return ((theta / cp.float32(math.pi)) % 1.0).astype(cp.float32)
    if mode == 'fabric':
        # Compass azimuth -> image-frame angle (x = east, y = south).
        az = math.radians(float(azimuth))
        a_img = math.atan2(-math.cos(az), math.sin(az))
        out = 0.5 + 0.5 * (u * cp.float32(math.cos(2 * a_img))
                           + v * cp.float32(math.sin(2 * a_img)))
    else:  # coherence
        out = cp.sqrt(u * u + v * v)
    if normalize and float(norm_scale) > 1e-12:
        out = cp.maximum((out - cp.float32(norm_lo)) / cp.float32(norm_scale), 0.0)
    return out.astype(cp.float32)


def compute_structure_tensor_block(block, *, radii=None, weights=None,
                                   st_output='coherence', azimuth=315.0,
                                   derivative_sigma=1.0, normalize=True,
                                   global_stats=None, pixel_size=1.0,
                                   pixel_scale_x=None, pixel_scale_y=None,
                                   **_ignored):
    """Standalone full computation on one CuPy block (reference / stats prepass).

    Radii too large for the block are clamped so the prepass windows stay
    meaningful; the production Dask path handles large radii via the shared
    overview machinery instead.
    """
    filled, nan_mask = nan_filled(block)
    if not radii:
        radii = [2, 8, 32]
    short = max(8, min(int(block.shape[0]), int(block.shape[1])))
    rs = sorted({max(1, min(int(round(float(r))), short // 4)) for r in radii})
    ws = None
    if weights is not None and len(list(weights)) == len(list(radii)):
        # Clamping may merge radii; only keep weights when the count still matches.
        ws = [float(w) for w in weights] if len(rs) == len(list(radii)) else None
    acc_u = cp.zeros(block.shape, dtype=cp.float32)
    acc_v = cp.zeros(block.shape, dtype=cp.float32)
    wsum = 0.0
    for i, r in enumerate(rs):
        u, v = _strike_uv(filled, float(r), derivative_sigma)
        w = float(ws[i]) if ws is not None else 1.0
        acc_u += cp.float32(w) * u
        acc_v += cp.float32(w) * v
        wsum += w
    if wsum > 1e-12:
        acc_u /= cp.float32(wsum)
        acc_v /= cp.float32(wsum)
    lo, scale = 0.0, 0.0
    if normalize and isinstance(global_stats, (tuple, list)) and len(global_stats) >= 2:
        lo, scale = float(global_stats[0]), float(global_stats[1])
    out = _st_output_from_uv(
        acc_u, acc_v, st_output=st_output, azimuth=azimuth,
        normalize=normalize and scale > 1e-12, norm_lo=lo, norm_scale=scale)
    return restore_nan(out, nan_mask)


def st_stretch_stat_func(data):
    """Robust display range ``(p2, p98 - p2)`` (orientation mode passes through
    unnormalized, so these stats are simply unused there)."""
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    lo = float(cp.percentile(valid, 2.0))
    hi = float(cp.percentile(valid, min(98.0, NORMAL_PERCENTILE)))
    scale = hi - lo
    if scale <= 1e-9:
        return (0.0, hi if hi > 1e-9 else 1.0)
    return (lo, scale)


def _st_finalize_block(block, u, v, *, st_output, azimuth, normalize,
                       norm_lo, norm_scale):
    nan_mask = cp.isnan(block)
    out = _st_output_from_uv(u, v, st_output=st_output, azimuth=azimuth,
                             normalize=normalize, norm_lo=norm_lo,
                             norm_scale=norm_scale)
    return restore_nan(out, nan_mask)


class StructureTensorAlgorithm(DaskAlgorithm):
    """Structure Tensor Fabric (orientation / anisotropy of terrain texture)."""

    def process(self, gpu_arr, **params):
        st_output = str(params.get('st_output', 'coherence')).lower()
        azimuth = float(params.get('azimuth', 315.0))
        sigma_d = float(params.get('derivative_sigma', 1.0))
        ps = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        is_geo = bool(params.get('is_geographic_dem', False))
        normalize = bool(params.get('normalize', True)) and st_output != 'orientation'

        radii, weights = _resolve_spatial_radii_weights(
            params.get('radii'), params.get('weights', None), ps)
        radii = [float(r) for r in radii]

        stats = params.get('global_stats', None)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2
                and float(stats[1]) > 1e-9):
            stats = (0.0, 0.0)  # scale 0 -> pass-through (raw magnitude)
            if normalize:
                logger.info(
                    "structure_tensor: no global stats available; output is the "
                    "raw (unstretched) %s field.", st_output)
                normalize = False

        thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
        common = dict(
            radius_kw='radius',
            depth_for_scale=lambda r: int(2 * float(r) + 4 * sigma_d + 4),
            is_large=lambda r: int(round(float(r))) > thr,
            pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
            is_geographic=is_geo,
            coarse_dem=params.get('_overview_coarse_dem'),
            coarse_decimation=params.get('_overview_decimation'),
            tile_origin=params.get('_tile_origin'),
            tile_full_shape=params.get('_tile_full_shape'),
            derivative_sigma=sigma_d,
        )
        # Two shared-machinery passes (u then v); tensors combine linearly, so a
        # weighted mean of the double-angle components IS the multiscale tensor.
        u_fields = multiscale_response_fields(
            gpu_arr, radii, block_fn=st_component_block,
            coarse_cache={}, component='u', **common)
        v_fields = multiscale_response_fields(
            gpu_arr, radii, block_fn=st_component_block,
            coarse_cache={}, component='v', **common)
        # Signed vector components only combine linearly; min/max/sum would
        # break the double-angle algebra, so agg is fixed to the weighted mean.
        agg = str(params.get('agg', 'mean')).lower()
        if agg not in ('mean', 'stack'):
            logger.info("structure_tensor combines scales as a tensor mean; "
                        "--agg %s ignored.", agg)
        U = _combine_multiscale_dask(u_fields, weights=weights, agg='mean')
        V = _combine_multiscale_dask(v_fields, weights=weights, agg='mean')
        return da.map_blocks(
            _st_finalize_block, gpu_arr, U, V,
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            st_output=st_output, azimuth=azimuth, normalize=normalize,
            norm_lo=float(stats[0]), norm_scale=float(stats[1]))

    def get_default_params(self):
        return {
            'st_output': 'coherence', 'azimuth': 315.0,
            'derivative_sigma': 1.0, 'normalize': True,
            'mode': 'spatial', 'radii': None, 'weights': None,
        }


__all__ = [
    "nan_filled", "gaussian_gradients", "st_component_block",
    "compute_structure_tensor_block", "st_stretch_stat_func",
    "StructureTensorAlgorithm",
]
