"""
FujiShaderGPU/algorithms/_impl_scale_drift.py

Scale-Drift Field -- a FujiShaderGPU-original terrain asymmetry measure.

Tracks how features MOVE through Gaussian scale-space (Koenderink 1984 "deep
structure"; Lindeberg 1994): for each pair of adjacent scales, one Lucas-Kanade
step (Lucas & Kanade 1981) estimates the apparent displacement between
L(x; s_i) and L(x; s_{i+1}), normalized by the scale gap.  Symmetric landforms
drift ~0; asymmetric ones (cuestas, tilted blocks, one-sided erosion) drift
systematically toward their gentle side -- a VECTOR version of the scalar
scale_space_surprise.  The measure, name, and formulation are original to this
project (no known prior publication; re-check before any paper).

Per pair (sorted scales s_i < s_{i+1}):

    It  = L_{i+1} - L_i
    (gx, gy) = grad( (L_i + L_{i+1}) / 2 )
    J   = G_w * [gx^2, gx gy; gx gy, gy^2],  b = G_w * (gx It, gy It)
    d_i = -(J + delta I)^-1 b / (s_{i+1} - s_i)      (Tikhonov-damped LK)

and the drift field is the pair-weighted mean D = sum w_i d_i.  Output modes
(--drift-output): ``magnitude`` (default, robust-stretched), ``direction``
(atan2 angle mapped to [0, 1)), ``divergence`` (0.5-centred tanh; sources =
ridges being "pushed apart", sinks = convergent forms).
"""
from __future__ import annotations
import logging

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import Constants, DaskAlgorithm
from ._nan_utils import (
    restore_nan, multiscale_response_fields,
)
from ._impl_structure_tensor import nan_filled

logger = logging.getLogger(__name__)

# Lucas-Kanade window sigma cap: bounds the combine-stage halo (4*cap + margin).
DRIFT_WINDOW_CAP = 24.0
_DEFAULT_SCALES = (2.0, 4.0, 8.0, 16.0, 32.0)


def _sorted_scales_and_pair_weights(scales, weights):
    """Ascending scales + per-adjacent-pair weights (mirrors scale_space_surprise)."""
    vals = []
    wl = None
    if weights is not None and len(list(weights)) == len(list(scales)):
        wl = [float(w) for w in weights]
    for i, s in enumerate(scales):
        try:
            sv = float(s)
        except (TypeError, ValueError):
            continue
        if sv > 0:
            vals.append((sv, wl[i] if wl is not None else None))
    vals.sort(key=lambda t: t[0])
    dedup = []
    for s, w in vals:
        if not dedup or abs(s - dedup[-1][0]) > 1e-9:
            dedup.append((s, w))
    if len(dedup) < 2:
        dedup = [(s, None) for s in _DEFAULT_SCALES]
    sorted_scales = [s for s, _ in dedup]
    pair_w = None
    if all(w is not None for _, w in dedup):
        pw = [0.5 * (dedup[i][1] + dedup[i + 1][1]) for i in range(len(dedup) - 1)]
        tot = float(sum(pw))
        if tot > 1e-12:
            pair_w = [p / tot for p in pw]
    return sorted_scales, pair_w


def _drift_smooth_block(block, *, scale, pixel_size=1.0, pixel_scale_x=None,
                        pixel_scale_y=None, **_ignored):
    """One scale's Gaussian level of the NaN-filled DEM."""
    filled, _ = nan_filled(block)
    return gaussian_filter(filled, sigma=max(0.5, float(scale)),
                           mode='nearest').astype(cp.float32)


def _drift_vector(smooths, scales, pair_w):
    """Pair-weighted mean drift vector (Dx, Dy) from Gaussian levels.

    Memory-lean: each pair's intermediates are freed as soon as they are
    consumed so the block's peak stays near a handful of full-size arrays.
    The naive version held ~15-20 simultaneous full-block arrays and ran a
    20 GB-VRAM GPU out of RMM pool on the global GEBCO combine; freeing eagerly
    (plus the smaller auto-tuned chunk for this algorithm) keeps it in budget.
    """
    shape = smooths[0].shape
    dx_acc = cp.zeros(shape, dtype=cp.float32)
    dy_acc = cp.zeros(shape, dtype=cp.float32)
    n_pairs = len(scales) - 1
    for i in range(n_pairs):
        lo, hi = smooths[i], smooths[i + 1]
        it = hi - lo
        gy, gx = cp.gradient(lo + hi)   # gradient(0.5*(lo+hi)) = 0.5*gradient(lo+hi)
        gx *= cp.float32(0.5)
        gy *= cp.float32(0.5)
        w_sig = min(max(1.5, float(scales[i])), DRIFT_WINDOW_CAP)
        jxx = gaussian_filter(gx * gx, sigma=w_sig, mode='nearest')
        jyy = gaussian_filter(gy * gy, sigma=w_sig, mode='nearest')
        jxy = gaussian_filter(gx * gy, sigma=w_sig, mode='nearest')
        bx = gaussian_filter(gx * it, sigma=w_sig, mode='nearest')
        by = gaussian_filter(gy * it, sigma=w_sig, mode='nearest')
        del gx, gy, it
        # Tikhonov damping keeps flats (rank-deficient J) at drift ~0.
        delta = cp.float32(1e-3) * (jxx + jyy) + cp.float32(1e-12)
        a11 = jxx + delta
        a22 = jyy + delta
        del jxx, jyy, delta
        det = a11 * a22 - jxy * jxy
        inv_det = 1.0 / det
        del det
        gap = cp.float32(max(1e-6, float(scales[i + 1]) - float(scales[i])))
        w = cp.float32(pair_w[i]) if pair_w is not None else cp.float32(1.0 / n_pairs)
        wg = w / gap
        dxi = -(a22 * bx - jxy * by) * inv_det
        dx_acc += wg * dxi
        del dxi, a22
        dyi = -(a11 * by - jxy * bx) * inv_det
        dy_acc += wg * dyi
        del dyi, a11, jxy, bx, by, inv_det
    return dx_acc, dy_acc


def _drift_output(dx, dy, *, drift_output, normalize, norm_lo, norm_scale):
    mode = str(drift_output or 'magnitude').lower()
    if mode == 'direction':
        ang = cp.arctan2(dy, dx)  # image frame
        return ((ang / (2.0 * cp.float32(cp.pi))) % 1.0).astype(cp.float32)
    if mode == 'divergence':
        ddy, _ = cp.gradient(dy)
        _, ddx = cp.gradient(dx)
        div = ddx + ddy
        if normalize and float(norm_scale) > 1e-12:
            return (0.5 + 0.5 * cp.tanh(div / cp.float32(norm_scale))).astype(cp.float32)
        return div.astype(cp.float32)
    mag = cp.sqrt(dx * dx + dy * dy)
    if normalize and float(norm_scale) > 1e-12:
        mag = cp.maximum((mag - cp.float32(norm_lo)) / cp.float32(norm_scale), 0.0)
    return mag.astype(cp.float32)


def _drift_combine_block(block, *smooths, scales, pair_w, drift_output,
                         normalize, norm_lo, norm_scale):
    nan_mask = cp.isnan(block)
    # Large-scale smooths arrive from the coarse-overview path with NaN
    # re-masked over the NoData footprint; refill before gradients/Gaussians
    # or the NaN erodes up to ~4*DRIFT_WINDOW_CAP px into valid pixels
    # (restore_nan below only restores the original footprint).  Same
    # treatment as _vs_combine_block.
    fillv = cp.nanmean(block)
    fillv = cp.where(cp.isfinite(fillv), fillv, cp.float32(0.0))
    sm = [cp.where(cp.isnan(s), fillv, s).astype(cp.float32) for s in smooths]
    dx, dy = _drift_vector(sm, scales, pair_w)
    out = _drift_output(dx, dy, drift_output=drift_output, normalize=normalize,
                        norm_lo=norm_lo, norm_scale=norm_scale)
    return restore_nan(out, nan_mask)


def compute_scale_drift_block(block, *, radii=None, scales=None, weights=None,
                              drift_output='magnitude', normalize=True,
                              global_stats=None, pixel_size=1.0,
                              pixel_scale_x=None, pixel_scale_y=None,
                              **_ignored):
    """Standalone full computation on one CuPy block (reference / stats prepass)."""
    if radii:
        scales = [float(r) for r in radii]
    if not scales:
        scales = list(_DEFAULT_SCALES)
    short = max(8, min(int(block.shape[0]), int(block.shape[1])))
    scales = [min(float(s), short / 8.0) for s in scales]
    sorted_scales, pair_w = _sorted_scales_and_pair_weights(scales, weights)
    filled, nan_mask = nan_filled(block)
    smooths = [gaussian_filter(filled, sigma=max(0.5, s), mode='nearest')
               for s in sorted_scales]
    dx, dy = _drift_vector(smooths, sorted_scales, pair_w)
    if not normalize:
        # Stats prepass: raw magnitude for magnitude/direction runs, raw
        # divergence for divergence runs (the stat is mode-matched).
        out = _drift_output(dx, dy, drift_output=drift_output, normalize=False,
                            norm_lo=0.0, norm_scale=0.0)
        return restore_nan(out, nan_mask)
    lo, scale = 0.0, 0.0
    if isinstance(global_stats, (tuple, list)) and len(global_stats) >= 2:
        lo, scale = float(global_stats[0]), float(global_stats[1])
    out = _drift_output(dx, dy, drift_output=drift_output,
                        normalize=scale > 1e-12, norm_lo=lo, norm_scale=scale)
    return restore_nan(out, nan_mask)


def drift_stretch_stat_func(data):
    """Robust display stats over the raw drift output.

    Magnitude (unsigned): ``(p2, p98 - p2)`` stretch.  Divergence (signed,
    ~0-centred): ``(0, p95(|x|))`` tanh scale.  The signedness of the pooled
    values selects the branch, so one stat function serves both modes.
    """
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    vmin = float(cp.min(valid))
    if vmin < 0.0:  # divergence (signed)
        scale = float(cp.percentile(cp.abs(valid), 95.0))
        return (0.0, scale if scale > 1e-12 else 1e-12)
    lo = float(cp.percentile(valid, 2.0))
    hi = float(cp.percentile(valid, 98.0))
    scale = hi - lo
    if scale <= 1e-12:
        return (0.0, hi if hi > 1e-12 else 1.0)
    return (lo, scale)


class ScaleDriftAlgorithm(DaskAlgorithm):
    """Scale-Drift Field -- original cross-scale feature-drift (asymmetry) map."""

    def process(self, gpu_arr, **params):
        radii = params.get('radii')
        scales = [float(r) for r in radii] if radii else \
            list(params.get('scales') or _DEFAULT_SCALES)
        sorted_scales, pair_w = _sorted_scales_and_pair_weights(
            scales, params.get('weights', None))
        drift_output = str(params.get('drift_output', 'magnitude')).lower()
        normalize = bool(params.get('normalize', True)) and drift_output != 'direction'
        ps = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)

        stats = params.get('global_stats', None)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2
                and float(stats[1]) > 1e-9):
            stats = (0.0, 0.0)
            if normalize:
                logger.info("scale_drift: no global stats; output is the raw "
                            "(unstretched) %s field.", drift_output)
                normalize = False

        smooths = multiscale_response_fields(
            gpu_arr, sorted_scales, block_fn=_drift_smooth_block,
            radius_kw='scale',
            depth_for_scale=lambda s: int(max(1, round(float(s) * 4))) + 1,
            pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
            is_geographic=bool(params.get('is_geographic_dem', False)),
            coarse_dem=params.get('_overview_coarse_dem'),
            coarse_decimation=params.get('_overview_decimation'),
            tile_origin=params.get('_tile_origin'),
            tile_full_shape=params.get('_tile_full_shape'))
        # LK window + divergence gradient need their own halo in the combine.
        depth = min(int(4 * DRIFT_WINDOW_CAP) + 4, Constants.MAX_DEPTH)
        try:
            min_chunk = min(min(ax) for ax in gpu_arr.chunks)
            depth = max(1, min(depth, int(min_chunk) - 1))
        except Exception:
            pass
        return da.map_overlap(
            _drift_combine_block, gpu_arr, *smooths,
            depth=depth, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            scales=sorted_scales, pair_w=pair_w, drift_output=drift_output,
            normalize=normalize, norm_lo=float(stats[0]),
            norm_scale=float(stats[1]))

    def get_default_params(self):
        return {
            'scales': list(_DEFAULT_SCALES), 'drift_output': 'magnitude',
            'normalize': True, 'mode': 'spatial', 'radii': None, 'weights': None,
        }


__all__ = [
    "compute_scale_drift_block", "drift_stretch_stat_func",
    "ScaleDriftAlgorithm", "DRIFT_WINDOW_CAP",
]
