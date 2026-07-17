"""
FujiShaderGPU/algorithms/_impl_frangi.py

Frangi Vesselness for terrain (multi-scale Hessian eigenvalue line filter).

Transfers the vessel-enhancement filter (Frangi et al. 1998, MICCAI; Sato et
al. 1998) to DEMs: per scale sigma, the scale-normalized Hessian
``sigma^2 * H`` (Lindeberg 1998) is eigen-analysed and

    R_B = l1 / l2        (blobness;   |l1| <= |l2|)
    S   = sqrt(l1^2 + l2^2)  (second-order structure energy)
    V   = exp(-R_B^2 / 2 beta^2) * (1 - exp(-S^2 / 2 c^2))

with the sign of ``l2`` selecting ridges (l2 < 0) or valleys (l2 > 0).
Because only eigenvalue *ratios* and a global energy scale ``c`` enter,
low-relief channel/levee/gully networks light up as strongly as mountain
ridges -- unlike plain curvature.  ``c`` comes from the global stats prepass
(robust p95 of S / 2); the raw ``normalize=False`` path returns S for exactly
that purpose.

Output modes (--feature-type): ``ridge`` / ``valley`` in [0, 1];
``both`` (default) = 0.5 + 0.5*(V_ridge - V_valley).
"""
from __future__ import annotations
import logging

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter

from ._base import Constants, DaskAlgorithm
from ._global_stats import estimate_global_stats_or_default
from ._nan_utils import (
    restore_nan, _resolve_spatial_radii_weights, _combine_multiscale_dask,
    large_radius_threshold, multiscale_response_fields,
)
from ._impl_structure_tensor import nan_filled

logger = logging.getLogger(__name__)


def _hessian_eigen(filled: cp.ndarray, sigma: float):
    """Scale-normalized Hessian eigenvalues (l1, l2) with |l1| <= |l2|."""
    s = max(0.8, float(sigma))
    norm = cp.float32(s * s)  # Lindeberg gamma=1 scale normalization
    hxx = gaussian_filter(filled, sigma=s, order=(0, 2), mode='nearest') * norm
    hyy = gaussian_filter(filled, sigma=s, order=(2, 0), mode='nearest') * norm
    hxy = gaussian_filter(filled, sigma=s, order=(1, 1), mode='nearest') * norm
    mean = 0.5 * (hxx + hyy)
    spread = cp.sqrt(0.25 * (hxx - hyy) ** 2 + hxy * hxy)
    e1 = mean + spread
    e2 = mean - spread
    swap = cp.abs(e1) > cp.abs(e2)
    l1 = cp.where(swap, e2, e1)  # smaller magnitude
    l2 = cp.where(swap, e1, e2)  # larger magnitude
    return l1, l2


def _vesselness(l1, l2, *, beta: float, c: float, feature_type: str):
    eps = cp.float32(1e-12)
    rb2 = (l1 / (l2 + cp.where(l2 >= 0, eps, -eps))) ** 2
    s2 = l1 * l1 + l2 * l2
    resp = cp.exp(-rb2 / cp.float32(2.0 * beta * beta)) * (
        1.0 - cp.exp(-s2 / cp.float32(max(2.0 * c * c, 1e-20))))
    ridge = cp.where(l2 < 0, resp, cp.float32(0.0))
    valley = cp.where(l2 > 0, resp, cp.float32(0.0))
    ft = str(feature_type or 'both').lower()
    if ft == 'ridge':
        return ridge.astype(cp.float32)
    if ft == 'valley':
        return valley.astype(cp.float32)
    return (ridge - valley).astype(cp.float32)  # signed, [-1, 1]


def frangi_response_block(block, *, radius, beta=0.5, frangi_c=1.0,
                          feature_type='both', pixel_size=1.0,
                          pixel_scale_x=None, pixel_scale_y=None, **_ignored):
    """One scale's vesselness field (sigma = radius/2, NaN-aware)."""
    filled, nan_mask = nan_filled(block)
    l1, l2 = _hessian_eigen(filled, max(0.8, float(radius) / 2.0))
    out = _vesselness(l1, l2, beta=float(beta), c=float(frangi_c),
                      feature_type=feature_type)
    return restore_nan(out, nan_mask)


def compute_frangi_block(block, *, radii=None, weights=None, beta=0.5,
                         feature_type='both', normalize=True, global_stats=None,
                         pixel_size=1.0, pixel_scale_x=None, pixel_scale_y=None,
                         agg='mean', **_ignored):
    """Standalone full computation on one CuPy block.

    ``normalize=False`` (the stats prepass) returns the multi-scale MAX of the
    structure energy S -- the input for the global ``c`` estimate.  The
    normalized path uses ``c`` from ``global_stats[1]`` (fallback: block p95/2).
    """
    filled, nan_mask = nan_filled(block)
    if not radii:
        radii = [2, 4, 8, 16]
    short = max(8, min(int(block.shape[0]), int(block.shape[1])))
    rs = sorted({max(1, min(int(round(float(r))), short // 4)) for r in radii})

    if not normalize:
        s_max = cp.zeros(block.shape, dtype=cp.float32)
        for r in rs:
            l1, l2 = _hessian_eigen(filled, max(0.8, float(r) / 2.0))
            s_max = cp.maximum(s_max, cp.sqrt(l1 * l1 + l2 * l2))
        return restore_nan(s_max, nan_mask)

    c = None
    if isinstance(global_stats, (tuple, list)) and len(global_stats) >= 2 \
            and float(global_stats[1]) > 1e-12:
        c = float(global_stats[1])
    if c is None:
        pooled = compute_frangi_block(
            block, radii=rs, normalize=False)
        c = frangi_c_stat_func(pooled)[1]

    ws = None
    if weights is not None and len(list(weights)) == len(list(radii)) \
            and len(rs) == len(list(radii)):
        ws = [float(w) for w in weights]
    agg_norm = str(agg or 'mean').lower()
    acc = None
    wsum = 0.0
    for i, r in enumerate(rs):
        resp = frangi_response_block(
            block, radius=float(r), beta=beta, frangi_c=c,
            feature_type=feature_type)
        if agg_norm == 'max':
            acc = resp if acc is None else cp.maximum(acc, resp)
        else:
            w = float(ws[i]) if ws is not None else 1.0
            acc = (cp.float32(w) * resp) if acc is None else acc + cp.float32(w) * resp
            wsum += w
    if agg_norm != 'max' and wsum > 1e-12:
        acc /= cp.float32(wsum)
    out = _display_map(acc, feature_type)
    return restore_nan(out, nan_mask)


def _display_map(combined: cp.ndarray, feature_type: str) -> cp.ndarray:
    """Signed 'both' response -> [0, 1] display; unsigned passes through."""
    if str(feature_type or 'both').lower() == 'both':
        return (0.5 + 0.5 * cp.clip(combined, -1.0, 1.0)).astype(cp.float32)
    return cp.clip(combined, 0.0, 1.0).astype(cp.float32)


def frangi_c_stat_func(data):
    """Global Frangi ``c`` = robust p95 of the structure energy S, halved
    (Frangi 1998 recommends half the maximum Hessian norm; p95 is its robust
    stand-in).  Returned as ``(0.0, c)`` so the generic stats plumbing accepts it."""
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    c = 0.5 * float(cp.percentile(valid, 95.0))
    return (0.0, c if c > 1e-12 else 1.0)


def _frangi_finalize_block(block, combined, *, feature_type):
    nan_mask = cp.isnan(block)
    return restore_nan(_display_map(combined, feature_type), nan_mask)


class FrangiAlgorithm(DaskAlgorithm):
    """Frangi vesselness: multi-scale Hessian ridge/valley network filter."""

    def process(self, gpu_arr, **params):
        beta = float(params.get('beta', 0.5))
        feature_type = str(params.get('feature_type', 'both')).lower()
        if feature_type not in ('ridge', 'valley', 'both'):
            logger.info("frangi: feature_type=%s unsupported; using 'both'.",
                        feature_type)
            feature_type = 'both'
        ps = float(params.get('pixel_size', 1.0))
        psx = params.get('pixel_scale_x', None)
        psy = params.get('pixel_scale_y', None)
        is_geo = bool(params.get('is_geographic_dem', False))

        radii, weights = _resolve_spatial_radii_weights(
            params.get('radii'), params.get('weights', None), ps)
        radii = [float(r) for r in radii]

        stats = params.get('global_stats', None)
        if isinstance(stats, (tuple, list)) and len(stats) >= 2 \
                and float(stats[1]) > 1e-12:
            c = float(stats[1])
        else:
            c = None

        if c is None:
            # Zarr / stats-less fallback: bounded central full-resolution window
            # (not strided full-array reads, and not per-block stats).
            depth = min(int(2 * max(radii)) + 6, Constants.MAX_DEPTH) if radii else 64
            stats = estimate_global_stats_or_default(
                gpu_arr, frangi_c_stat_func, compute_frangi_block,
                {
                    'radii': [max(1, int(r)) for r in radii],
                    'normalize': False,
                    'beta': beta,
                    'feature_type': feature_type,
                    'pixel_size': ps,
                    'pixel_scale_x': psx,
                    'pixel_scale_y': psy,
                },
                depth=depth, algorithm_name='frangi', default=(0.0, 1.0),
            )
            c = float(stats[1])

        thr = large_radius_threshold(gpu_arr, fallback=max(radii) if radii else 64)
        responses = multiscale_response_fields(
            gpu_arr, radii, block_fn=frangi_response_block, radius_kw='radius',
            depth_for_scale=lambda r: int(2 * float(r) + 6),
            is_large=lambda r: int(round(float(r))) > thr,
            pixel_size=ps, pixel_scale_x=psx, pixel_scale_y=psy,
            is_geographic=is_geo,
            coarse_dem=params.get('_overview_coarse_dem'),
            coarse_decimation=params.get('_overview_decimation'),
            tile_origin=params.get('_tile_origin'),
            tile_full_shape=params.get('_tile_full_shape'),
            beta=beta, frangi_c=c, feature_type=feature_type)
        combined = _combine_multiscale_dask(
            responses, weights=weights, agg=params.get('agg', 'mean'))
        return da.map_blocks(
            _frangi_finalize_block, gpu_arr, combined,
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            feature_type=feature_type)

    def get_default_params(self):
        return {
            'beta': 0.5, 'feature_type': 'both',
            'mode': 'spatial', 'radii': None, 'weights': None,
        }


__all__ = [
    "frangi_response_block", "compute_frangi_block", "frangi_c_stat_func",
    "FrangiAlgorithm",
]
