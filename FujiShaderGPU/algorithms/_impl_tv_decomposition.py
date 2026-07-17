"""
FujiShaderGPU/algorithms/_impl_tv_decomposition.py

Total Variation structure-texture decomposition (ROF / TV-L1).

Solves ``min_u TV(u) + lambda * ||u - z||`` with the Chambolle-Pock primal-dual
scheme (2011): ``u`` is the cartoon/structure surface (edges preserved), and
``v = z - u`` the fine texture.  Unlike Gaussian unsharp masking (TopoUSM), a
cliff stays entirely in ``u`` -- the texture band shows micro-relief with NO
halo/ghost around scarps.

TV-L1 (Chan & Esedoglu 2005, default) has a contrast-independent scale law:
a disc of radius R vanishes from ``u`` iff ``lambda < 2 / R``.  ``--tv-scale``
is that feature DIAMETER s (px): ``lambda = 4 / s``.  For L2 (ROF) the same
mapping is kept as a heuristic (scale selection then also depends on contrast).

Information propagates ~1 px per iteration, so a halo of ``iterations`` pixels
makes the tile output exactly chunk-independent; iterations are capped so the
halo stays inside Constants.MAX_DEPTH.

Output (--tv-component): ``texture`` (default; 0.5-centred tanh display using
the global p90(|v|) scale) or ``structure`` (raw smoothed elevation, like blur).
"""
from __future__ import annotations
import logging

import cupy as cp

from ._base import Constants, DaskAlgorithm
from ._global_stats import estimate_global_stats_or_default
from ._nan_utils import restore_nan
from ._impl_structure_tensor import nan_filled

logger = logging.getLogger(__name__)

TV_MAX_ITERATIONS = 140


def _grad(u):
    """Forward differences with Neumann boundary (last row/col zero)."""
    gx = cp.zeros_like(u)
    gy = cp.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div(px, py):
    """Backward-difference divergence (adjoint of -_grad)."""
    out = cp.zeros_like(px)
    out[:, 0] += px[:, 0]
    out[:, 1:] += px[:, 1:] - px[:, :-1]
    out[:, -1] -= px[:, -1]
    out[0, :] += py[0, :]
    out[1:, :] += py[1:, :] - py[:-1, :]
    out[-1, :] -= py[-1, :]
    return out


def _tv_structure(f: cp.ndarray, *, lam: float, iterations: int,
                  fidelity: str) -> cp.ndarray:
    """Chambolle-Pock primal-dual TV solve; returns the structure part u."""
    tau = cp.float32(0.25)
    sigma = cp.float32(0.5)  # tau * sigma * ||grad||^2 (=8) = 1
    lam32 = cp.float32(lam)
    u = f.copy()
    ubar = f.copy()
    px = cp.zeros_like(f)
    py = cp.zeros_like(f)
    is_l1 = str(fidelity or 'l1').lower() != 'l2'
    for _ in range(int(iterations)):
        gx, gy = _grad(ubar)
        px = px + sigma * gx
        py = py + sigma * gy
        norm = cp.maximum(1.0, cp.sqrt(px * px + py * py))
        px = px / norm
        py = py / norm
        u_old = u
        v = u + tau * _div(px, py)
        if is_l1:
            d = v - f
            t = tau * lam32
            u = f + cp.sign(d) * cp.maximum(cp.abs(d) - t, 0.0)
        else:
            u = (v + tau * lam32 * f) / (1.0 + tau * lam32)
        ubar = 2.0 * u - u_old
    return u.astype(cp.float32)


def compute_tv_texture_block(block, *, tv_scale=32.0, iterations=120,
                             fidelity='l1', component='texture',
                             normalize=True, global_stats=None,
                             radii=None, pixel_size=1.0,
                             pixel_scale_x=None, pixel_scale_y=None,
                             **_ignored):
    """TV decomposition on one CuPy block (NaN-aware).

    ``normalize=False`` (stats prepass) returns the raw signed texture ``v``.
    """
    filled, nan_mask = nan_filled(block)
    s = float(tv_scale)
    if radii:  # unified --radii: first value sets the structure scale
        try:
            s = float(list(radii)[0])
        except (TypeError, ValueError, IndexError):
            pass
    s = max(2.0, s)
    lam = 4.0 / s
    iters = int(max(10, min(int(iterations), TV_MAX_ITERATIONS)))
    u = _tv_structure(filled, lam=lam, iterations=iters, fidelity=fidelity)

    if str(component or 'texture').lower() == 'structure':
        return restore_nan(u, nan_mask)

    v = (filled - u).astype(cp.float32)
    if not normalize:
        return restore_nan(v, nan_mask)
    scale = None
    if isinstance(global_stats, (tuple, list)) and len(global_stats) >= 2 \
            and float(global_stats[1]) > 1e-12:
        scale = float(global_stats[1])
    if scale is None:
        vals = cp.abs(v[~nan_mask])
        scale = float(cp.percentile(vals, 90.0)) if vals.size else 0.0
    if scale <= 1e-12:
        out = cp.full(block.shape, 0.5, dtype=cp.float32)
    else:
        out = 0.5 + 0.5 * cp.tanh(v / cp.float32(scale))
    return restore_nan(out.astype(cp.float32), nan_mask)


def tv_texture_stat_func(data):
    """Global display scale: robust p90 of |texture|."""
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    scale = float(cp.percentile(cp.abs(valid), 90.0))
    return (0.0, scale if scale > 1e-12 else 1e-12)


class TVDecompositionAlgorithm(DaskAlgorithm):
    """Edge-preserving structure-texture split (no halo around scarps)."""

    def process(self, gpu_arr, **params):
        iters = int(max(10, min(int(params.get('iterations', 120)),
                                TV_MAX_ITERATIONS)))
        if int(params.get('iterations', 120)) > TV_MAX_ITERATIONS:
            logger.info("tv_decomposition: iterations clamped to %d "
                        "(halo budget).", TV_MAX_ITERATIONS)
        depth = min(iters + 2, Constants.MAX_DEPTH)
        try:
            min_chunk = min(min(ax) for ax in gpu_arr.chunks)
            depth = max(1, min(depth, int(min_chunk) - 1))
        except Exception:
            pass
        stats = params.get('global_stats', None)
        component = params.get('component', 'texture')
        if (str(component or 'texture').lower() != 'structure'
                and not (isinstance(stats, (tuple, list)) and len(stats) >= 2
                         and float(stats[1]) > 1e-12)):
            stats = estimate_global_stats_or_default(
                gpu_arr, tv_texture_stat_func, compute_tv_texture_block,
                {
                    'tv_scale': float(params.get('tv_scale', 32.0)),
                    'iterations': iters,
                    'fidelity': params.get('fidelity', 'l1'),
                    'component': component,
                    'normalize': False,
                    'radii': params.get('radii', None),
                    'pixel_size': float(params.get('pixel_size', 1.0)),
                    'pixel_scale_x': params.get('pixel_scale_x', None),
                    'pixel_scale_y': params.get('pixel_scale_y', None),
                },
                depth=depth, algorithm_name='tv_decomposition', default=(0.0, 1.0),
            )
        return gpu_arr.map_overlap(
            compute_tv_texture_block, depth=depth, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            tv_scale=float(params.get('tv_scale', 32.0)),
            iterations=iters,
            fidelity=params.get('fidelity', 'l1'),
            component=component,
            normalize=True, global_stats=stats,
            radii=params.get('radii', None),
            pixel_size=float(params.get('pixel_size', 1.0)),
            pixel_scale_x=params.get('pixel_scale_x', None),
            pixel_scale_y=params.get('pixel_scale_y', None))

    def get_default_params(self):
        return {
            'tv_scale': 32.0, 'iterations': 120, 'fidelity': 'l1',
            'component': 'texture',
        }


__all__ = [
    "compute_tv_texture_block", "tv_texture_stat_func",
    "TVDecompositionAlgorithm", "TV_MAX_ITERATIONS",
]
