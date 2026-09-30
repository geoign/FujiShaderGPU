"""
FujiShaderGPU/algorithms/_pyramid_fill.py

Backend-neutral push-pull (multigrid) void fill.

A single implementation shared by the GPU pipeline (``_nan_utils`` calls it with
CuPy) and the CPU preprocessing fill (``io.dem_preprocess`` calls it with NumPy
when no GPU is present).  The array module ``xp`` is injected so this module
imports neither NumPy nor CuPy at the top level and stays usable on either backend
(resampling uses only ``xp`` array ops: 2x2 box sums down, cell-centred bilinear up).

Why push-pull: it solves a membrane-like, minimal-curvature interpolation over the
voids.  That is the lowest-frequency surface consistent with the surrounding
terrain, which is exactly what avoids inventing relief inside a void -- the failure
mode of the old "nearest valid value + a single Gaussian" fill (flat Voronoi
plateaus from the nearest fallback, plus the coarse grid's own undulations smeared
into featureless voids).  Small voids are filled from fine pyramid levels and large
voids from coarse levels, so the ``2, 8, 32, ...`` radii progression is implicit in
the x2 levels.
"""
from __future__ import annotations


def _box_coarsen(a, fy, fx, xp):
    """Sum ``a`` over non-overlapping ``fy x fx`` blocks (``fy``/``fx`` in {1, 2}).

    An odd axis is zero-padded first; callers pad value*weight and weight alike,
    so the padding carries zero weight and never biases the valid-weighted mean.
    """
    h, w = a.shape[:2]
    ph, pw = (-h) % fy, (-w) % fx
    if ph or pw:
        a = xp.pad(a, ((0, ph), (0, pw)))
    nh, nw = (h + ph) // fy, (w + pw) // fx
    return a.reshape(nh, fy, nw, fx).sum(axis=(1, 3))


def _upsample_centered(a, th, tw, fy, fx, xp):
    """Bilinear, cell-centre-aligned upsample by ``(fy, fx)`` cropped to ``(th, tw)``.

    Consistent with ``_box_coarsen``: coarse cell ``j`` covers fine cells
    ``fy*j .. fy*j+fy-1``, so fine index ``i`` samples coarse coordinate
    ``(i + 0.5) / f - 0.5`` (edge-clamped, i.e. ``mode="nearest"``).
    """
    f32 = xp.float32
    for axis, (n_out, f) in enumerate(((th, fy), (tw, fx))):
        n_in = a.shape[axis]
        if f == 1 or n_in == 1:
            idx = xp.minimum(xp.arange(n_out) // f, n_in - 1)
            a = xp.take(a, idx, axis=axis)
            continue
        c = (xp.arange(n_out, dtype=f32) + f32(0.5)) / f32(f) - f32(0.5)
        c = xp.clip(c, f32(0.0), f32(n_in - 1))
        i0 = xp.floor(c).astype(xp.int64)
        i1 = xp.minimum(i0 + 1, n_in - 1)
        t = (c - i0.astype(f32)).astype(f32)
        shape = (-1, 1) if axis == 0 else (1, -1)
        t = t.reshape(shape)
        a = (xp.take(a, i0, axis=axis) * (f32(1.0) - t)
             + xp.take(a, i1, axis=axis) * t)
    return a.astype(f32)


def pushpull_fill(coarse, valid, *, xp, zoom=None):
    """Membrane-like void fill via a push-pull image pyramid.

    ``coarse`` : float32 grid (values at invalid cells are ignored).
    ``valid``  : bool mask of finite/known cells (same shape).
    ``xp``     : array module (``numpy`` or ``cupy``).
    ``zoom``   : unused; kept so existing callers passing ``scipy.ndimage.zoom`` /
                 ``cupyx.scipy.ndimage.zoom`` keep working.  Resampling is done
                 with explicit 2x2 box sums and cell-centred bilinear upsampling:
                 a corner-aligned ``zoom(order=1)`` made row/col 0 of every
                 coarser level sample only row/col 0 of the finer one, so a void
                 corner cell starved the 1x1 apex (weight 0 -> 0.0) and the pull
                 step ramped corner voids toward 0 m.

    Returns a fully-finite float32 surface; ``valid`` cells are preserved exactly.
    """
    f32 = xp.float32
    out = coarse.astype(f32, copy=True)
    if bool(valid.all()):
        return out
    if not bool(valid.any()):
        # No reference data at all; nothing meaningful to fill with.
        return xp.zeros_like(out, dtype=f32)

    eps = f32(1e-6)
    # Level 0: value*weight and weight, so a block sum averages finite cells only.
    w = valid.astype(f32)
    vw = xp.where(valid, out, f32(0.0)).astype(f32)

    vws = [vw]
    ws = [w]
    factors = []
    # ---- push: coarsen x2 (valid-weighted 2x2 box) until support is full or 1x1 ----
    # Halve until every cell has support (ws.min() > 0) or the grid collapses to a
    # single cell.  Gating on max(shape) (not min) keeps collapsing the longer axis
    # after the short one hits 1px, so a wide void in a high-aspect-ratio grid still
    # reaches full support at the 1x1 apex instead of being left unfilled.
    while max(vws[-1].shape[:2]) > 1 and float(ws[-1].min()) <= 1e-6:
        ch, cw = vws[-1].shape[:2]
        fy, fx = (2 if ch > 1 else 1), (2 if cw > 1 else 1)
        num = _box_coarsen(vws[-1], fy, fx, xp)
        den = _box_coarsen(ws[-1], fy, fx, xp)
        wv = xp.minimum(den, f32(1.0))
        # carry value*weight forward so the next level keeps averaging finite
        # contributors only (num/den is this level's valid-weighted mean).
        mean = xp.where(den > eps, num / xp.maximum(den, eps), f32(0.0))
        vws.append((mean * wv).astype(f32))
        ws.append(wv.astype(f32))
        factors.append((fy, fx))

    # ---- pull: synthesise from coarsest up, fill only unsupported cells ----
    # The box-sum apex covers the whole grid, so it always has support here; the
    # global valid mean is a belt-and-braces fallback so no void can pull 0.0.
    global_mean = f32(float(vw.sum()) / max(float(w.sum()), 1.0))
    filled = xp.where(ws[-1] > eps, vws[-1] / xp.maximum(ws[-1], eps),
                      global_mean).astype(f32)
    for lvl in range(len(vws) - 2, -1, -1):
        th, tw = vws[lvl].shape[:2]
        fy, fx = factors[lvl]
        up = _upsample_centered(filled, th, tw, fy, fx, xp)
        wl = ws[lvl]
        vl = xp.where(wl > eps, vws[lvl] / xp.maximum(wl, eps), f32(0.0))
        filled = xp.where(wl > eps, vl, up).astype(f32)

    # Preserve the original known cells exactly (push-pull only invents voids).
    return xp.where(valid, out, filled).astype(f32)


__all__ = ["pushpull_fill"]
