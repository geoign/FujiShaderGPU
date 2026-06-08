"""
FujiShaderGPU/algorithms/_pyramid_fill.py

Backend-neutral push-pull (multigrid) void fill.

A single implementation shared by the GPU pipeline (``_nan_utils`` calls it with
CuPy) and the CPU preprocessing fill (``io.dem_preprocess`` calls it with NumPy
when no GPU is present).  The array module ``xp`` and a ``zoom`` function with the
``scipy.ndimage.zoom`` signature are injected so this module imports neither NumPy
nor CuPy at the top level and stays usable on either backend.

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


def pushpull_fill(coarse, valid, *, xp, zoom):
    """Membrane-like void fill via a push-pull image pyramid.

    ``coarse`` : float32 grid (values at invalid cells are ignored).
    ``valid``  : bool mask of finite/known cells (same shape).
    ``xp``     : array module (``numpy`` or ``cupy``).
    ``zoom``   : ``scipy.ndimage.zoom`` / ``cupyx.scipy.ndimage.zoom``.

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
    # Level 0: value*weight and weight, so a plain zoom averages finite cells only.
    w = valid.astype(f32)
    vw = xp.where(valid, out, f32(0.0)).astype(f32)

    vws = [vw]
    ws = [w]
    # ---- push: coarsen x2 (valid-weighted) until support is full or 1x1 ----
    # Halve until every cell has support (ws.min() > 0) or the grid collapses to a
    # single cell.  Gating on max(shape) (not min) keeps collapsing the longer axis
    # after the short one hits 1px, so a wide void in a high-aspect-ratio grid still
    # reaches full support at the 1x1 apex instead of being left unfilled.
    while max(vws[-1].shape[:2]) > 1 and float(ws[-1].min()) <= 1e-6:
        ch, cw = vws[-1].shape[:2]
        nh, nw = max(1, ch // 2), max(1, cw // 2)
        zy, zx = nh / float(ch), nw / float(cw)
        num = zoom(vws[-1], zoom=(zy, zx), order=1, mode="nearest")
        den = zoom(ws[-1], zoom=(zy, zx), order=1, mode="nearest")
        wv = xp.minimum(den, f32(1.0))
        # carry value*weight forward so the next level's zoom keeps averaging finite
        # contributors only (num/den is this level's valid-weighted mean).
        mean = xp.where(den > eps, num / xp.maximum(den, eps), f32(0.0))
        vws.append((mean * wv).astype(f32))
        ws.append(wv.astype(f32))

    # ---- pull: synthesise from coarsest up, fill only unsupported cells ----
    filled = xp.where(ws[-1] > eps, vws[-1] / xp.maximum(ws[-1], eps),
                      f32(0.0)).astype(f32)
    for lvl in range(len(vws) - 2, -1, -1):
        th, tw = vws[lvl].shape[:2]
        fh, fw = filled.shape[:2]
        up = zoom(filled, zoom=(th / float(fh), tw / float(fw)),
                  order=1, mode="nearest")[:th, :tw]
        wl = ws[lvl]
        vl = xp.where(wl > eps, vws[lvl] / xp.maximum(wl, eps), f32(0.0))
        filled = xp.where(wl > eps, vl, up).astype(f32)

    # Preserve the original known cells exactly (push-pull only invents voids).
    return xp.where(valid, out, filled).astype(f32)


__all__ = ["pushpull_fill"]
