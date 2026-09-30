"""Regression tests for the 2026-10-01 algorithm review (batch B).

LIC hillshade sign, specular edge padding, scale-space-surprise NoData fill,
fractal_anomaly radii order, single-chunk stats fallback, tiling-independent
nan_filled.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
da = pytest.importorskip("dask.array")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - no GPU
    pytest.skip("CUDA device required", allow_module_level=True)


def _rows(h, w):
    return cp.arange(h, dtype=cp.float32)[:, None] * cp.ones((1, w), cp.float32)


# 1. LIC hillshade composite follows the compute_hillshade_block sign rule ----
def test_lic_hillshade_composite_matches_hillshade_orientation():
    from FujiShaderGPU.algorithms._impl_lic import compute_lic_block
    from FujiShaderGPU.algorithms._impl_hillshade import compute_hillshade_block

    r = _rows(96, 96)
    kw = dict(azimuth=315, altitude=45, pixel_size=1.0,
              pixel_scale_x=1.0, pixel_scale_y=-1.0)
    c = slice(20, -20)
    for z in (1000 + 0.5 * r, 1000 - 0.5 * r):  # north-facing, south-facing
        hs = float(cp.mean(compute_hillshade_block(z, **kw)[c, c]))
        lic_hs = compute_lic_block(z, length=10, composite='hillshade', **kw)
        lic_no = compute_lic_block(z, length=10, composite='none', **kw)
        implied = float(cp.mean(lic_hs[c, c]) / cp.mean(lic_no[c, c]))
        assert implied == pytest.approx(hs, abs=0.02)


# 2. Specular roughness uses reflect padding (no zero-padding blow-up) --------
def test_specular_roughness_edge_not_zero_padded():
    from FujiShaderGPU.algorithms._impl_specular import _roughness_p95_block

    x = cp.arange(128, dtype=cp.float32)[None, :] * cp.ones((128, 1), cp.float32)
    z = 1500.0 + 2.0 * x  # smooth ramp
    rough = _roughness_p95_block(z, 20)
    edge, interior = float(rough[0, 64]), float(rough[64, 64])
    assert interior > 0
    assert edge < 3.0 * interior  # was ~100x with mode='constant'
    # Same as running on a reflect-padded array (what a halo would give).
    zp = cp.pad(z, 20, mode='symmetric')
    ref = _roughness_p95_block(zp, 20)[20:-20, 20:-20]
    assert float(cp.abs(rough - ref).max()) < 1e-3


# 3. Scale-space surprise keeps NoData (0 m) out of every scale ---------------
def test_scale_space_surprise_nan_edge_comparable_to_interior():
    from FujiShaderGPU.algorithms._impl_experimental import (
        compute_scale_space_surprise_block, _sss_smooth_block)

    rng = np.random.default_rng(0)
    z = (3000 + rng.normal(0, 1, (128, 192))).astype(np.float32)
    z[:, 128:] = np.nan
    z = cp.asarray(z)
    raw = compute_scale_space_surprise_block(z, scales=[1, 2, 4, 8, 16], normalize=False)
    interior = float(cp.nanmean(raw[:, 40:80]))
    near_edge = float(cp.nanmean(raw[:, 120:128]))
    assert near_edge < 3.0 * interior  # was ~2000x (0 m mixed into the blur)
    sm = _sss_smooth_block(z, scale=16)
    assert float(cp.abs(sm[:, 100:128] - 3000.0).max()) < 2.0


# 4. fractal_anomaly is independent of the --radii order ----------------------
def test_fractal_anomaly_radii_order_invariant():
    from FujiShaderGPU.algorithms._impl_fractal_anomaly import (
        compute_fractal_dimension_block, FractalAnomalyAlgorithm)

    rng = np.random.default_rng(1)
    z = cp.asarray(np.cumsum(np.cumsum(rng.normal(0, 1, (96, 96)), 0), 1)
                   .astype(np.float32) * 0.05)
    kw = dict(normalize=False, relief_p10=0.1, relief_p75=2.0)
    a = compute_fractal_dimension_block(z, radii=[2, 4, 6, 8, 12],
                                        weights=[1, 2, 3, 4, 5], **kw)
    b = compute_fractal_dimension_block(z, radii=[12, 8, 6, 4, 2],
                                        weights=[5, 4, 3, 2, 1], **kw)
    assert float(cp.abs(a - b).max()) == 0.0

    p = dict(global_stats=(0.0, 0.5), relief_p10=0.1, relief_p75=2.0)
    arr = da.from_array(z, chunks=z.shape, asarray=False)
    o1 = FractalAnomalyAlgorithm().process(arr, radii=[2, 4, 6, 8, 12], **p).compute()
    o2 = FractalAnomalyAlgorithm().process(arr, radii=[12, 8, 6, 4, 2], **p).compute()
    assert float(cp.abs(o1 - o2).max()) == 0.0


# 5. Single chunk without global_stats still normalizes -----------------------
def test_single_chunk_without_stats_normalizes():
    from FujiShaderGPU.algorithms._impl_fractal_anomaly import FractalAnomalyAlgorithm
    from FujiShaderGPU.algorithms._impl_visual_saliency import VisualSaliencyAlgorithm

    rng = np.random.default_rng(1)
    z = cp.asarray(np.cumsum(np.cumsum(rng.normal(0, 1, (128, 128)), 0), 1)
                   .astype(np.float32) * 0.05)
    arr = da.from_array(z, chunks=z.shape, asarray=False)
    fa = FractalAnomalyAlgorithm().process(arr, radii=[2, 4, 6, 8, 12]).compute()
    assert abs(float(cp.median(fa))) < 0.2  # was ~ -0.8 with fixed (0, 0.5)
    vs = VisualSaliencyAlgorithm().process(arr, radii=[2, 4, 8, 16]).compute()
    assert float(cp.percentile(vs, 99)) == pytest.approx(1.0, abs=0.2)


# 6. nan_filled is local, so tilings agree near NoData ------------------------
def test_nan_filled_is_local_not_block_mean():
    from FujiShaderGPU.algorithms._impl_structure_tensor import nan_filled

    x = cp.arange(160, dtype=cp.float32)[None, :] * cp.ones((64, 1), cp.float32)
    z = 4.0 * x
    z[48:, :] = cp.nan
    f, mask = nan_filled(z)
    assert bool(cp.isfinite(f).all())
    assert bool((f[~mask] == z[~mask]).all())
    # Just below the coast the fill continues the local ramp, not the block mean.
    assert float(cp.abs(f[49, 20:140] - z[47, 20:140]).max()) < 5.0


def test_frangi_one_vs_two_chunks_agree_near_nodata():
    from FujiShaderGPU.algorithms._impl_frangi import FrangiAlgorithm

    rng = np.random.default_rng(0)
    h, w = 128, 256
    x = np.mgrid[0:h, 0:w][1].astype(np.float32)
    z = (x * 4.0 + rng.normal(0, 0.5, (h, w))).astype(np.float32)
    z[100:, :] = np.nan
    z = cp.asarray(z)
    p = dict(radii=[4, 8], global_stats=(0.0, 0.5))
    one = FrangiAlgorithm().process(da.from_array(z, chunks=(h, w), asarray=False), **p).compute()
    two = FrangiAlgorithm().process(da.from_array(z, chunks=(h, 128), asarray=False), **p).compute()
    d = cp.abs(one - two)
    assert float(cp.nanmax(d)) < 1e-3  # was ~1.0 with the per-block nanmean fill
