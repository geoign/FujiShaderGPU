"""Numeric sanity tests for the 2026-07 algorithm batch.

structure_tensor / frangi / lic / phase_congruency / tv_decomposition /
scale_drift -- each is validated on a synthetic CuPy DEM against the defining
property claimed in ALGORITHM_CANDIDATES.md, plus a Dask process() smoke test.
Requires a CUDA GPU (like the rest of the suite).
"""
import math

import pytest

cp = pytest.importorskip("cupy")
da = pytest.importorskip("dask.array")


def _finite(a):
    return a[~cp.isnan(a)]


# ---------------------------------------------------------------------------
# Synthetic terrains
# ---------------------------------------------------------------------------
def striped_dem(n=256, angle_deg=30.0, wavelength=16.0):
    """Parallel sinusoidal ridges whose strike is ``angle_deg`` (image frame,
    measured from the +x axis toward +y)."""
    yy, xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                         cp.arange(n, dtype=cp.float32), indexing="ij")
    a = math.radians(angle_deg)
    # Coordinate ACROSS the stripes (perpendicular to the strike).
    across = -xx * math.sin(a) + yy * math.cos(a)
    return (10.0 * cp.sin(2 * math.pi * across / wavelength)).astype(cp.float32)


def ridge_dem(n=192):
    """One horizontal ridge line at row n//2 (gaussian cross-section)."""
    yy = cp.arange(n, dtype=cp.float32)[:, None]
    profile = 20.0 * cp.exp(-((yy - n // 2) ** 2) / (2 * 4.0 ** 2))
    return cp.broadcast_to(profile, (n, n)).copy()


def asymmetric_hill(n=256, steep_left=True):
    """1D-extruded hill: steep on one side, gentle on the other."""
    xx = cp.arange(n, dtype=cp.float32)[None, :]
    c = n // 2
    steep = cp.exp(-((xx - c) ** 2) / (2 * 8.0 ** 2))
    gentle = cp.exp(-((xx - c) ** 2) / (2 * 40.0 ** 2))
    prof = cp.where(xx < c, steep, gentle) if steep_left else \
        cp.where(xx < c, gentle, steep)
    return cp.broadcast_to(100.0 * prof, (n, n)).copy()


# ---------------------------------------------------------------------------
# structure_tensor
# ---------------------------------------------------------------------------
def test_structure_tensor_orientation_recovers_strike():
    from FujiShaderGPU.algorithms._impl_structure_tensor import (
        compute_structure_tensor_block,
    )
    for angle in (0.0, 30.0, 60.0, 120.0):
        dem = striped_dem(angle_deg=angle)
        out = compute_structure_tensor_block(
            dem, radii=[8], st_output="orientation", normalize=False)
        # Interior only (boundary smearing) and modulo 180 degrees.
        core = out[64:-64, 64:-64]
        est = float(cp.median(core)) * 180.0
        diff = min(abs(est - angle), 180.0 - abs(est - angle))
        assert diff < 6.0, f"strike {angle} deg estimated as {est:.1f}"


def test_structure_tensor_coherence_high_on_fabric_low_on_noise():
    from FujiShaderGPU.algorithms._impl_structure_tensor import (
        compute_structure_tensor_block,
    )
    fabric = compute_structure_tensor_block(
        striped_dem(), radii=[8], st_output="coherence", normalize=False)
    rng = cp.random.RandomState(0)
    noise_dem = rng.standard_normal((256, 256)).astype(cp.float32) * 10.0
    noise = compute_structure_tensor_block(
        noise_dem, radii=[8], st_output="coherence", normalize=False)
    assert float(cp.median(_finite(fabric[32:-32, 32:-32]))) > 0.7
    assert float(cp.median(_finite(noise[32:-32, 32:-32]))) < 0.3


def test_structure_tensor_nan_preserved():
    from FujiShaderGPU.algorithms._impl_structure_tensor import (
        compute_structure_tensor_block,
    )
    dem = striped_dem()
    dem[:8, :8] = cp.nan
    out = compute_structure_tensor_block(dem, radii=[4], normalize=False)
    assert bool(cp.isnan(out[:8, :8]).all())
    assert not bool(cp.isnan(out[64:, 64:]).any())


# ---------------------------------------------------------------------------
# frangi
# ---------------------------------------------------------------------------
def test_frangi_ridge_detected_and_polarity():
    from FujiShaderGPU.algorithms._impl_frangi import compute_frangi_block
    dem = ridge_dem()
    out = compute_frangi_block(dem, radii=[4, 8], feature_type="both")
    n = dem.shape[0]
    on_ridge = float(cp.median(out[n // 2, 32:-32]))
    off_ridge = float(cp.median(out[n // 4, 32:-32]))
    assert on_ridge > 0.8          # ridge -> bright (0.5-centred signed map)
    assert abs(off_ridge - 0.5) < 0.1  # flat -> neutral
    inv = compute_frangi_block(-dem, radii=[4, 8], feature_type="both")
    assert float(cp.median(inv[n // 2, 32:-32])) < 0.2  # valley -> dark


def test_frangi_contrast_invariance():
    # Eigenvalue ratios + global c make a faint ridge score like a strong one
    # when each is normalized by its own global stats.
    from FujiShaderGPU.algorithms._impl_frangi import compute_frangi_block
    dem = ridge_dem()
    strong = compute_frangi_block(dem, radii=[4, 8], feature_type="ridge")
    faint = compute_frangi_block(dem * 0.01, radii=[4, 8], feature_type="ridge")
    n = dem.shape[0]
    s = float(cp.median(strong[n // 2, 32:-32]))
    f = float(cp.median(faint[n // 2, 32:-32]))
    assert abs(s - f) < 0.05


# ---------------------------------------------------------------------------
# lic
# ---------------------------------------------------------------------------
def test_lic_output_range_and_determinism():
    from FujiShaderGPU.algorithms._impl_lic import compute_lic_block
    dem = asymmetric_hill()
    out1 = compute_lic_block(dem, length=10, composite="none")
    out2 = compute_lic_block(dem, length=10, composite="none")
    assert bool(cp.allclose(out1, out2))  # elevation-hash noise: deterministic
    vals = _finite(out1)
    assert float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0
    assert float(vals.std()) > 0.01  # actually textured


def test_lic_streamline_correlation_direction():
    # Flow-LIC on a plane tilted along x: texture must be correlated ALONG the
    # flow (x) direction, i.e. x-neighbor diffs are smaller than y-neighbor diffs.
    from FujiShaderGPU.algorithms._impl_lic import compute_lic_block
    n = 256
    xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                     cp.arange(n, dtype=cp.float32), indexing="ij")[1]
    rng = cp.random.RandomState(1)
    dem = xx * 5.0 + rng.standard_normal((n, n)).astype(cp.float32) * 0.5
    out = compute_lic_block(dem, length=12, composite="none")
    core = out[32:-32, 32:-32]
    dx_var = float(cp.mean((core[:, 1:] - core[:, :-1]) ** 2))
    dy_var = float(cp.mean((core[1:, :] - core[:-1, :]) ** 2))
    assert dx_var < dy_var * 0.7


# ---------------------------------------------------------------------------
# phase_congruency
# ---------------------------------------------------------------------------
def test_phase_congruency_amplitude_invariance():
    from FujiShaderGPU.algorithms._impl_phase_congruency import (
        compute_phase_congruency_block,
    )
    n = 256
    xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                     cp.arange(n, dtype=cp.float32), indexing="ij")[1]
    step_big = cp.where(xx > n // 2, 100.0, 0.0).astype(cp.float32)
    step_small = cp.where(xx > n // 2, 0.5, 0.0).astype(cp.float32)
    pc_big = compute_phase_congruency_block(step_big, feature_type="edge")
    pc_small = compute_phase_congruency_block(step_small, feature_type="edge")
    col = n // 2
    big = float(cp.max(pc_big[64:-64, col - 2:col + 3]))
    small = float(cp.max(pc_small[64:-64, col - 2:col + 3]))
    assert big > 0.5 and small > 0.5      # both detected...
    assert abs(big - small) < 0.15        # ...with near-equal strength


def test_phase_congruency_flat_is_quiet():
    from FujiShaderGPU.algorithms._impl_phase_congruency import (
        compute_phase_congruency_block,
    )
    rng = cp.random.RandomState(2)
    dem = rng.standard_normal((256, 256)).astype(cp.float32) * 0.01
    pc = compute_phase_congruency_block(dem, feature_type="edge")
    # Pure noise: the T threshold should suppress most of the field.
    assert float(cp.mean(_finite(pc))) < 0.25


# ---------------------------------------------------------------------------
# tv_decomposition
# ---------------------------------------------------------------------------
def test_tv_step_stays_in_structure_sinusoid_goes_to_texture():
    from FujiShaderGPU.algorithms._impl_tv_decomposition import (
        compute_tv_texture_block,
    )
    n = 256
    xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                     cp.arange(n, dtype=cp.float32), indexing="ij")[1]
    step = cp.where(xx > n // 2, 50.0, 0.0).astype(cp.float32)
    ripple = (2.0 * cp.sin(2 * math.pi * xx / 8.0)).astype(cp.float32)
    v = compute_tv_texture_block(
        step + ripple, tv_scale=24.0, iterations=120, normalize=False)
    core = v[32:-32, 32:-32]
    # Texture carries the ripple amplitude (~2 m)...
    assert 1.0 < float(cp.percentile(cp.abs(core), 90.0)) < 4.0
    # ...but NOT the 50 m step: no large residual anywhere near the scarp.
    col = n // 2
    assert float(cp.max(cp.abs(v[32:-32, col - 8:col + 8]))) < 10.0


def test_tv_structure_component_keeps_step():
    from FujiShaderGPU.algorithms._impl_tv_decomposition import (
        compute_tv_texture_block,
    )
    n = 128
    xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                     cp.arange(n, dtype=cp.float32), indexing="ij")[1]
    step = cp.where(xx > n // 2, 50.0, 0.0).astype(cp.float32)
    u = compute_tv_texture_block(step, tv_scale=16.0, iterations=100,
                                 component="structure")
    left = float(cp.median(u[:, : n // 4]))
    right = float(cp.median(u[:, -n // 4:]))
    assert abs(left - 0.0) < 2.0 and abs(right - 50.0) < 2.0


# ---------------------------------------------------------------------------
# scale_drift
# ---------------------------------------------------------------------------
def test_scale_drift_asymmetric_hill_drifts_and_symmetric_does_not():
    from FujiShaderGPU.algorithms._impl_scale_drift import (
        compute_scale_drift_block,
    )
    asym = compute_scale_drift_block(
        asymmetric_hill(steep_left=True), scales=[2, 4, 8, 16],
        drift_output="magnitude", normalize=False)
    n = asym.shape[0]
    c = n // 2
    on_hill = float(cp.median(asym[c - 8:c + 8, c - 20:c + 20]))

    yy, xx = cp.meshgrid(cp.arange(n, dtype=cp.float32),
                         cp.arange(n, dtype=cp.float32), indexing="ij")
    sym_dem = 100.0 * cp.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 20.0 ** 2))
    sym = compute_scale_drift_block(
        sym_dem.astype(cp.float32), scales=[2, 4, 8, 16],
        drift_output="magnitude", normalize=False)
    at_peak = float(cp.abs(sym[c, c]))
    assert on_hill > 5.0 * max(at_peak, 1e-6)


def test_scale_drift_direction_points_toward_gentle_side():
    from FujiShaderGPU.algorithms._impl_scale_drift import (
        compute_scale_drift_block,
    )
    # steep_left=True -> gentle side is +x -> drift angle ~0 (mod 1).
    out = compute_scale_drift_block(
        asymmetric_hill(steep_left=True), scales=[2, 4, 8, 16],
        drift_output="direction", normalize=False)
    n = out.shape[0]
    c = n // 2
    ang = float(cp.median(out[c - 4:c + 4, c - 4:c + 4]))  # [0, 1)
    dist_to_zero = min(ang, 1.0 - ang)
    assert dist_to_zero < 0.15


# ---------------------------------------------------------------------------
# Dask process() smoke tests (chunked -> also exercises map_overlap paths)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, params", [
    ("structure_tensor", {"radii": [4, 8], "st_output": "coherence"}),
    ("structure_tensor", {"radii": [4, 8], "st_output": "fabric"}),
    ("frangi", {"radii": [4, 8], "feature_type": "both"}),
    ("lic", {"length": 8}),
    ("phase_congruency", {"radii": [4, 8, 16]}),
    ("tv_decomposition", {"tv_scale": 16.0, "iterations": 40}),
    ("scale_drift", {"radii": [2, 4, 8], "drift_output": "magnitude"}),
])
def test_dask_process_smoke(name, params):
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS
    dem = asymmetric_hill(n=256) + striped_dem(n=256, angle_deg=45.0)
    dem[:4, :4] = cp.nan
    arr = da.from_array(dem, chunks=(128, 128), asarray=False)
    algo = ALGORITHMS[name]
    out = algo.process(arr, pixel_size=1.0, mode="spatial", **params).compute()
    assert out.shape == dem.shape
    assert out.dtype == cp.float32
    assert bool(cp.isnan(out[:4, :4]).all())      # NoData preserved
    assert not bool(cp.isnan(out[16:, 16:]).any())  # interior finite
    assert float(cp.abs(_finite(out)).max()) < 1e6


def test_registry_contains_new_algorithms_on_both_backends():
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS
    from FujiShaderGPU.core.tile_processor import DEFAULT_ALGORITHMS
    new = ["structure_tensor", "frangi", "lic", "phase_congruency",
           "tv_decomposition", "scale_drift"]
    for name in new:
        assert name in ALGORITHMS
        assert name in DEFAULT_ALGORITHMS
    assert list(ALGORITHMS.keys()) == list(DEFAULT_ALGORITHMS.keys())
