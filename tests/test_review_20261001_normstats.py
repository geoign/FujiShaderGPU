"""Review 2026-10-01 regressions for the global normalization pre-pass.

1. ambient_occlusion / openness in ``--mode spatial``: the [p1, p99] display
   stretch must be estimated from the weighted multi-radius field the main pass
   stretches, not from the single-radius local block (output filled only
   ~[0.3, 0.9] instead of [0, 1]).
2. npr_edges in ``--mode local``: the edge threshold must come from a global
   gradient distribution, so the output does not depend on the chunk/tile
   layout.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
rasterio = pytest.importorskip("rasterio")
da = pytest.importorskip("dask.array")

from rasterio.transform import from_origin  # noqa: E402


def _write_dem(path, z, px=1.0):
    h, w = z.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
        crs="EPSG:32654", transform=from_origin(0, h * px, px, px),
    ) as dst:
        dst.write(z.astype(np.float32), 1)
    return str(path)


def _smooth(a, n=4):
    for _ in range(n):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


@pytest.fixture(scope="module")
def rough_dem(tmp_path_factory):
    rng = np.random.default_rng(3)
    n = 384
    z = _smooth(rng.standard_normal((n, n)), 3) * 30.0 + _smooth(rng.standard_normal((n, n)), 40) * 300.0
    return _write_dem(tmp_path_factory.mktemp("ns") / "dem.tif", z), z.astype(np.float32)


@pytest.mark.parametrize("algorithm,extra", [
    ("ambient_occlusion", dict(radius=10.0, num_samples=8)),
    ("openness", dict(max_distance=20, num_directions=8)),
])
@pytest.mark.parametrize("mode", ["spatial", "local"])
def test_ao_openness_prepass_matches_main_pass_field(rough_dem, algorithm, extra, mode):
    from FujiShaderGPU.algorithms._norm_stats import _compute_norm_stats_tiled
    from FujiShaderGPU.algorithms._global_stats import robust_unsigned_stretch_stat_func
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS

    path, z = rough_dem
    params = dict(mode=mode, radii=[2, 8, 32] if mode == "spatial" else [1],
                  weights=[0.6, 0.3, 0.1] if mode == "spatial" else [1.0],
                  pixel_size=1.0, pixel_scale_x=1.0, pixel_scale_y=-1.0, **extra)
    stats = _compute_norm_stats_tiled(path, algorithm, params)
    assert stats is not None

    raw = ALGORITHMS[algorithm].process(
        da.from_array(cp.asarray(z), chunks=192, asarray=False), **params).compute()
    m = 64
    actual = robust_unsigned_stretch_stat_func(raw[m:-m, m:-m])
    assert abs(float(stats[0]) - float(actual[0])) < 0.02
    assert abs(float(stats[1]) - float(actual[1])) < 0.02 * float(actual[1]) + 0.01


def test_npr_local_threshold_is_tiling_independent(tmp_path):
    from FujiShaderGPU.algorithms._impl_npr_edges import NPREdgesAlgorithm
    from FujiShaderGPU.algorithms._norm_stats import inject_global_stats
    from FujiShaderGPU.algorithms.tile.dask_bridge import _direct_npr_edges, _merged_params

    rng = np.random.default_rng(0)
    h = w = 256
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 100 * np.exp(-((x - 64) ** 2 + (y - 128) ** 2) / (2 * 30 ** 2)) + rng.normal(0, 0.3, (h, w))
    # Right half: gentle relief, so per-half gradient percentiles differ.
    z[:, 128:] = 5 * np.sin(x[:, 128:] / 15.0) + 5 * np.sin(y[:, 128:] / 20.0) + rng.normal(0, 0.3, (h, 128))
    z = z.astype(np.float32)
    path = _write_dem(tmp_path / "npr.tif", z, px=10.0)

    alg = NPREdgesAlgorithm()
    params = dict(pixel_size=10.0, mode="local", radii=[1], weights=[1.0],
                  pixel_scale_x=10.0, pixel_scale_y=-10.0)
    inject_global_stats(path, "npr_edges", params)
    assert params.get("_npr_grad_stats"), "local mode must get a global gradient threshold"

    zc = cp.asarray(z)
    one = alg.process(da.from_array(zc, chunks=(h, w), asarray=False), **params).compute()
    two = alg.process(da.from_array(zc, chunks=(h, w // 2), asarray=False), **params).compute()
    assert float(cp.abs(one - two).max()) < 1e-6

    # Tile backend direct path: two halo'd tiles must reproduce the single tile.
    mp = _merged_params(alg, params)
    halo = 16
    full = _direct_npr_edges(zc, mp)
    left = _direct_npr_edges(zc[:, :w // 2 + halo], mp)[:, :w // 2]
    right = _direct_npr_edges(zc[:, w // 2 - halo:], mp)[:, halo:]
    assert float(cp.abs(full - cp.concatenate([left, right], axis=1)).max()) < 1e-6
    assert float(cp.abs(full - one)[8:-8, 8:-8].max()) < 1e-6
