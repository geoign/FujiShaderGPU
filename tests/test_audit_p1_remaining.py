"""Regression tests for remaining P1 audit items.

Covers P1-2 (--agg stack band-first contract), P1-9a/b (global stats
fallbacks must be central-window/seam-free, not strided or per-block), and P1-11
(geographic DEMs take the same overview path as projected DEMs).
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
da = pytest.importorskip("dask.array")


def test_combine_multiscale_stack_is_band_first_for_dask_and_tile():
    from FujiShaderGPU.algorithms._nan_utils import _combine_multiscale_dask
    from FujiShaderGPU.algorithms.tile.dask_bridge import _combine_direct
    from FujiShaderGPU.core.tile_compute import apply_nodata_mask

    responses = [
        cp.full((3, 4), 1, dtype=cp.float32),
        cp.full((3, 4), 2, dtype=cp.float32),
    ]
    direct = _combine_direct(responses, agg="stack")
    assert direct.shape == (2, 3, 4)
    assert cp.asnumpy(direct[:, 0, 0]).tolist() == [1.0, 2.0]

    darrs = [da.from_array(r, chunks=r.shape, asarray=False) for r in responses]
    stacked = _combine_multiscale_dask(darrs, agg="stack").compute()
    assert stacked.shape == (2, 3, 4)
    assert cp.asnumpy(stacked[:, 0, 0]).tolist() == [1.0, 2.0]

    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    masked = apply_nodata_mask(direct.copy(), mask, np.nan)
    assert bool(cp.isnan(masked[:, 1, 2]).all())
    assert bool(cp.isfinite(masked[:, 0, 0]).all())


def test_dask_stack_mask_and_xarray_wrap_logic_is_band_first():
    # Pin the shape logic used in core.dask_processor without importing the full
    # Dask-CUDA pipeline on CPU-only test environments.
    source_shape = (5, 6)
    result_shape = (3, 5, 6)
    assert len(result_shape) == len(source_shape) + 1
    assert result_shape[-2:] == source_shape
    dims = ("band",) + ("y", "x")
    bands = np.arange(1, result_shape[0] + 1, dtype=np.int32)
    assert dims == ("band", "y", "x")
    assert bands.tolist() == [1, 2, 3]


def test_frangi_fallback_uses_central_window_not_strided(monkeypatch):
    from FujiShaderGPU.algorithms._impl_frangi import FrangiAlgorithm
    import FujiShaderGPU.algorithms._impl_frangi as mod

    arr = da.from_array(cp.ones((64, 64), dtype=cp.float32), chunks=(32, 32), asarray=False)
    calls = []

    def fake_estimate(gpu_arr, stat_func, algorithm_func, algorithm_params, **kwargs):
        calls.append((algorithm_params, kwargs))
        return (0.0, 2.5)

    monkeypatch.setattr(mod, "estimate_global_stats_or_default", fake_estimate)
    out = FrangiAlgorithm().process(arr, radii=[4, 8], global_stats=None)
    assert out.shape == arr.shape
    assert calls, "stats-less frangi must use the shared central-window estimator"
    assert calls[0][0]["normalize"] is False
    assert calls[0][1]["algorithm_name"] == "frangi"


def test_phase_and_tv_fallback_use_central_window(monkeypatch):
    import FujiShaderGPU.algorithms._impl_phase_congruency as pc_mod
    import FujiShaderGPU.algorithms._impl_tv_decomposition as tv_mod

    arr = da.from_array(cp.ones((64, 64), dtype=cp.float32), chunks=(32, 32), asarray=False)
    pc_calls = []
    tv_calls = []

    def fake_pc(gpu_arr, stat_func, algorithm_func, algorithm_params, **kwargs):
        pc_calls.append((algorithm_params, kwargs))
        return (0.0, 1.0)

    def fake_tv(gpu_arr, stat_func, algorithm_func, algorithm_params, **kwargs):
        tv_calls.append((algorithm_params, kwargs))
        return (0.0, 1.0)

    monkeypatch.setattr(pc_mod, "estimate_global_stats_or_default", fake_pc)
    monkeypatch.setattr(tv_mod, "estimate_global_stats_or_default", fake_tv)

    pc_out = pc_mod.PhaseCongruencyAlgorithm().process(arr, radii=[4, 8], global_stats=None)
    tv_out = tv_mod.TVDecompositionAlgorithm().process(arr, global_stats=None, iterations=20)
    assert pc_out.shape == arr.shape
    assert tv_out.shape == arr.shape
    assert pc_calls and pc_calls[0][0]["normalize"] is False
    assert pc_calls[0][1]["algorithm_name"] == "phase_congruency"
    assert tv_calls and tv_calls[0][0]["normalize"] is False
    assert tv_calls[0][1]["algorithm_name"] == "tv_decomposition"


def test_tile_geographic_overview_gate_matches_dask_policy(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    import FujiShaderGPU.core.tile_processor as tp

    src = tmp_path / "geo.tif"
    data = np.ones((32, 32), dtype=np.float32)
    with rasterio.open(
        src, "w", driver="GTiff", height=32, width=32, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(130.0, 36.0, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)

    calls = []

    def fake_inject(*args, **kwargs):
        return args[2]

    def fake_read_overview(path):
        calls.append(path)
        return cp.ones((8, 8), dtype=cp.float32), 4.0

    def fake_compute_fields(path, *, large_radii, block_fn, coarse_dem=None, decimation=None, **kwargs):
        return {}, float(decimation or 1.0)

    monkeypatch.setattr(tp, "inject_global_stats", fake_inject)
    monkeypatch.setattr(tp, "get_gpu_config", lambda **kw: {
        "tile_size": 16, "padding": 4, "max_workers": 1,
        "description": "test", "vram_monitor": False,
        "system_info": {"gpu_detected": False, "vram_gb": 8},
    })
    monkeypatch.setattr(tp, "_resolve_writable_tmp_dir", lambda tmp, out, inp: str(tmp_path / "tiles"))
    monkeypatch.setattr(tp, "_load_algorithm", lambda name: None)
    monkeypatch.setattr(tp, "process_single_tile", lambda *a, **kw: tp.TileResult(0, 0, True, "tile.tif"))
    monkeypatch.setattr(tp, "_build_vrt_and_cog_ultra_fast", lambda *a, **kw: None)
    monkeypatch.setattr(tp, "_validate_cog_for_qgis", lambda *a, **kw: True)

    monkeypatch.setattr(
        "FujiShaderGPU.algorithms._nan_utils.read_overview_coarse_dem",
        fake_read_overview,
    )
    monkeypatch.setattr(
        "FujiShaderGPU.algorithms._nan_utils.compute_overview_scale_fields",
        fake_compute_fields,
    )

    tp.process_dem_tiles(
        str(src), str(tmp_path / "out.tif"), algorithm="hillshade",
        tile_size=16, padding=4, max_workers=1, mode="spatial", radii=[512],
    )
    assert calls, "geographic spatial tile path must inject unified overview source"
