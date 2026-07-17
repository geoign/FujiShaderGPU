"""Regression coverage for findings from the post-P5 audit."""
import importlib.util
from pathlib import Path
import platform
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_linux_core_api_imports_cluster_factory_from_its_owner(monkeypatch):
    package = "_audit_fujishader_core"
    core_dir = REPO_ROOT / "FujiShaderGPU" / "core"
    run_pipeline = object()
    make_cluster = object()

    gpu_memory = types.ModuleType(f"{package}.gpu_memory")
    gpu_memory.gpu_memory_pool = object()
    dask_processor = types.ModuleType(f"{package}.dask_processor")
    dask_processor.run_pipeline = run_pipeline
    dask_cluster = types.ModuleType(f"{package}.dask_cluster")
    dask_cluster.make_cluster = make_cluster
    monkeypatch.setitem(sys.modules, f"{package}.gpu_memory", gpu_memory)
    monkeypatch.setitem(sys.modules, f"{package}.dask_processor", dask_processor)
    monkeypatch.setitem(sys.modules, f"{package}.dask_cluster", dask_cluster)
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    spec = importlib.util.spec_from_file_location(
        package,
        core_dir / "__init__.py",
        submodule_search_locations=[str(core_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    spec.loader.exec_module(module)

    assert module.run_pipeline is run_pipeline
    assert module.make_cluster is make_cluster
    assert module.__all__ == ["gpu_memory_pool", "run_pipeline", "make_cluster"]


def test_preprocess_worker_gdal_cache_respects_aggregate_budget():
    from FujiShaderGPU.io.dem_preprocess import _worker_gdal_cache_mb

    per_worker = _worker_gdal_cache_mb(available_gb=1.0, n_workers=8)
    aggregate_budget = int(1.0 * 1024 * 0.4)
    assert per_worker == 51
    assert per_worker * 8 <= aggregate_budget


def test_ao_radius_and_large_norm_stat_halo_are_not_capped():
    from FujiShaderGPU.algorithms._norm_stats import (
        _norm_stat_halo_pixels,
        _norm_stat_window_geometry,
    )

    params = {"radius": 5000}
    halo = _norm_stat_halo_pixels("ambient_occlusion", params)
    margin, tile = _norm_stat_window_geometry(
        "ambient_occlusion", params, max_tile=4096,
    )
    assert halo == 5016
    assert margin == halo
    assert tile >= 4 * margin
