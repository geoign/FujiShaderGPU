"""Regression tests for P4 configuration and dead-code hygiene fixes."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _src(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _load_install_gdal():
    module_name = "_fujishader_test_install_gdal"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "tools/install_gdal.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_dask_cluster(monkeypatch):
    if "dask_cuda" not in sys.modules:
        fake_dask_cuda = types.ModuleType("dask_cuda")
        fake_dask_cuda.LocalCUDACluster = object
        monkeypatch.setitem(sys.modules, "dask_cuda", fake_dask_cuda)
    return importlib.import_module("FujiShaderGPU.core.dask_cluster")


def test_cluster_runtime_scopes_dask_config_and_logging(monkeypatch):
    from dask import config as dask_config

    cluster_module = _import_dask_cluster(monkeypatch)
    distributed_logger = cluster_module.logging.getLogger("distributed.core")
    previous_level = distributed_logger.level
    observed = {}

    def _make_cluster(_memory_fraction):
        observed["rechunk"] = dask_config.get("array.rechunk.method")
        observed["pause"] = dask_config.get("distributed.worker.memory.pause")
        return "cluster", "client"

    monkeypatch.setattr(cluster_module, "make_cluster", _make_cluster)
    with dask_config.set({"array.rechunk.method": "p2p"}):
        with cluster_module.cluster_runtime(0.5) as runtime:
            assert runtime == ("cluster", "client")
            assert distributed_logger.level == cluster_module.logging.WARNING
        assert dask_config.get("array.rechunk.method") == "p2p"

    assert observed == {"rechunk": "tasks", "pause": 0.85}
    assert distributed_logger.level == previous_level


def test_deprecated_dask_and_rmm_environment_paths_are_removed():
    linux_source = _src("FujiShaderGPU/cli/linux_cli.py")
    cluster_source = _src("FujiShaderGPU/core/dask_cluster.py")
    processor_source = _src("FujiShaderGPU/core/dask_processor.py")

    assert "DASK_DISTRIBUTED__" not in linux_source
    assert "RMM_POOL_SIZE" not in linux_source
    assert "RMM_ALLOCATOR" not in linux_source
    assert "DASK_RUNTIME_CONFIG" in cluster_source
    assert "with dask_config.set(DASK_RUNTIME_CONFIG)" in cluster_source
    assert "cluster_runtime(memory_fraction)" in processor_source
    assert "dask_config.set" not in processor_source


def test_allocator_selection_is_module_scoped_not_thread_scoped():
    source = _src("FujiShaderGPU/core/gpu_memory.py")

    assert source.count("cp.cuda.set_allocator(") == 1
    assert source.index("_initialize_allocator()") < source.index("def get_gpu_context")
    get_context_body = source[source.index("def get_gpu_context"):]
    assert "set_allocator" not in get_context_body


@pytest.mark.parametrize(("layout", "expected"), [("COG", True), ("TILED", False), ("not-cog", False)])
def test_cog_validator_requires_exact_cog_layout(tmp_path, monkeypatch, layout, expected):
    from FujiShaderGPU.io import cog_validator

    class _Overview:
        XSize = 256
        YSize = 256

    class _Band:
        def GetBlockSize(self):
            return (512, 512)

        def GetOverviewCount(self):
            return 4

        def GetOverview(self, _index):
            return _Overview()

    class _Dataset:
        RasterXSize = 4096
        RasterYSize = 4096
        RasterCount = 1

        def GetRasterBand(self, _index):
            return _Band()

        def GetMetadata(self, domain=None):
            return {"LAYOUT": layout, "COMPRESSION": "ZSTD"} if domain == "IMAGE_STRUCTURE" else {}

    path = tmp_path / "result.tif"
    path.write_bytes(b"test")
    monkeypatch.setattr(cog_validator.gdal, "Open", lambda *_args: _Dataset())

    assert cog_validator._validate_cog_for_qgis(str(path)) is expected


def test_overview_levels_and_threads_are_resource_adaptive():
    dask_source = _src("FujiShaderGPU/core/dask_processor.py")
    builder_source = _src("FujiShaderGPU/io/cog_builder.py")

    assert '"GDAL_NUM_THREADS", str(max(1, container_cpu_count()))' in dask_source
    assert "if level <= min_dimension" in dask_source
    assert builder_source.count("if level <= min_dimension") == 2


def test_windows_tile_pipeline_forwards_progress_flag():
    windows_source = _src("FujiShaderGPU/cli/windows_cli.py")
    tile_source = _src("FujiShaderGPU/core/tile_processor.py")

    assert 'show_progress=params["show_progress"]' in windows_source
    assert "show_progress: bool = True" in tile_source
    assert "if show_progress and completed_count % 10 == 0" in tile_source


def test_p4_dead_branches_and_duplicate_helpers_are_removed():
    bridge_source = _src("FujiShaderGPU/algorithms/tile/dask_bridge.py")
    normalization_source = _src("FujiShaderGPU/algorithms/_normalization.py")
    global_stats_source = _src("FujiShaderGPU/algorithms/_global_stats.py")
    topousm_source = _src("FujiShaderGPU/algorithms/_impl_topousm_fast.py")
    fractal_source = _src("FujiShaderGPU/algorithms/_impl_fractal_anomaly.py")

    assert "_resolve_radii_weights" not in bridge_source
    assert "compute_slope_spatial_block" not in bridge_source
    assert "compute_specular_spatial_block" not in bridge_source
    assert 'params.get("mode", "local")' in bridge_source  # live npr_edges guard remains
    assert "robust_unsigned_stretch_stat_func" not in normalization_source
    assert global_stats_source.count("def robust_unsigned_stretch_stat_func") == 1
    assert "cp.linspace" not in fractal_source
    assert "if len(radii) > 4" not in topousm_source


def test_fractal_debug_tool_calls_production_implementation():
    source = _src("tools/debug_fractal_anomaly.py")

    assert "from FujiShaderGPU.algorithms._impl_fractal_anomaly import" in source
    assert source.count("compute_fractal_dimension_block(") == 2
    assert "fractal_stat_func(raw)" in source
    assert "median_filter" not in source
    assert "beta = cov" not in source


def test_install_gdal_verify_handles_empty_output(monkeypatch):
    installer = _load_install_gdal()
    monkeypatch.setattr(installer.subprocess, "check_output", lambda *_args, **_kwargs: "")

    assert installer._verify() is None


def test_install_gdal_conda_detection_uses_active_interpreter(tmp_path, monkeypatch):
    installer = _load_install_gdal()
    monkeypatch.setattr(installer.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "unrelated-conda"))
    assert not installer._in_conda()

    (tmp_path / "conda-meta").mkdir()
    assert installer._in_conda()


def test_install_gdal_reports_missing_sudo(monkeypatch):
    installer = _load_install_gdal()
    monkeypatch.setattr(installer.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(installer.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="sudo.*unavailable"):
        installer._sudo_prefix()


def test_install_gdal_dry_run_never_verifies(monkeypatch):
    installer = _load_install_gdal()
    monkeypatch.setattr(installer.sys, "argv", ["install_gdal.py", "--dry-run"])
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(installer, "_in_conda", lambda: False)
    monkeypatch.setattr(installer, "_gdal_config_version", lambda: None)
    monkeypatch.setattr(
        installer,
        "_verify",
        lambda: (_ for _ in ()).throw(AssertionError("verification ran during dry-run")),
    )

    assert installer.main() == 2


def test_install_gdal_checks_numpy_install_result(monkeypatch):
    installer = _load_install_gdal()
    calls = []
    monkeypatch.setattr(installer, "_gdal_include_dir", lambda: None)
    monkeypatch.setattr(
        installer,
        "_run",
        lambda command, **_kwargs: calls.append(command) or 9,
    )

    assert not installer._pip_build_binding("3.10.0", dry=False)
    assert len(calls) == 1
