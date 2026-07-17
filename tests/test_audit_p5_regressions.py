"""Regression coverage for the final P5 audit cleanup."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _src(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_duplicate_radii_sum_and_normalize_their_weights():
    from FujiShaderGPU.core.tile_compute import deduplicate_radii_weights

    radii, weights = deduplicate_radii_weights([4, 4, 8], [0.2, 0.3, 0.5])
    assert radii == [4, 8]
    assert weights == pytest.approx([0.5, 0.5])


def test_tile_spatial_policy_and_vram_constants_are_shared():
    from FujiShaderGPU.core.tile_processor import SPATIAL_TILE_ALGORITHMS

    source = _src("FujiShaderGPU/core/tile_processor.py")
    assert "npr_edges" in SPATIAL_TILE_ALGORITHMS
    assert "spatial_algorithms =" not in source
    assert "VRAM_OVERHEAD_MULTIPLIER" in source
    assert "4.0 * 15.0" not in source
    assert source.index("user_radii_specified =") < source.index("Auto spatial radii")


def test_thread_cached_tile_readers_are_explicitly_closed(monkeypatch):
    from FujiShaderGPU.core import tile_io

    class _Reader:
        closed = False

        def read(self, *_args, **_kwargs):
            return np.ones((2, 2), dtype=np.float32)

        def close(self):
            self.closed = True

    reader = _Reader()
    tile_io.close_all_tile_readers()
    monkeypatch.setattr(tile_io.rasterio, "open", lambda *_args, **_kwargs: reader)
    tile_io.read_tile_window("input.tif", object())
    tile_io.close_all_tile_readers()
    assert reader.closed


def test_dask_compute_failure_is_not_retried_by_cog_fallback(tmp_path, monkeypatch):
    fake_dask_cuda = types.ModuleType("dask_cuda")
    fake_dask_cuda.LocalCUDACluster = object
    monkeypatch.setitem(sys.modules, "dask_cuda", fake_dask_cuda)

    import FujiShaderGPU.core.dask_processor as processor

    fallback_calls = []
    monkeypatch.setattr(processor, "check_gdal_version", lambda: (3, 8))
    monkeypatch.setattr(processor, "get_cog_options", lambda _dtype: {})
    monkeypatch.setattr(
        processor,
        "_materialize_dataarray",
        lambda *_args: (_ for _ in ()).throw(MemoryError("compute OOM")),
    )
    monkeypatch.setattr(
        processor, "_fallback_cog_write", lambda *_args: fallback_calls.append(True),
    )

    class _Data:
        dtype = np.dtype("float32")
        rio = type("Rio", (), {"crs": None})()

    with pytest.raises(MemoryError, match="compute OOM"):
        processor._write_cog_da_original(_Data(), tmp_path / "out.tif")
    assert fallback_calls == []


def test_dask_cluster_returns_explicit_memory_budget_not_private_attribute():
    source = _src("FujiShaderGPU/core/dask_cluster.py")
    processor_source = _src("FujiShaderGPU/core/dask_processor.py")
    assert "cluster._fujishader_mem" not in source
    assert "return cluster, client, memory_budget" in source
    assert "cluster_memory_budget" in processor_source
    assert "rmm_max_gb = min(" in source


def test_dask_input_validation_and_zarr_crs_warning_are_present():
    source = _src("FujiShaderGPU/core/dask_io.py")
    assert source.count("if da_in.ndim != 2") == 2
    assert "Zarr input has no CRS metadata" in source
    assert "dask.diagnostics" not in source


def test_output_range_rejects_reversed_override():
    from FujiShaderGPU.io.output_encoding import resolve_output_range

    with pytest.raises(ValueError, match="high > low"):
        resolve_output_range("hillshade", override=(1.0, 0.0))


def test_missing_crs_that_looks_geographic_is_rejected():
    from affine import Affine
    from rasterio.coords import BoundingBox
    from FujiShaderGPU.io.raster_info import metric_pixel_scales_from_metadata

    with pytest.raises(ValueError, match="look geographic"):
        metric_pixel_scales_from_metadata(
            transform=Affine(0.01, 0, 130, 0, -0.01, 40),
            crs=None,
            bounds=BoundingBox(130, 30, 140, 40),
        )


def test_tmp_directory_environment_value_cannot_be_a_file(tmp_path, monkeypatch):
    from FujiShaderGPU.utils.paths import resolve_tmp_dir

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setenv("FUJISHADER_TMP_DIR", str(file_path))
    with pytest.raises(NotADirectoryError, match="points to a file"):
        resolve_tmp_dir(tmp_path / "fallback")


def test_container_cpu_quota_uses_ceiling(monkeypatch):
    from FujiShaderGPU.utils import cpu

    monkeypatch.setattr(cpu.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cpu, "_cpuset_count", lambda: None)
    monkeypatch.setattr(cpu, "_cfs_quota_cores", lambda: 2.1)
    assert cpu.container_cpu_count() == 3


def test_multiscale_negative_weights_are_clamped():
    import dask.array as da
    from FujiShaderGPU.algorithms._nan_utils import _combine_multiscale_dask

    first = da.from_array(np.ones((2, 2), dtype=np.float32), chunks=(2, 2))
    second = da.from_array(np.full((2, 2), 3.0, dtype=np.float32), chunks=(2, 2))
    result = _combine_multiscale_dask([first, second], weights=[-1.0, 2.0]).compute()
    assert np.all(result == 3.0)


def test_numpy_weight_array_is_accepted_for_spatial_profile():
    from FujiShaderGPU.algorithms._nan_utils import _resolve_spatial_radii_weights

    radii, weights = _resolve_spatial_radii_weights(
        [2, 8], np.asarray([1.0, 3.0], dtype=np.float32), 1.0,
    )
    assert radii == [2, 8]
    assert weights == pytest.approx([0.25, 0.75])


def test_hybrid_scale_keys_do_not_round_and_collide():
    source = _src("FujiShaderGPU/algorithms/_nan_utils.py")
    hybrid = source[source.index("def _hybrid_combine_wrapper"):]
    assert "small_map = {float(s)" in hybrid
    assert "int(round(float(s)))" not in hybrid


def test_empty_multi_light_list_has_clear_validation_error():
    import cupy as cp
    from FujiShaderGPU.algorithms.common.kernels import multi_light_uncertainty

    with pytest.raises(ValueError, match="at least one"):
        multi_light_uncertainty(cp.ones((4, 4), dtype=cp.float32), azimuths=[])


def test_algorithm_array_defaults_are_immutable():
    from FujiShaderGPU.algorithms._impl_fractal_anomaly import compute_fractal_dimension_block
    from FujiShaderGPU.algorithms._impl_topousm_fast import compute_topousm_fast_efficient_block
    from FujiShaderGPU.algorithms._impl_visual_saliency import compute_visual_saliency_block

    assert inspect.signature(compute_fractal_dimension_block).parameters["radii"].default is None
    assert inspect.signature(compute_topousm_fast_efficient_block).parameters["radii"].default is None
    assert inspect.signature(compute_visual_saliency_block).parameters["scales"].default is None


def test_new_tile_algorithms_are_public_exports():
    import FujiShaderGPU.algorithms as algorithms

    expected = {
        "StructureTensorAlgorithm", "FrangiAlgorithm", "LICAlgorithm",
        "PhaseCongruencyAlgorithm", "TVDecompositionAlgorithm", "ScaleDriftAlgorithm",
    }
    assert expected.issubset(set(algorithms.__all__))


def test_p5_source_contracts_for_io_cli_and_metadata():
    preprocess = _src("FujiShaderGPU/io/dem_preprocess.py")
    cog = _src("FujiShaderGPU/io/cog_builder.py")
    base = _src("FujiShaderGPU/cli/base.py")
    system = _src("FujiShaderGPU/config/system_config.py")
    pyproject = _src("pyproject.toml")

    assert "initializer=_init_strip_worker" in preprocess
    assert "fill_mode='all' left" in preprocess
    assert cog.count("-allow_projection_difference") == 2
    assert "IGNORE_COG_LAYOUT_BREAK" not in cog
    assert "SetNoDataValue" not in cog
    assert "is_remote =" in base
    assert "Hardware detection is deferred" in system
    assert 'license = "MIT"' in pyproject
    assert "Operating System :: OS Independent" not in pyproject
