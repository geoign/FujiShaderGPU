"""Regression tests for P2 audit performance/OOM fixes."""
from pathlib import Path

import pytest


def _src(rel):
    import FujiShaderGPU
    return (Path(FujiShaderGPU.__file__).parent / rel).read_text(encoding="utf-8")


def test_radius_downsample_factor_is_block_shape_deterministic():
    from FujiShaderGPU.algorithms._nan_utils import _radius_to_downsample_factor

    kwargs = dict(radius=512, pixel_size=1.0, algorithm_name="topousm_fast")
    f_edge = _radius_to_downsample_factor(block_shape=(256, 4096), **kwargs)
    f_tile = _radius_to_downsample_factor(block_shape=(2048, 2048), **kwargs)
    f_dask = _radius_to_downsample_factor(block_shape=(4096, 4096), **kwargs)
    f_none = _radius_to_downsample_factor(block_shape=None, **kwargs)
    assert f_edge == f_tile == f_dask == f_none


def test_norm_stats_skip_modes_and_large_halo_window():
    from FujiShaderGPU.algorithms._norm_stats import (
        _norm_stats_unused_for_mode,
        _norm_stat_halo_pixels,
    )

    assert _norm_stats_unused_for_mode("structure_tensor", {"st_output": "orientation"})
    assert _norm_stats_unused_for_mode("scale_drift", {"drift_output": "direction"})
    assert _norm_stats_unused_for_mode("tv_decomposition", {"component": "structure"})
    assert not _norm_stats_unused_for_mode("scale_drift", {"drift_output": "magnitude"})
    # Large radii must not be capped to 1024px; halo/window sizing should contain
    # the algorithm footprint for N-3/M-20.
    assert _norm_stat_halo_pixels("topousm_fast", {"radii": [2048]}) > 1024
    assert _norm_stat_halo_pixels("fractal_anomaly", {"radii": [2048]}) > 4096


def test_hillshade_uses_stack_free_normal_math():
    src = _src("algorithms/_impl_hillshade.py")
    assert "normal = cp.stack" not in src
    assert "dot(normalized" in src


def test_structure_tensor_shares_coarse_cache_between_components():
    src = _src("algorithms/_impl_structure_tensor.py")
    assert "_coarse_cache = {}" in src
    assert "coarse_cache=_coarse_cache, component='u'" in src
    assert "coarse_cache=_coarse_cache, component='v'" in src


def test_openness_distance_uses_host_hypot_not_cupy_scalar_kernel():
    src = _src("algorithms/_impl_openness.py")
    assert "np.hypot(phys_dx, phys_dy)" in src
    assert "cp.sqrt(phys_dx" not in src


def test_dask_writer_no_late_prefetch_config_and_scatter_coarse_fields():
    src = _src("core/dask_processor.py")
    assert "prefetch_config" not in src
    assert "dask_config.set" not in src
    nan_src = _src("algorithms/_nan_utils.py")
    assert "def _scatter_to_workers" in src
    assert "client.scatter(v, broadcast=True" in src
    assert "def _scatter_if_client" in nan_src
    assert "coarse_resp = _scatter_if_client" in nan_src
    assert 'params["_topousm_fast_coarse_field"] = _scatter_to_workers' in src
    assert 'params[f"{_pfx}_large_fields"] = _scatter_to_workers' in src
    assert "Resolved multi-radius run" in src
    assert "gpu_arr = gpu_arr.rechunk((chunk, chunk))" in src


def test_gdal_cache_floor_and_nodata_inference_are_bounded():
    gdal_src = _src("config/gdal_config.py")
    dask_src = _src("core/dask_processor.py")
    tile_src = _src("core/tile_processor.py")
    assert "cache_mb = int(max(64, cache_mb))" in gdal_src
    assert "max(64, min(2048" in dask_src
    assert "out_shape=" in tile_src
    assert "Resampling.nearest" in tile_src
    assert "inferred nodata=0 from a decimated raster border sample" in tile_src
