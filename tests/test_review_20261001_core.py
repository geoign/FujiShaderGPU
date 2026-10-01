"""Regression tests for the 2026-10-01 review: I/O, pixel scales and shared helpers."""
import importlib
import sys
import types

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
rasterio = pytest.importorskip("rasterio")
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402


def _import_dask_processor(monkeypatch):
    """Import dask_processor without requiring dask-cuda on Windows."""
    pytest.importorskip("rioxarray")
    pytest.importorskip("distributed")
    if "dask_cuda" not in sys.modules:
        fake_dask_cuda = types.ModuleType("dask_cuda")
        fake_dask_cuda.LocalCUDACluster = object
        monkeypatch.setitem(sys.modules, "dask_cuda", fake_dask_cuda)
    return importlib.import_module("FujiShaderGPU.core.dask_processor")


def _write_tif(path, arr, *, transform, crs=None, nodata=None, overviews=None, scale=None):
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
        dtype=str(arr.dtype), crs=crs, transform=transform, nodata=nodata, tiled=True,
    ) as dst:
        dst.write(arr, 1)
        if scale is not None:
            dst.scales = (scale,)
        if overviews:
            dst.build_overviews(overviews, Resampling.average)


# --- curvature: one derivative frame on north-up rasters -------------------

@pytest.mark.parametrize("curvature_type", ["mean", "gaussian", "planform", "profile"])
def test_curvature_invariant_to_pixel_scale_y_sign(curvature_type):
    from FujiShaderGPU.algorithms._impl_curvature import compute_curvature_block

    y, x = np.mgrid[0:101, 0:101].astype(np.float32)
    cone = cp.asarray(100.0 - np.hypot(x - 50, y - 50))
    north_up = compute_curvature_block(
        cone, curvature_type=curvature_type, pixel_scale_x=1.0, pixel_scale_y=-1.0)
    south_up = compute_curvature_block(
        cone, curvature_type=curvature_type, pixel_scale_x=1.0, pixel_scale_y=1.0)
    cp.testing.assert_allclose(north_up[5:-5, 5:-5], south_up[5:-5, 5:-5], atol=1e-6)
    if curvature_type == "planform":
        # A cone is radially symmetric: south and east of the peak must match.
        assert abs(float(north_up[80, 50]) - float(north_up[50, 80])) < 1e-3


# --- coarse down/up sampling registration ----------------------------------

@pytest.mark.parametrize("n,factor", [(1024, 8), (1000, 16)])
def test_upsample_is_cell_centre_registered(n, factor):
    from FujiShaderGPU.algorithms._nan_utils import _downsample_nan_aware, _upsample_to_shape

    y, x = cp.mgrid[0:n, 0:n].astype(cp.float32)
    ramp = x + 0.5 * y
    coarse = _downsample_nan_aware(ramp, factor)
    inner = (slice(factor, n - 2 * factor), slice(factor, n - 2 * factor))
    for kw in ({"factor": factor}, {}):
        up = _upsample_to_shape(coarse, ramp.shape, **kw)
        assert up.shape == ramp.shape
        assert float(cp.abs(up - ramp)[inner].max()) < 1e-3


def test_multiscale_terrain_default_scales_have_no_chunk_seams():
    da = pytest.importorskip("dask.array")
    import dask
    from scipy.ndimage import gaussian_filter
    from FujiShaderGPU.algorithms._impl_multiscale_terrain import (
        MultiscaleDaskAlgorithm, compute_multiscale_combined_raw, multiscale_stat_func,
    )

    n = 2176  # > 2048 so the large default scales take the coarse path
    rng = np.random.default_rng(1)
    z = (gaussian_filter(rng.standard_normal((n, n)), 3) * 60
         + gaussian_filter(rng.standard_normal((n, n)), 60) * 20000).astype(np.float32)
    g = cp.asarray(z)
    params = dict(scales=[1, 10, 50, 100],
                  global_stats=multiscale_stat_func(
                      compute_multiscale_combined_raw(g, scales=[1, 10, 50, 100])))
    with dask.config.set(scheduler="synchronous"):
        ref = MultiscaleDaskAlgorithm().process(
            da.from_array(g, chunks=n, asarray=False), **params).compute()
        out = MultiscaleDaskAlgorithm().process(
            da.from_array(g, chunks=1088, asarray=False), **params).compute()
    seam = cp.abs(out - ref)[300:-300, 1086:1090]
    assert float(seam.max()) < 5e-3


# --- metric pixel scales ----------------------------------------------------

def test_dask_scale_detection_refuses_degree_pixels_without_crs(tmp_path, monkeypatch):
    dp = _import_dask_processor(monkeypatch)
    from FujiShaderGPU.core.dask_io import load_input_dataarray

    path = tmp_path / "nocrs.tif"
    _write_tif(path, np.ones((20, 30), np.float32),
               transform=from_origin(138, 36, 1 / 3600, 1 / 3600))
    dem = load_input_dataarray(str(path), 16)
    with pytest.raises(ValueError, match="look geographic"):
        dp._detect_metric_scales_from_dataarray(dem)


def test_dask_validate_inputs_accepts_remote_sources(monkeypatch):
    dp = _import_dask_processor(monkeypatch)
    dp.validate_inputs("https://example.com/dem.tif")
    dp.validate_inputs("/vsis3/bucket/dem.tif")
    with pytest.raises(FileNotFoundError):
        dp.validate_inputs("definitely_missing_local_dem.tif")


# --- Zarr input -------------------------------------------------------------

@pytest.mark.parametrize("name,dims", [
    ("z", ("y", "x")),
    ("elevation", ("lat", "lon")),
])
def test_zarr_input_keeps_crs_and_metric_scales(tmp_path, monkeypatch, name, dims):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    dp = _import_dask_processor(monkeypatch)
    from FujiShaderGPU.core.dask_io import load_input_dataarray

    yd, xd = dims
    lat = 36 - np.arange(20) / 3600
    lon = 138 + np.arange(30) / 3600
    arr = xr.DataArray(np.random.rand(20, 30).astype("float32"), dims=dims,
                       coords={yd: lat, xd: lon}, name=name)
    arr = arr.rio.set_spatial_dims(x_dim=xd, y_dim=yd).rio.write_crs("EPSG:4326")
    path = tmp_path / "in.zarr"
    arr.to_dataset(name=name).to_zarr(path, mode="w", zarr_format=2)

    dem = load_input_dataarray(str(path), 16)
    assert dem.ndim == 2
    assert dem.rio.crs is not None and dem.rio.crs.to_epsg() == 4326
    sx, sy, _mean, is_geo, _lat = dp._detect_metric_scales_from_dataarray(dem)
    assert is_geo
    assert 20 < sx < 30 and -32 < sy < -29  # ~1 arc-second at 36N, north-up


# --- scaled input / NoData override source ----------------------------------

def test_reject_scaled_input():
    from FujiShaderGPU.io.raster_info import reject_scaled_input

    reject_scaled_input(1.0, 0.0, "a.tif")
    reject_scaled_input(None, None, "a.tif")
    with pytest.raises(ValueError, match="scale=0.1"):
        reject_scaled_input(0.1, 0.0, "a.tif")


def test_dask_loader_rejects_scaled_dem(tmp_path):
    pytest.importorskip("rioxarray")
    from FujiShaderGPU.core.dask_io import load_input_dataarray

    path = tmp_path / "scaled.tif"
    _write_tif(path, np.full((16, 16), 800, np.int16),
               transform=from_origin(0, 160, 10, 10), crs="EPSG:32654", scale=0.1)
    with pytest.raises(ValueError, match="scaled rasters are not supported"):
        load_input_dataarray(str(path), 16)


def test_nodata_override_source_masks_sentinel_in_decimated_reads(tmp_path):
    pytest.importorskip("osgeo")
    from FujiShaderGPU.io.raster_info import nodata_override_source

    arr = np.full((512, 512), 100.0, np.float32)
    arr[:, :150] = -9999.0
    run_dir = tmp_path / "run [1]"
    run_dir.mkdir()
    path = run_dir / "dem.tif"
    _write_tif(path, arr, transform=from_origin(0, 5120, 10, 10), crs="EPSG:32654",
               overviews=[2, 4, 8])

    assert nodata_override_source(str(path), None) == str(path)
    assert nodata_override_source(str(path), float("nan")) == str(path)
    src = nodata_override_source(str(path), -9999.0)
    with rasterio.open(src) as ds:
        assert ds.nodata == -9999.0
        assert ds.overviews(1) == [2, 4, 8]
        sample = ds.read(1, out_shape=(64, 64), resampling=Resampling.average, masked=True)
    assert float(sample.min()) == 100.0 and float(sample.max()) == 100.0


def test_tile_backend_rejects_scaled_dem(tmp_path):
    from FujiShaderGPU.core.tile_processor import process_dem_tiles

    path = tmp_path / "scaled.tif"
    _write_tif(path, np.full((64, 64), 800, np.int16),
               transform=from_origin(0, 640, 10, 10), crs="EPSG:32654", scale=0.1)
    with pytest.raises(ValueError, match="scaled rasters are not supported"):
        process_dem_tiles(str(path), str(tmp_path / "out.tif"),
                          tmp_tile_dir=str(tmp_path / "tiles"), algorithm="slope",
                          show_progress=False)


def test_blur_local_mode_uses_blur_radius():
    from FujiShaderGPU.algorithms._impl_blur import _resolve_radius

    assert _resolve_radius({"mode": "local", "radii": [1], "radius": 40.0}) == 40.0
    assert _resolve_radius({"mode": "spatial", "radii": [8, 32], "radius": 40.0}) == 8.0
    assert _resolve_radius({"radius": 12.0}) == 12.0


@pytest.mark.parametrize("length,chunk", [(2600, 512), (3000, 1024), (5000, 4096), (300, 1024), (4096, 4096)])
def test_balanced_chunks_have_no_thin_tail(monkeypatch, length, chunk):
    dp = _import_dask_processor(monkeypatch)
    sizes = dp._balanced_chunks(length, chunk)
    assert sum(sizes) == length
    assert max(sizes) <= chunk
    assert len(sizes) == -(-length // chunk)
    assert min(sizes) >= min(length, chunk) // 2


def test_frangi_balanced_chunks_match_single_block():
    """A thin trailing chunk used to cap every multiscale halo (seams)."""
    da = pytest.importorskip("dask.array")
    import dask
    from scipy.ndimage import gaussian_filter
    from FujiShaderGPU.algorithms._impl_frangi import FrangiAlgorithm
    from FujiShaderGPU.core.dask_processor import _balanced_chunks

    rng = np.random.default_rng(3)
    z = (gaussian_filter(rng.standard_normal((1300, 1100)), 4) * 50
         + gaussian_filter(rng.standard_normal((1300, 1100)), 40) * 3000).astype(np.float32)
    g = cp.asarray(z)
    params = dict(radii=[2, 8, 32], weights=[0.5, 0.3, 0.2], pixel_size=10.0,
                  pixel_scale_x=10.0, pixel_scale_y=-10.0, global_stats=(0.0, 1.0))
    chunks = tuple(_balanced_chunks(n, 256) for n in g.shape)
    with dask.config.set(scheduler="synchronous"):
        ref = FrangiAlgorithm().process(da.from_array(g, chunks=g.shape, asarray=False), **params).compute()
        out = FrangiAlgorithm().process(da.from_array(g, chunks=chunks, asarray=False), **params).compute()
    assert float(cp.abs(out - ref).max()) < 1e-3
