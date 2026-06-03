import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# dem_preprocess imports rasterio + osgeo at module load.
pytest.importorskip("rasterio")
pytest.importorskip("osgeo")
pytest.importorskip("scipy")

import numpy as np
import rasterio
from rasterio.transform import from_origin

from FujiShaderGPU.io import dem_preprocess
from FujiShaderGPU.io.dem_preprocess import _fill_coarse_surface, _coarse_shape


def _write_test_raster(path, data, *, nodata=None):
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, data.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_fill_coarse_surface_fills_voids_and_preserves_valid():
    coarse = np.arange(64, dtype=np.float32).reshape(8, 8)
    valid = np.ones((8, 8), dtype=bool)
    valid[3:5, 3:5] = False  # interior void

    out = _fill_coarse_surface(coarse, valid)

    # Every cell is finite (no NaN remains) and valid cells are untouched.
    assert np.isfinite(out).all()
    assert np.allclose(out[valid], coarse[valid])
    # Filled cells land within the data range (smooth interpolation, no extrapolation blow-up).
    assert float(out[~valid].min()) >= float(coarse[valid].min()) - 1e-3
    assert float(out[~valid].max()) <= float(coarse[valid].max()) + 1e-3


def test_fill_coarse_surface_all_invalid_returns_zeros():
    coarse = np.full((4, 4), np.nan, dtype=np.float32)
    valid = np.zeros((4, 4), dtype=bool)

    out = _fill_coarse_surface(coarse, valid)

    assert np.isfinite(out).all()
    assert np.all(out == 0.0)


def test_coarse_shape_caps_longest_side():
    ch, cw = _coarse_shape(width=240000, height=220000, coarse_max=2048)
    assert max(ch, cw) <= 2048
    # Aspect ratio is roughly preserved.
    assert abs((cw / ch) - (240000 / 220000)) < 0.05


def test_preprocess_fill_none_uses_direct_translate_fast_path(tmp_path, monkeypatch):
    src_path = tmp_path / "input.tif"
    dst_path = tmp_path / "output.tif"
    data = np.arange(256, dtype=np.float32).reshape(16, 16)
    data[0, 0] = -9999.0
    _write_test_raster(src_path, data, nodata=-9999.0)

    calls = []

    def fake_fast_translate(src_path_arg, dst_path_arg, **kwargs):
        calls.append((Path(src_path_arg), Path(dst_path_arg), kwargs))
        Path(dst_path_arg).write_bytes(b"fake-cog")

    def fail_streaming_translate(*args, **kwargs):
        raise AssertionError("streaming path should not be used")

    monkeypatch.setattr(dem_preprocess, "_translate_source_to_cog_fast", fake_fast_translate)
    monkeypatch.setattr(dem_preprocess, "_translate_to_cog", fail_streaming_translate)

    dem_preprocess.preprocess_dem_to_cog(
        str(src_path),
        str(dst_path),
        fill_mode="none",
        overwrite=True,
        detect_nodata=False,
    )

    assert len(calls) == 1
    assert calls[0][0] == src_path
    assert calls[0][1] == dst_path
    assert calls[0][2]["src_nodata"] == -9999.0
    assert np.isnan(calls[0][2]["dst_nodata"])


def test_preprocess_noop_fill_uses_direct_translate_fast_path(tmp_path, monkeypatch):
    src_path = tmp_path / "input.tif"
    dst_path = tmp_path / "output.tif"
    data = np.arange(256, dtype=np.float32).reshape(16, 16)
    _write_test_raster(src_path, data)

    calls = []

    def fake_fast_translate(src_path_arg, dst_path_arg, **kwargs):
        calls.append((Path(src_path_arg), Path(dst_path_arg), kwargs))
        Path(dst_path_arg).write_bytes(b"fake-cog")

    def fail_streaming_translate(*args, **kwargs):
        raise AssertionError("streaming path should not be used")

    monkeypatch.setattr(dem_preprocess, "_translate_source_to_cog_fast", fake_fast_translate)
    monkeypatch.setattr(dem_preprocess, "_translate_to_cog", fail_streaming_translate)

    dem_preprocess.preprocess_dem_to_cog(
        str(src_path),
        str(dst_path),
        fill_mode="enclosed",
        overwrite=True,
        detect_nodata=False,
    )

    assert len(calls) == 1
    assert calls[0][0] == src_path
    assert calls[0][1] == dst_path
    assert calls[0][2]["src_nodata"] is None
    assert np.isnan(calls[0][2]["dst_nodata"])
