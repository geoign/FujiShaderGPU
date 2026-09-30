"""Regression tests for the 2026-10-01 review of ``prepare`` void filling."""
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("osgeo")
pytest.importorskip("scipy")

import numpy as np
import rasterio
from rasterio.transform import from_origin

from FujiShaderGPU.algorithms._pyramid_fill import pushpull_fill
from FujiShaderGPU.io.dem_preprocess import (
    _detect_sentinel_nodata,
    preprocess_dem_to_cog,
)


def _pushpull(coarse, valid, backend):
    # ``zoom`` is passed as the production callers do (it is now unused).
    if backend == "numpy":
        from scipy.ndimage import zoom

        return pushpull_fill(coarse, valid, xp=np, zoom=zoom)
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
        from cupyx.scipy.ndimage import zoom as cp_zoom
    except Exception as exc:  # pragma: no cover - no usable CUDA device
        pytest.skip(f"CuPy backend unavailable: {exc}")
    out = pushpull_fill(cp.asarray(coarse), cp.asarray(valid), xp=cp, zoom=cp_zoom)
    return cp.asnumpy(out)


def _write(path, data, *, nodata=None):
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs="EPSG:32654",
        transform=from_origin(0, data.shape[0], 1, 1), nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _prepare(tmp_path, data, *, nodata=None, **kwargs):
    src = tmp_path / "in.tif"
    dst = tmp_path / "out.tif"
    _write(src, data, nodata=nodata)
    kwargs.setdefault("max_workers", 1)
    preprocess_dem_to_cog(str(src), str(dst), overwrite=True, **kwargs)
    with rasterio.open(dst) as ds:
        return ds.read(1)


# ---- 1) push-pull corner starvation -----------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("shape", [(64, 64), (205, 2048), (37, 51)])
def test_pushpull_void_touching_nw_corner_keeps_constant(shape, backend):
    coarse = np.full(shape, 5000.0, np.float32)
    valid = np.ones(shape, bool)
    valid[:, : shape[1] // 2] = False  # void touches (0, 0)
    out = _pushpull(coarse, valid, backend)
    np.testing.assert_allclose(out, 5000.0, rtol=0, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_pushpull_island_corners_follow_island_edge(backend):
    n = 257
    y, x = np.mgrid[0:n, 0:n]
    coarse = (1000 + 500 * np.sin(x / 50.0) * np.cos(y / 70.0)).astype(np.float32)
    valid = (x - n / 2) ** 2 + (y - n / 2) ** 2 < (n / 3) ** 2
    out = _pushpull(coarse, valid, backend)
    lo, hi = float(coarse[valid].min()), float(coarse[valid].max())
    assert np.isfinite(out).all()
    # Every fill (corners included) stays within the island's value range --
    # the old corner-aligned pyramid pulled all four corners to 0.0.
    assert out[~valid].min() >= lo - 1.0
    assert out[~valid].max() <= hi + 1.0
    np.testing.assert_array_equal(out[valid], coarse[valid])


# ---- 2) voids smaller than a coarse cell ------------------------------------
def _unsampled_index(n, n_samples, start):
    """First index >= start such that neither it nor the next one is sampled."""
    sampled = {int((j + 0.5) * n / n_samples) for j in range(n_samples)}
    return next(i for i in range(start, n) if i not in sampled
                and i + 1 not in sampled)


@pytest.mark.parametrize("fill_mode", ["enclosed", "all"])
def test_voids_smaller_than_coarse_cell_are_filled(tmp_path, fill_mode):
    n, coarse_max = 300, 32  # 32x32 coarse, 128x128 nearest sub-samples
    y, x = np.mgrid[0:n, 0:n]
    dem = (100 + 0.1 * x + 0.2 * y).astype(np.float32)
    i = _unsampled_index(n, coarse_max * 4, n // 3)
    j = _unsampled_index(n, coarse_max * 4, n // 2)
    dem[i, i] = -9999.0                   # 1-px void
    dem[j:j + 2, i:i + 2] = -9999.0       # 2x2 void (unsampled rows)
    out = _prepare(tmp_path, dem, nodata=-9999.0, fill_mode=fill_mode,
                   coarse_max=coarse_max, detect_nodata=False)
    assert np.isfinite(out).all()
    assert abs(float(out[i, i]) - (100 + 0.1 * i + 0.2 * i)) < 5.0


# ---- 3) enclosed: ragged coastline exterior ---------------------------------
def test_enclosed_keeps_ragged_border_connected_sea(tmp_path):
    h, w = 200, 2000
    y, x = np.mgrid[0:h, 0:w]
    dem = (100 + 0.01 * x + 0.02 * y).astype(np.float32)
    sea = x < 505 + (y % 7)               # touches the left border
    hole = (np.abs(y - 100) < 10) & (np.abs(x - 1500) < 10)
    dem[sea | hole] = -9999.0
    out = _prepare(tmp_path, dem, nodata=-9999.0, fill_mode="enclosed",
                   coarse_max=64, detect_nodata=False)
    assert int(np.isfinite(out[sea]).sum()) == 0
    assert np.isfinite(out[hole]).all()
    assert np.isfinite(out[~(sea | hole)]).all()


# ---- 4) --nodata auto rule 2 vs a declared NoData ---------------------------
def test_sentinel_range_extreme_skipped_when_disallowed():
    rng = np.random.default_rng(0)
    sample = rng.uniform(260, 400, size=(64, 64)).astype(np.float32)
    sample[:8, :] = 250.0  # 12.5% at the data minimum
    valid = np.ones_like(sample, dtype=bool)
    assert _detect_sentinel_nodata(sample, valid) == 250.0
    assert _detect_sentinel_nodata(sample, valid, allow_range_extreme=False) is None
    sample[:8, :] = -32768.0  # a known sentinel is still detected
    assert _detect_sentinel_nodata(sample, valid, allow_range_extreme=False) == -32768.0


def test_declared_nodata_keeps_flat_lake_at_data_minimum(tmp_path):
    n = 300
    y, x = np.mgrid[0:n, 0:n]
    dem = (260 + 0.05 * x + 0.03 * y + 5 * np.sin(x / 40.0)).astype(np.float32)
    lake = (x - 150) ** 2 + (y - 150) ** 2 < 45 ** 2   # ~7% of the area
    dem[lake] = 250.0
    dem[:3, :3] = -9999.0
    out = _prepare(tmp_path, dem, nodata=-9999.0, fill_mode="enclosed")
    np.testing.assert_array_equal(out[lake], 250.0)


# ---- 5) staging dir: TMP/TEMP are not an explicit redirect ------------------
def test_resolve_tmp_dir_ignores_tmp_temp(tmp_path, monkeypatch):
    from FujiShaderGPU.utils.paths import resolve_tmp_dir

    for name in ("FUJISHADER_TMP_DIR", "CPL_TMPDIR", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TMP", str(tmp_path / "systmp"))
    monkeypatch.setenv("TEMP", str(tmp_path / "systmp"))
    chosen, origin = resolve_tmp_dir(tmp_path / "next_to_output")
    assert chosen == tmp_path / "next_to_output"
    assert origin is None

    monkeypatch.setenv("FUJISHADER_TMP_DIR", str(tmp_path / "explicit"))
    chosen, origin = resolve_tmp_dir(tmp_path / "next_to_output")
    assert chosen == tmp_path / "explicit"
    assert origin == "FUJISHADER_TMP_DIR"


def test_undeclared_sentinel_ignores_averaged_overviews(tmp_path):
    """Overviews averaged across an undeclared 0 sentinel must not leak mixed
    'terrain' values into the fill of the sea."""
    from rasterio.enums import Resampling

    h, w = 400, 3000
    y, x = np.mgrid[0:h, 0:w]
    dem = (5000 + 0.001 * x + 0.0007 * y).astype(np.float32)
    dem[x < 1003] = 0.0  # undeclared sea sentinel
    src = tmp_path / "ovr.tif"
    _write(src, dem)
    with rasterio.open(src, "r+") as ds:
        ds.build_overviews([2, 4, 8, 16], Resampling.average)
    dst = tmp_path / "out.tif"
    preprocess_dem_to_cog(str(src), str(dst), overwrite=True, fill_mode="all", max_workers=1)
    with rasterio.open(dst) as ds:
        out = ds.read(1)
    sea = out[:, :1003]
    assert np.isfinite(sea).all()
    assert float(sea.min()) > 4990.0
