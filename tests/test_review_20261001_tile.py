"""Regression tests for the 2026-10-01 review of the Windows tile backend.

1. halo sized from --scales / blur radius / local AO radius / openness
   max_distance (no seams between tile sizes)
2. explicit --pixel-size overrides the metadata pixel scales
3. skipped all-NoData tiles no longer shrink the output extent (incl. --cog-only)
4. band-first stack with a 3/4 px wide edge tile is not read as HxWxC
5. Python API without ``mode`` behaves as mode="spatial" end to end
6. tile directories containing '[' / ']' are globbed literally
"""
import os

import numpy as np
import pytest

pytest.importorskip("cupy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _make_dem(path, H, W, *, crs="EPSG:32654", px=5.0, origin=(500000.0, 4000000.0),
              nodata=None, void_cols_from=None, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    z = np.zeros((H, W), np.float64)
    for s, a in ((2, 3), (6, 20), (20, 80)):
        z += a * gaussian_filter(rng.standard_normal((H, W)), s) * s
    z = (z - z.min() + 100.0).astype(np.float32)
    if void_cols_from is not None:
        z[:, void_cols_from:] = nodata
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                crs=crs, transform=from_origin(origin[0], origin[1], px, px),
                tiled=True, blockxsize=128, blockysize=128)
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(z, 1)
        dst.build_overviews([2, 4], rasterio.enums.Resampling.average)
    return str(path)


def _run(tmp_path, inp, tag, **kw):
    from FujiShaderGPU.core.tile_processor import process_dem_tiles
    out = str(tmp_path / f"out_{tag}.tif")
    kw.setdefault("tmp_tile_dir", str(tmp_path / f"tiles_{tag}"))
    process_dem_tiles(inp, out, show_progress=False, max_workers=1, **kw)
    with rasterio.open(out) as s:
        return s.read().astype(np.float64), s.transform, s.width, s.height


def _max_diff(a, b):
    d = np.abs(a - b)
    d[~(np.isfinite(a) & np.isfinite(b))] = 0.0
    return float(d.max())


# 1 -------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm,params", [
    ("visual_saliency", {"scales": [2, 4, 8, 32]}),
    ("blur", {"radius": 20.0}),
    ("openness", {"mode": "local", "max_distance": 80}),
    ("ambient_occlusion", {"mode": "local", "radius": 80}),
])
def test_halo_covers_non_radii_size_params(tmp_path, algorithm, params):
    inp = _make_dem(tmp_path / "dem.tif", 320, 320)
    small = _run(tmp_path, inp, "s", algorithm=algorithm, tile_size=128, **dict(params))[0]
    big = _run(tmp_path, inp, "b", algorithm=algorithm, tile_size=1024, **dict(params))[0]
    assert small.shape == big.shape
    assert _max_diff(small, big) < 1e-4


def test_required_padding_reads_scales_and_local_sizes():
    from FujiShaderGPU.core.tile_processor import _required_padding_for_algorithm as req
    kw = dict(sigma=10.0, pixel_size=1.0, target_distances=None, tile_size=1024)
    assert req("visual_saliency", {"mode": "spatial", "scales": [2, 4, 8, 64]}, **kw) >= 320
    assert req("scale_space_surprise", {"mode": "spatial", "scales": [1, 2, 64]}, **kw) >= 257
    assert req("blur", {"mode": "spatial", "radius": 40}, **kw) >= 161
    assert req("openness", {"mode": "local", "max_distance": 200}, **kw) >= 201
    assert req("ambient_occlusion", {"mode": "local", "radius": 100}, **kw) >= 101


# 2 -------------------------------------------------------------------------
def test_explicit_pixel_size_overrides_metadata(tmp_path):
    inp = _make_dem(tmp_path / "dem.tif", 192, 192, px=5.0)
    meta = _run(tmp_path, inp, "meta", algorithm="slope", mode="local", tile_size=128)[0]
    one = _run(tmp_path, inp, "one", algorithm="slope", mode="local", tile_size=128,
               pixel_size=1.0)[0]
    assert np.nanmean(one) > np.nanmean(meta) * 1.5


def test_explicit_pixel_size_skips_geographic_tile_override(tmp_path):
    proj = _make_dem(tmp_path / "proj.tif", 192, 192, px=5.0)
    geo = _make_dem(tmp_path / "geo.tif", 192, 192, crs="EPSG:4326", px=0.0001,
                    origin=(138.0, 60.0))
    a = _run(tmp_path, proj, "p", algorithm="slope", mode="local", tile_size=128,
             pixel_size=3.0)[0]
    b = _run(tmp_path, geo, "g", algorithm="slope", mode="local", tile_size=128,
             pixel_size=3.0)[0]
    assert _max_diff(a, b) < 1e-4


# 3 + 6 ---------------------------------------------------------------------
def test_skipped_tiles_keep_extent_and_bracket_dir(tmp_path):
    from FujiShaderGPU.core.tile_processor import process_dem_tiles
    inp = _make_dem(tmp_path / "dem.tif", 384, 384, nodata=-9999.0, void_cols_from=150)
    tiles = tmp_path / "run[1]"
    arr, tr, w, h = _run(tmp_path, inp, "void", algorithm="slope", mode="local",
                         tile_size=128, tmp_tile_dir=str(tiles), keep_tiles=True)
    with rasterio.open(inp) as s:
        assert (w, h) == (s.width, s.height)
        assert tr == s.transform
    assert np.isnan(arr[0, :, 160:]).all()
    assert np.isfinite(arr[0, 5:-5, 5:140]).all()
    assert len([f for f in os.listdir(tiles) if f.startswith("tile_")]) == 6  # 3 skipped

    # --cog-only on the same (bracketed) tile dir keeps the full extent too.
    out2 = str(tmp_path / "out_cogonly.tif")
    process_dem_tiles(inp, out2, tmp_tile_dir=str(tiles), cog_only=True,
                      show_progress=False)
    with rasterio.open(out2) as s2, rasterio.open(inp) as s:
        assert (s2.width, s2.height) == (s.width, s.height)
        assert s2.transform == s.transform


# 4 -------------------------------------------------------------------------
def test_write_tile_output_band_first_width3(tmp_path):
    from FujiShaderGPU.core.tile_io import write_tile_output
    arr = np.arange(45, dtype=np.float32).reshape(3, 5, 3)
    prof = dict(driver="GTiff", height=5, width=3, count=3, dtype="float32",
                transform=from_origin(0, 5, 1, 1))
    path = str(tmp_path / "t.tif")
    write_tile_output(path, arr, prof)
    with rasterio.open(path) as s:
        np.testing.assert_array_equal(s.read(), arr)


def test_format_output_uses_core_shape():
    from FujiShaderGPU.core.tile_processor import _format_algorithm_output
    arr = np.zeros((3, 100, 3), np.float32)
    out, _ = _format_algorithm_output(arr, "hillshade", core_shape=(100, 3))
    assert out.shape == (3, 100, 3)


def test_hillshade_stack_edge_tile_width3(tmp_path):
    inp = _make_dem(tmp_path / "dem.tif", 64, 1027)
    kw = dict(algorithm="hillshade", radii=[2, 8, 32], agg="stack")
    a, _, w, h = _run(tmp_path, inp, "t1024", tile_size=1024, **kw)
    assert (w, h) == (1027, 64) and a.shape[0] == 3
    b = _run(tmp_path, inp, "t2048", tile_size=2048, **kw)[0]
    assert _max_diff(a[:, :, 1020:], b[:, :, 1020:]) < 1e-4


# 5 -------------------------------------------------------------------------
def test_mode_unset_matches_spatial(tmp_path):
    inp = _make_dem(tmp_path / "dem.tif", 256, 256)
    kw = dict(algorithm="ambient_occlusion", tile_size=256, radii=[2, 8])
    unset = _run(tmp_path, inp, "unset", **kw)[0]
    spatial = _run(tmp_path, inp, "spatial", mode="spatial", **kw)[0]
    assert _max_diff(unset, spatial) < 1e-6
