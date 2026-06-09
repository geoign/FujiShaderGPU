"""End-to-end tile-pipeline parity between projected and geographic DEMs.

Regression for two related bugs:

1. The tile backend pre-multiplied geographic DEMs by ``elevation_scale``
   while the shared normalization stats were computed on raw elevation, so the
   normalized output washed out by ~1/pixel_size (p99 ~0.07 instead of ~1).
2. ``geographic_mode`` flipped hillshade's azimuth and inverted its output,
   giving geographic DEMs a completely different (inverted) tone.

Both are gone: the same terrain processed as projected and as geographic must
now produce comparable outputs.
"""
import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402


@pytest.fixture(scope="module")
def terrain():
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 1.0, (300, 300)).astype(np.float32)
    # Cheap smoothing without scipy: separable box blur a few times.
    for _ in range(3):
        base = (np.roll(base, 1, 0) + base + np.roll(base, -1, 0)) / 3.0
        base = (np.roll(base, 1, 1) + base + np.roll(base, -1, 1)) / 3.0
    return (base * 300.0 + 500.0).astype(np.float32)


def _write_dem(path, data, transform, crs):
    profile = dict(
        driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform,
        tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _run(tmp_path, name, dem_path, algorithm):
    from FujiShaderGPU.core.tile_processor import process_dem_tiles

    out_path = os.path.join(tmp_path, f"{name}.tif")
    process_dem_tiles(
        dem_path, out_path,
        tmp_tile_dir=os.path.join(tmp_path, f"tiles_{name}"),
        algorithm=algorithm, mode="local", tile_size=512,
    )
    with rasterio.open(out_path) as src:
        return src.read(1)


@pytest.fixture(scope="module")
def dem_paths(tmp_path_factory, terrain):
    tmp = tmp_path_factory.mktemp("geo_parity")
    p_proj = str(tmp / "dem_proj.tif")
    p_geo = str(tmp / "dem_geo.tif")
    # ~30 m pixels: 30 m projected vs ~0.00027 deg at lat 35.5.
    _write_dem(p_proj, terrain, from_origin(0, 10000, 30, 30), "EPSG:32654")
    _write_dem(p_geo, terrain, from_origin(138.0, 35.5, 0.00027, 0.00027), "EPSG:4326")
    return str(tmp), p_proj, p_geo


def test_topousm_fast_normalization_matches_across_crs(dem_paths):
    tmp, p_proj, p_geo = dem_paths
    out_proj = _run(tmp, "tp_proj", p_proj, "topousm_fast")
    out_geo = _run(tmp, "tp_geo", p_geo, "topousm_fast")

    def p99abs(a):
        v = a[np.isfinite(a)]
        return float(np.percentile(np.abs(v), 99))

    pp, pg = p99abs(out_proj), p99abs(out_geo)
    # Normalized magnitude must be ~1 on both CRS (the elevation_scale bug
    # collapsed the geographic output to ~0.07 while projected stayed ~1).
    assert 0.6 < pp < 1.6, pp
    assert 0.6 < pg < 1.6, pg
    assert 0.7 < pg / pp < 1.4


def test_hillshade_tone_matches_across_crs(dem_paths):
    tmp, p_proj, p_geo = dem_paths
    out_proj = _run(tmp, "hs_proj", p_proj, "hillshade")
    out_geo = _run(tmp, "hs_geo", p_geo, "hillshade")

    vp = out_proj[np.isfinite(out_proj)]
    vg = out_geo[np.isfinite(out_geo)]
    # Same terrain, near-identical metric pixel size -> mean tone must agree
    # closely (the geographic_mode inversion made flat ground 0.29 vs 0.71).
    assert abs(float(vp.mean()) - float(vg.mean())) < 0.05
