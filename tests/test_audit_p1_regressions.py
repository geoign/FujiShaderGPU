"""Regression tests for the Phase 1 audit fixes (AUDIT_VERIFICATION_20260717.md).

Each test pins one fixed defect: M-1 (NaN lost under numeric nodata), H-3
(scale_drift NaN erosion from coarse smooths), H-1 (topousm_fast tile stats
computed from post-split small radii), L-38/N-6 (+/-inf quantized to valid
codes on both backends).
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("dask.array")


# ---------------------------------------------------------------------------
# M-1: numeric nodata + pre-existing NaN must both mask
# ---------------------------------------------------------------------------

def test_nodata_mask_merges_nan_with_numeric_nodata():
    from FujiShaderGPU.core.tile_processor import _build_nodata_mask

    data = np.array([[1.0, -9999.0], [np.nan, 2.0]], dtype=np.float32)
    mask = _build_nodata_mask(data, -9999.0)
    assert mask.tolist() == [[False, True], [True, False]]


def test_nodata_mask_does_not_swallow_values_near_nodata():
    from FujiShaderGPU.core.tile_processor import _build_nodata_mask

    # A relative tolerance on -9999 would mask everything within ~0.1;
    # -9998.9 is valid data.
    data = np.array([-9999.0, -9998.9, 0.0], dtype=np.float32)
    mask = _build_nodata_mask(data, -9999.0)
    assert mask.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# L-38 / N-6: +/-inf is NoData in quantization, on both backends
# ---------------------------------------------------------------------------

_QP = {"a_coef": 100.0, "b_coef": 0.0, "dn_min": 0, "dn_max": 255}


def test_quantize_array_maps_nonfinite_to_nodata_code():
    from FujiShaderGPU.io.output_encoding import quantize_array

    arr = np.array([np.inf, -np.inf, np.nan, 1.0], dtype=np.float32)
    dn = quantize_array(arr, _QP, "uint8")
    assert dn.tolist() == [0, 0, 0, 100]


def test_quantize_block_cp_matches_numpy_quantizer():
    # dask_processor pulls in the whole Dask-CUDA pipeline at import.
    pytest.importorskip("dask_cuda")
    from FujiShaderGPU.core.dask_processor import _quantize_block_cp

    block = cp.asarray([np.inf, -np.inf, np.nan, 1.0], dtype=cp.float32)
    dn = _quantize_block_cp(
        block, a_coef=100.0, b_coef=0.0, dn_min=0, dn_max=255, cp_dtype=cp.uint8
    )
    assert cp.asnumpy(dn).tolist() == [0, 0, 0, 100]


# ---------------------------------------------------------------------------
# H-3: scale_drift combine must not let NaN in coarse smooths erode outward
# ---------------------------------------------------------------------------

def test_drift_combine_block_does_not_erode_nan_from_smooths():
    from FujiShaderGPU.algorithms._impl_scale_drift import (
        _drift_combine_block,
        _drift_smooth_block,
    )

    cp.random.seed(0)
    dem = (cp.random.random((96, 96), dtype=cp.float32) * 100.0).astype(cp.float32)
    blob = cp.zeros((96, 96), dtype=bool)
    blob[40:48, 40:48] = True
    dem = cp.where(blob, cp.nan, dem)

    scales = [2.0, 4.0]
    # Emulate the coarse-overview path: smooth fields arrive with NaN
    # re-masked over the NoData footprint.
    smooths = [
        cp.where(blob, cp.nan, _drift_smooth_block(dem, scale=s)) for s in scales
    ]
    out = _drift_combine_block(
        dem, *smooths, scales=scales, pair_w=None,
        drift_output="magnitude", normalize=False, norm_lo=0.0, norm_scale=0.0,
    )
    # NaN footprint must be exactly the input footprint (no growth).
    assert bool(cp.all(cp.isnan(out) == blob))


# ---------------------------------------------------------------------------
# M-33 / N-2: one precise deg->m conversion, anisotropy preserved
# ---------------------------------------------------------------------------

def test_meters_per_degree_precise_series():
    from FujiShaderGPU.io.raster_info import meters_per_degree

    m_lon, m_lat = meters_per_degree(35.0)
    # WGS84 series reference values at 35N.
    assert m_lat == pytest.approx(110940.5, abs=15)
    assert m_lon == pytest.approx(91288.0, abs=15)


def test_metric_pixel_scales_geographic_anisotropic():
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from FujiShaderGPU.io.raster_info import (
        meters_per_degree,
        metric_pixel_scales_from_metadata,
    )

    t = Affine(0.001, 0.0, 130.0, 0.0, -0.001, 36.0)
    crs = rasterio.crs.CRS.from_epsg(4326)
    bounds = (130.0, 35.0, 131.0, 36.0)  # center latitude 35.5
    sx, sy, mean_m, is_geo, lat = metric_pixel_scales_from_metadata(
        transform=t, crs=crs, bounds=bounds
    )
    assert is_geo and lat == pytest.approx(35.5)
    m_lon, m_lat = meters_per_degree(35.5)
    # Signed, per-axis, and anisotropic (the isotropic mean misstates each
    # axis by ~10% here).
    assert sx == pytest.approx(0.001 * m_lon, rel=1e-9)
    assert sy == pytest.approx(-0.001 * m_lat, rel=1e-9)
    assert abs(sy) > abs(sx)


# ---------------------------------------------------------------------------
# H-1: topousm_fast tile prepass must use the pre-split full radii
# ---------------------------------------------------------------------------

def test_topousm_fast_prepass_uses_full_radii(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from FujiShaderGPU.algorithms._norm_stats import _compute_norm_stats_tiled

    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    terrain = (
        np.sin(xx / 13.0) * 40.0
        + np.cos(yy / 29.0) * 60.0
        + np.sin((xx + yy) / 97.0) * 120.0
    ).astype(np.float32)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        str(path), "w", driver="GTiff", height=h, width=w, count=1, dtype="float32"
    ) as dst:
        dst.write(terrain, 1)

    full = {"radii": [4, 16, 64], "weights": [0.5, 0.3, 0.2]}
    split = {
        "radii": [4], "weights": [1.0],
        "_topousm_fast_full_radii": [4, 16, 64],
        "_topousm_fast_full_weights": [0.5, 0.3, 0.2],
    }
    small = {"radii": [4], "weights": [1.0]}

    s_full = _compute_norm_stats_tiled(str(path), "topousm_fast", full)
    s_split = _compute_norm_stats_tiled(str(path), "topousm_fast", split)
    s_small = _compute_norm_stats_tiled(str(path), "topousm_fast", small)
    assert s_full is not None and s_split is not None and s_small is not None

    # Post-split params with the stashed full lists must reproduce the
    # full-radii stats (Dask parity) ...
    assert list(s_split) == pytest.approx(list(s_full), rel=1e-4)
    # ... and the small radii alone must NOT (otherwise this test proves nothing).
    assert not (list(s_small) == pytest.approx(list(s_full), rel=0.05))
