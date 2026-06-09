import numpy as np
import pytest


def test_signed_algorithm_uses_nan_nodata():
    from FujiShaderGPU.core.tile_processor import _format_algorithm_output

    arr = np.array([[0.0, -0.5], [0.2, 1.0]], dtype=np.float32)
    out, out_nodata = _format_algorithm_output(
        result_core=arr,
        algorithm="fractal_anomaly",
    )

    assert out.dtype == np.float32
    assert np.isnan(out_nodata)
    assert out[0, 0] == 0.0


def test_hillshade_keeps_nan_nodata_through_clip():
    """A NaN NoData fill must survive hillshade's [0, 1] clip as NaN.

    Regression: the tile pipeline used to fill NoData with the numeric input
    sentinel (e.g. -9999), which hillshade's clip turned into a *valid* black
    0.0 pixel while the output tag switched to NaN.
    """
    from FujiShaderGPU.core.tile_processor import _format_algorithm_output

    arr = np.array([[np.nan, 0.5], [1.2, -0.1]], dtype=np.float32)
    out, out_nodata = _format_algorithm_output(
        result_core=arr,
        algorithm="hillshade",
    )

    assert np.isnan(out_nodata)
    assert np.isnan(out[0, 0])          # NoData stays NaN, not 0.0
    assert out[1, 0] == 1.0             # clip upper
    assert out[1, 1] == 0.0             # clip lower


def test_apply_nodata_mask_fills_nan():
    cp = pytest.importorskip("cupy")
    from FujiShaderGPU.core.tile_compute import apply_nodata_mask

    result = cp.ones((4, 4), dtype=cp.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    out = apply_nodata_mask(result, mask, np.nan)
    out_np = cp.asnumpy(out)
    assert np.isnan(out_np[0]).all()
    assert (out_np[1:] == 1.0).all()
