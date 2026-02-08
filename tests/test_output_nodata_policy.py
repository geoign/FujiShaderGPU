import numpy as np


def test_signed_algorithm_uses_nan_nodata():
    from FujiShaderGPU.core.tile_processor import _format_algorithm_output

    arr = np.array([[0.0, -0.5], [0.2, 1.0]], dtype=np.float32)
    out, out_nodata = _format_algorithm_output(
        result_core=arr,
        algorithm="fractal_anomaly",
        algo_params={},
        nodata=0.0,
    )

    assert out.dtype == np.float32
    assert np.isnan(out_nodata)
    assert out[0, 0] == 0.0

