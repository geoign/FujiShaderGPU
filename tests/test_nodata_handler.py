import numpy as np

from FujiShaderGPU.utils.nodata_handler import _handle_nodata_ultra_fast


def test_nodata_fill_does_not_use_hard_zero_for_sparse_mask():
    dem = np.array(
        [
            [100.0, 102.0, 104.0],
            [106.0, 0.0, 110.0],
            [112.0, 114.0, 116.0],
        ],
        dtype=np.float32,
    )
    mask = dem == 0.0

    filled = _handle_nodata_ultra_fast(dem, mask)

    valid = dem[~mask]
    assert float(filled[1, 1]) != 0.0
    assert float(filled[1, 1]) >= float(valid.min())
    assert float(filled[1, 1]) <= float(valid.max())
    assert np.allclose(filled[~mask], dem[~mask])


def test_nodata_fill_all_masked_returns_zero():
    dem = np.zeros((4, 4), dtype=np.float32)
    mask = np.ones_like(dem, dtype=bool)

    filled = _handle_nodata_ultra_fast(dem, mask)

    assert np.all(filled == 0.0)
