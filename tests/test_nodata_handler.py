import numpy as np
import pytest

from FujiShaderGPU.utils.nodata_handler import (
    SCIPY_AVAILABLE,
    _handle_nodata_ultra_fast,
    fill_enclosed_nodata_holes,
)


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


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy is required for hole classification")
def test_fill_enclosed_nodata_holes_preserves_edge_connected_nodata():
    dem = np.arange(36, dtype=np.float32).reshape(6, 6)
    mask = np.zeros_like(dem, dtype=bool)
    mask[2:4, 2:4] = True
    mask[:, 0] = True
    dem[mask] = np.nan

    filled, remaining, hole_count = fill_enclosed_nodata_holes(dem, mask)

    assert hole_count == 1
    assert np.isfinite(filled[2:4, 2:4]).all()
    assert remaining[:, 0].all()
    assert not remaining[2:4, 2:4].any()
    assert np.isnan(filled[:, 0]).all()


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy is required for dense hole filling")
def test_fill_enclosed_nodata_holes_dense_path():
    dem = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros_like(dem, dtype=bool)
    mask[2:8:2, 2:8:2] = True
    dem[mask] = np.nan

    filled, remaining, hole_count = fill_enclosed_nodata_holes(
        dem,
        mask,
        max_holes_for_interpolation=2,
    )

    assert hole_count > 2
    assert np.isfinite(filled[mask]).all()
    assert not remaining.any()
