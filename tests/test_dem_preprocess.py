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

from FujiShaderGPU.io.dem_preprocess import _fill_coarse_surface, _coarse_shape


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
