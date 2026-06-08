import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cp = pytest.importorskip("cupy")


def _radial(H=101, W=101):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)


def _openness(dem, otype):
    from FujiShaderGPU.algorithms._impl_openness import compute_openness_vectorized
    out = compute_openness_vectorized(
        cp.asarray(dem.astype(np.float32)),
        openness_type=otype, num_directions=16, max_distance=40,
    )
    return cp.asnumpy(out)


def test_positive_openness_high_on_convex_low_on_concave():
    r = _radial()
    peak = (50 - r)
    pit = (r - 50)
    c = 50
    pos_peak = _openness(peak, "positive")[c, c]
    pos_pit = _openness(pit, "positive")[c, c]
    # Yokoyama positive openness: convex (peak) > concave (pit).
    assert pos_peak > pos_pit


def test_negative_openness_high_on_concave_low_on_convex():
    r = _radial()
    peak = (50 - r)
    pit = (r - 50)
    c = 50
    neg_peak = _openness(peak, "negative")[c, c]
    neg_pit = _openness(pit, "negative")[c, c]
    # Negative openness: concave (pit) > convex (peak).
    assert neg_pit > neg_peak


def test_flat_terrain_is_fully_open_both_signs():
    flat = np.zeros((101, 101), np.float32)
    c = 50
    assert _openness(flat, "positive")[c, c] == pytest.approx(1.0, abs=1e-3)
    assert _openness(flat, "negative")[c, c] == pytest.approx(1.0, abs=1e-3)
