import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cp = pytest.importorskip('cupy')


def test_rvi_stat_func_is_robust_to_sparse_extremes():
    from FujiShaderGPU.algorithms.dask_shared import rvi_stat_func

    base = cp.random.normal(0.0, 0.01, size=(2048,), dtype=cp.float32)
    # Inject a few extreme outliers that should not dominate scale.
    base[:3] = cp.asarray([10.0, -8.0, 12.0], dtype=cp.float32)

    scale = float(rvi_stat_func(base)[0])

    assert scale > 0.0
    # Should stay near bulk distribution, not outlier magnitude.
    assert scale < 0.2
