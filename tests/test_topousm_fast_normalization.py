import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cp = pytest.importorskip('cupy')


def test_topousm_fast_stat_func_is_robust_to_sparse_extremes():
    from FujiShaderGPU.algorithms.dask_shared import topousm_fast_stat_func

    base = cp.random.normal(0.0, 0.01, size=(2048,), dtype=cp.float32)
    # Inject a few extreme outliers that should not dominate scale.
    base[:3] = cp.asarray([10.0, -8.0, 12.0], dtype=cp.float32)

    scale = float(topousm_fast_stat_func(base)[0])

    assert scale > 0.0
    # Should stay near bulk distribution, not outlier magnitude.
    assert scale < 0.2


def test_topousm_fast_stat_func_does_not_clip_broad_ridge_tail():
    from FujiShaderGPU.algorithms.dask_shared import topousm_fast_stat_func

    data = cp.concatenate(
        [
            cp.full((800,), 0.01, dtype=cp.float32),
            cp.full((200,), 0.5, dtype=cp.float32),
        ]
    )

    scale = float(topousm_fast_stat_func(data)[0])

    assert scale == pytest.approx(0.5, rel=1e-2)


def test_topousm_fast_norm_func_is_linear_without_clip():
    from FujiShaderGPU.algorithms.dask_shared import topousm_fast_norm_func

    # Normalization is a pure linear divide by the p99 scale (no overflow clip):
    # the high-amplitude tail beyond +/-1 passes through unclipped.
    data = cp.asarray([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=cp.float32)
    result = topousm_fast_norm_func(data, (2.0,), cp.isnan(data))

    assert cp.asnumpy(result).tolist() == pytest.approx([-2.0, -0.5, 0.0, 0.5, 2.0])
