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


def test_rvi_stat_func_does_not_clip_broad_ridge_tail():
    from FujiShaderGPU.algorithms.dask_shared import rvi_stat_func

    data = cp.concatenate(
        [
            cp.full((800,), 0.01, dtype=cp.float32),
            cp.full((200,), 0.5, dtype=cp.float32),
        ]
    )

    scale = float(rvi_stat_func(data)[0])

    assert scale == pytest.approx(0.108, rel=1e-2)


def test_rvi_norm_func_preserves_overflow_up_to_hard_cap():
    from FujiShaderGPU.algorithms.dask_shared import rvi_norm_func

    data = cp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=cp.float32)
    result = rvi_norm_func(data, (1.0,), cp.isnan(data))

    assert cp.asnumpy(result).tolist() == pytest.approx([-1.5, -1.0, 0.0, 1.0, 1.5])


def test_rvi_result_stats_sample_existing_result_not_second_rvi():
    da = pytest.importorskip("dask.array")
    from FujiShaderGPU.algorithms._impl_rvi import compute_rvi_result_stats

    arr = cp.concatenate(
        [
            cp.full((800,), 0.01, dtype=cp.float32),
            cp.full((200,), 0.5, dtype=cp.float32),
        ]
    ).reshape(20, 50)
    rvi = da.from_array(arr, chunks=(10, 25))

    scale = float(compute_rvi_result_stats(rvi)[0])

    assert scale == pytest.approx(0.108, rel=1e-2)
