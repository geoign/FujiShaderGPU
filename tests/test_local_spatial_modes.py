import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cp = pytest.importorskip('cupy')
da = pytest.importorskip('dask.array')


@pytest.mark.parametrize('algo_key, extra_params', [
    ('slope', {'unit': 'degree'}),
    ('specular', {'roughness_scale': 8.0, 'shininess': 12.0}),
    ('atmospheric_scattering', {'scattering_strength': 0.6}),
])
def test_local_and_spatial_modes_smoke(algo_key, extra_params):
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS

    x = cp.random.random((64, 64), dtype=cp.float32)
    xd = da.from_array(x, chunks=(32, 32))
    algo = ALGORITHMS[algo_key]

    y_local = algo.process(xd, mode='local', **extra_params).compute()
    y_spatial = algo.process(
        xd,
        mode='spatial',
        radii=[2, 4, 8],
        weights=[0.5, 0.3, 0.2],
        agg='mean',
        **extra_params,
    ).compute()

    assert y_local.shape == (64, 64)
    assert y_spatial.shape == (64, 64)
    assert str(y_local.dtype) == 'float32'
    assert str(y_spatial.dtype) == 'float32'


def test_multiscale_terrain_accepts_global_stats():
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS

    x = cp.random.random((64, 64), dtype=cp.float32)
    xd = da.from_array(x, chunks=(32, 32))
    algo = ALGORITHMS['multiscale_terrain']
    y = algo.process(
        xd,
        scales=[1, 4, 8],
        weights=[0.5, 0.3, 0.2],
        global_stats=(-0.2, 0.2),
    ).compute()

    assert y.shape == (64, 64)
    assert str(y.dtype) == 'float32'


def test_fractal_anomaly_uses_provided_global_stats(monkeypatch):
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS
    import FujiShaderGPU.algorithms.dask_shared as dask_shared

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("compute_global_stats must not be called when global_stats is provided")

    monkeypatch.setattr(dask_shared, "compute_global_stats", _should_not_be_called)

    x = cp.random.random((256, 256), dtype=cp.float32)
    xd = da.from_array(x, chunks=(128, 128))
    algo = ALGORITHMS['fractal_anomaly']
    y = algo.process(
        xd,
        radii=[2, 4, 8, 16, 32],
        global_stats=(2.5, 0.5),
    ).compute()

    assert y.shape == (256, 256)
    assert str(y.dtype) == 'float32'
