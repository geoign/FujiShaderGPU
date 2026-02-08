import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cp = pytest.importorskip('cupy')
da = pytest.importorskip('dask.array')


@pytest.mark.parametrize('algo_key', [
    'scale_space_surprise',
    'multi_light_uncertainty',
])
def test_dask_algorithm_smoke(algo_key):
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS

    x = cp.random.random((64, 64), dtype=cp.float32)
    xd = da.from_array(x, chunks=(32, 32))
    y = ALGORITHMS[algo_key].process(xd).compute()

    assert y.shape == (64, 64)
    assert str(y.dtype) == 'float32'


@pytest.mark.parametrize('cls_name', [
    'ScaleSpaceSurpriseAlgorithm',
    'MultiLightUncertaintyAlgorithm',
])
def test_tile_algorithm_smoke(cls_name):
    import FujiShaderGPU.algorithms.tile_shared as tile_shared

    cls = getattr(tile_shared, cls_name)
    x = cp.random.random((64, 64), dtype=cp.float32)
    y = cls().process(x)

    assert y.shape == (64, 64)
    assert str(y.dtype) == 'float32'
