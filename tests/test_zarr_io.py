import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

xr = pytest.importorskip('xarray')
pytest.importorskip('zarr')

from FujiShaderGPU.core.dask_io import is_zarr_path, load_input_dataarray, write_zarr_output  # noqa: E402


def test_zarr_path_detection():
    assert is_zarr_path('a.zarr')
    assert not is_zarr_path('a.tif')


def test_zarr_roundtrip(tmp_path):
    src = tmp_path / 'input.zarr'
    out = tmp_path / 'output.zarr'

    arr = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=('y', 'x'),
        name='dem',
    )
    arr.to_dataset(name='dem').to_zarr(src, mode='w', zarr_format=2)

    loaded = load_input_dataarray(str(src), chunk=2)
    assert loaded.shape == (2, 2)

    write_zarr_output(loaded, out, show_progress=False)
    ds = xr.open_zarr(out)
    assert 'dem' in ds.data_vars or 'result' in ds.data_vars
