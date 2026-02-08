"""Dask pipeline I/O helpers (COG/Zarr)."""
from __future__ import annotations

from pathlib import Path
import warnings
import rasterio
import xarray as xr
import rioxarray as rxr
from dask.diagnostics import ProgressBar


def is_zarr_path(path: str) -> bool:
    return str(path).lower().endswith('.zarr')


def select_data_var(ds: xr.Dataset) -> str:
    for candidate in ('elevation', 'dem', 'band_data', 'data'):
        if candidate in ds.data_vars:
            return candidate
    if not ds.data_vars:
        raise ValueError('No data variables found in Zarr dataset.')
    return next(iter(ds.data_vars))


def load_input_dataarray(src_path: str, chunk: int) -> xr.DataArray:
    if is_zarr_path(src_path):
        try:
            import zarr  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Zarr input requires the `zarr` package. Install it with `pip install zarr`.'
            ) from exc
        ds_or_da = xr.open_zarr(src_path, chunks='auto')
        if isinstance(ds_or_da, xr.Dataset):
            da_in = ds_or_da[select_data_var(ds_or_da)]
        else:
            da_in = ds_or_da
        if 'band' in da_in.dims and da_in.sizes.get('band', 1) == 1:
            da_in = da_in.squeeze('band', drop=True)
        if {'y', 'x'}.issubset(set(da_in.dims)):
            return da_in.chunk({'y': chunk, 'x': chunk}).astype('float32')
        y_dim, x_dim = da_in.dims[-2], da_in.dims[-1]
        return da_in.chunk({y_dim: chunk, x_dim: chunk}).astype('float32')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
        return (
            rxr.open_rasterio(
                src_path,
                masked=True,
                chunks={'y': chunk, 'x': chunk},
                lock=False,
            )
            .squeeze()
            .astype('float32')
        )


def write_zarr_output(data: xr.DataArray, dst: Path, show_progress: bool = True):
    try:
        import zarr  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Zarr output requires the `zarr` package. Install it with `pip install zarr`.'
        ) from exc

    ds = data.to_dataset(name=(data.name or 'result'))
    if show_progress:
        with ProgressBar():
            ds.to_zarr(str(dst), mode='w', consolidated=True, zarr_format=2)
    else:
        ds.to_zarr(str(dst), mode='w', consolidated=True, zarr_format=2)
