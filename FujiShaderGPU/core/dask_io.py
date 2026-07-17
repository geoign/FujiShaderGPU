"""Dask pipeline I/O helpers (COG/Zarr)."""
from __future__ import annotations

from pathlib import Path
import logging
import warnings
import rasterio
import xarray as xr
import rioxarray as rxr
from distributed import get_client, progress

logger = logging.getLogger(__name__)


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
        if da_in.ndim != 2:
            raise ValueError(
                f"DEM input must be exactly 2D after selecting a data variable; "
                f"got dims={da_in.dims}, shape={da_in.shape}"
            )
        if getattr(da_in.rio, "crs", None) is None:
            logger.warning(
                "Zarr input has no CRS metadata; GeoTIFF output will be ungeoreferenced. "
                "Store a spatial_ref/CRS coordinate in the Zarr source to preserve georeferencing."
            )
        if {'y', 'x'}.issubset(set(da_in.dims)):
            return da_in.chunk({'y': chunk, 'x': chunk}).astype('float32')
        y_dim, x_dim = da_in.dims[-2], da_in.dims[-1]
        return da_in.chunk({y_dim: chunk, x_dim: chunk}).astype('float32')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
        da_in = rxr.open_rasterio(
            src_path,
            masked=True,
            chunks={'y': chunk, 'x': chunk},
            lock=False,
        )
        if "band" in da_in.dims and da_in.sizes.get("band") == 1:
            da_in = da_in.squeeze("band", drop=True)
        if da_in.ndim != 2:
            raise ValueError(
                f"DEM input must contain exactly one 2D band; got dims={da_in.dims}, "
                f"shape={da_in.shape}"
            )
        return da_in.astype("float32")


def write_zarr_output(data: xr.DataArray, dst: Path, show_progress: bool = True):
    try:
        import zarr  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Zarr output requires the `zarr` package. Install it with `pip install zarr`.'
        ) from exc

    ds = data.to_dataset(name=(data.name or 'result'))
    delayed = ds.to_zarr(
        str(dst), mode='w', consolidated=True, zarr_format=2, compute=False,
    )
    try:
        client = get_client()
    except ValueError:
        # Library use outside run_pipeline has no distributed client.
        delayed.compute()
    else:
        future = client.compute(delayed)
        if show_progress:
            progress(future, interval="1s")
        future.result()
