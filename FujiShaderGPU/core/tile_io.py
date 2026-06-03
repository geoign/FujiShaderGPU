"""Tile pipeline raster I/O helpers."""
from __future__ import annotations

import threading
import numpy as np
import rasterio
from rasterio.windows import Window

_thread_local = threading.local()


def _get_thread_reader(input_cog_path: str):
    """Return a per-thread rasterio reader to avoid per-tile open/close churn."""
    reader = getattr(_thread_local, "reader", None)
    reader_path = getattr(_thread_local, "reader_path", None)
    if reader is not None and reader_path == input_cog_path and not reader.closed:
        return reader
    if reader is not None:
        try:
            reader.close()
        except Exception:
            pass
    reader = rasterio.open(input_cog_path, "r")
    _thread_local.reader = reader
    _thread_local.reader_path = input_cog_path
    return reader


def read_tile_window(input_cog_path: str, window: Window) -> np.ndarray:
    src = _get_thread_reader(input_cog_path)
    return src.read(1, window=window, out_dtype=np.float32)


def write_tile_output(tile_filename: str, result_core: np.ndarray, tile_profile: dict):
    with rasterio.open(tile_filename, 'w', **tile_profile) as dst:
        if result_core.ndim == 2:
            dst.write(result_core, 1)
            return

        if result_core.ndim == 3:
            # HxWxC -> CxHxW for rasterio
            if result_core.shape[-1] == tile_profile.get("count", result_core.shape[-1]):
                dst.write(np.moveaxis(result_core, -1, 0))
                return
            # Already band-first
            if result_core.shape[0] == tile_profile.get("count", result_core.shape[0]):
                dst.write(result_core)
                return

        raise ValueError(
            f"Unsupported tile array shape {result_core.shape} for profile count={tile_profile.get('count')}"
        )
