"""Tile pipeline raster I/O helpers."""
from __future__ import annotations

import threading
import atexit
import numpy as np
import rasterio
from rasterio.windows import Window

_thread_local = threading.local()
_readers_lock = threading.Lock()
_open_readers = {}


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
    with _readers_lock:
        _open_readers[id(reader)] = reader
    return reader


def close_all_tile_readers() -> None:
    """Close cached per-thread readers explicitly at executor/pipeline shutdown."""
    with _readers_lock:
        readers = list(_open_readers.values())
        _open_readers.clear()
    for reader in readers:
        try:
            reader.close()
        except Exception:
            pass


atexit.register(close_all_tile_readers)


def read_tile_window(input_cog_path: str, window: Window) -> np.ndarray:
    src = _get_thread_reader(input_cog_path)
    return src.read(1, window=window, out_dtype=np.float32)


def write_tile_output(tile_filename: str, result_core: np.ndarray, tile_profile: dict):
    with rasterio.open(tile_filename, 'w', **tile_profile) as dst:
        if result_core.ndim == 2:
            dst.write(result_core, 1)
            return

        if result_core.ndim == 3:
            # Decide the layout from the known raster (h, w) in the profile, not
            # from the band count alone: a (C,H,W) stack whose edge tile is 3 or
            # 4 px wide has shape[-1] == count and was transposed as HxWxC.
            count = tile_profile.get("count")
            hw = (tile_profile.get("height"), tile_profile.get("width"))
            if None not in hw:
                hw = (int(hw[0]), int(hw[1]))
                if tuple(result_core.shape[-2:]) == hw and (
                        count is None or result_core.shape[0] == count):
                    # Already band-first.
                    dst.write(result_core)
                    return
                if tuple(result_core.shape[:2]) == hw and (
                        count is None or result_core.shape[-1] == count):
                    # HxWxC -> CxHxW for rasterio
                    dst.write(np.moveaxis(result_core, -1, 0))
                    return
            else:
                # No size in the profile: fall back to the band-count check,
                # band-first (the formal stack contract) first.
                if result_core.shape[0] == tile_profile.get("count", result_core.shape[0]):
                    dst.write(result_core)
                    return
                if result_core.shape[-1] == tile_profile.get("count", result_core.shape[-1]):
                    dst.write(np.moveaxis(result_core, -1, 0))
                    return

        raise ValueError(
            f"Unsupported tile array shape {result_core.shape} for profile count={tile_profile.get('count')}"
        )
