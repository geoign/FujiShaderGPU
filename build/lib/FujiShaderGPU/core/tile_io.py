"""Tile pipeline raster I/O helpers."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window


def read_tile_window(input_cog_path: str, window: Window) -> np.ndarray:
    with rasterio.open(input_cog_path, 'r') as src:
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
