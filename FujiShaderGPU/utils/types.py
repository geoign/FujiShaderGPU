"""
FujiShaderGPU/utils/types.py
"""
from typing import Optional, NamedTuple

# Class that holds a tile processing result
class TileResult(NamedTuple):
    """Holds a tile processing result."""
    tile_y: int
    tile_x: int
    success: bool
    filename: Optional[str] = None
    error_message: Optional[str] = None
    skipped_reason: Optional[str] = None
