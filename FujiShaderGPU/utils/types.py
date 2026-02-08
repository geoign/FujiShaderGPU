"""
FujiShaderGPU/utils/types.py
"""
from typing import Optional, NamedTuple

# タイル処理結果を格納するクラス
class TileResult(NamedTuple):
    """タイル処理結果を格納"""
    tile_y: int
    tile_x: int
    success: bool
    filename: Optional[str] = None
    error_message: Optional[str] = None
    skipped_reason: Optional[str] = None
