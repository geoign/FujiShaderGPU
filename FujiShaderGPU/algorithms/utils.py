"""
FujiShaderGPU/algorithms/utils.py
"""
import math

def calculate_padding(sigma: float, safety_factor: float = 5.0) -> int:
    """
    Gaussianブラーのσ値に基づいて適切なpadding値を計算
    """
    required_padding = int(math.ceil(sigma * safety_factor))
    min_padding = 16
    # 32の倍数に切り上げ（キャッシュライン最適化）
    padding = max(min_padding, ((required_padding + 31) // 32) * 32)
    return padding
