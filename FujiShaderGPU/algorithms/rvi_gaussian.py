"""
FujiShaderGPU/algorithms/rvi_gaussian.py
"""
from typing import List
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage

def _compute_multiscale_rvi_ultra_fast(dem_gpu: cp.ndarray, target_distances: List[float], weights: List[float], pixel_size: float) -> cp.ndarray:
    """
    超高速マルチスケールRVI計算（バッチ処理対応）
    """
    sigma_values = [max(0.5, dist / pixel_size) for dist in target_distances]
    
    # 結果初期化
    rvi_combined = cp.zeros_like(dem_gpu, dtype=cp.float32)
    
    # バッチ処理でメモリ効率向上
    for sigma, weight in zip(sigma_values, weights):
        # 最適化されたGaussianフィルタ
        dem_blur = cpx_ndimage.gaussian_filter(
            dem_gpu, sigma=sigma, mode="nearest", truncate=4.0
        )
        rvi_scale = dem_gpu - dem_blur
        
        # インプレース演算でメモリ節約
        rvi_combined += weight * rvi_scale
        
        del dem_blur, rvi_scale
    
    return rvi_combined

