"""
FujiShaderGPU/utils/nodata_handler.py
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

# scipyのインポート（フル活用）
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scipy が利用できません。一部の CPU フォールバックに切り替えます。"
    )

def _handle_nodata_ultra_fast(dem_tile: np.ndarray, mask_nodata: np.ndarray) -> np.ndarray:
    """
    超高速NoData処理
    """
    dem_processed = dem_tile.copy()
    nodata_ratio = np.count_nonzero(mask_nodata) / mask_nodata.size
    
    if nodata_ratio < 0.1:
        # 高速ゼロ置換
        dem_processed[mask_nodata] = 0.0
    elif nodata_ratio < 0.5 and SCIPY_AVAILABLE:
        # Scipy高速補間
        try:
            indices = distance_transform_edt(mask_nodata, return_distances=False, return_indices=True)
            dem_processed[mask_nodata] = dem_tile[tuple(indices[:, mask_nodata])]
        except Exception:
            # フォールバック
            valid_data = dem_tile[~mask_nodata]
            if len(valid_data) > 0:
                dem_processed[mask_nodata] = np.mean(valid_data)
            else:
                dem_processed[mask_nodata] = 0.0
    else:
        # 高速平均値補間
        valid_data = dem_tile[~mask_nodata]
        if len(valid_data) > 0:
            dem_processed[mask_nodata] = np.mean(valid_data)
        else:
            dem_processed[mask_nodata] = 0.0
    
    return dem_processed
