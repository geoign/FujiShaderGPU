"""
FujiShaderGPU/config/gdal_config.py
"""
import os, logging
from osgeo import gdal

def _configure_gdal_ultra_performance(gpu_config: dict):
    """
    GDAL超高速設定（T4/L4対応）
    """
    sys_info = gpu_config["system_info"]
    memory_gb = sys_info["memory_gb"]
    gpu_name = sys_info.get("gpu_name", "").upper()
    
    # GPU別メモリ最適化
    if "A100" in gpu_name or memory_gb >= 64:
        cache_mb = 32768  # 32GB - A100環境
    elif "L4" in gpu_name or memory_gb >= 32:
        cache_mb = 20480  # 20GB - L4環境（24GB VRAM考慮）
    elif "T4" in gpu_name or memory_gb >= 24:
        cache_mb = 12288  # 12GB - T4環境（16GB VRAM考慮）
    elif memory_gb >= 16:
        cache_mb = 8192   # 8GB - RTX4070環境
    else:
        cache_mb = 4096   # 4GB - 最小構成
    
    # GPU性能に応じたワーカー数調整
    if "A100" in gpu_name:
        dataset_pool_size = "5000"
        swath_multiplier = 1.0
    elif "L4" in gpu_name:
        dataset_pool_size = "3000"
        swath_multiplier = 0.8
    elif "T4" in gpu_name:
        dataset_pool_size = "2000"
        swath_multiplier = 0.6
    else:  # RTX4070等
        dataset_pool_size = "1500"
        swath_multiplier = 0.5
    
    ultra_configs = {
        'GDAL_CACHEMAX': str(cache_mb),
        'GDAL_MAX_DATASET_POOL_SIZE': dataset_pool_size,
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'VSI_CACHE': 'YES',
        'VSI_CACHE_SIZE': str(cache_mb * 1024 * 1024),
        'GDAL_SWATH_SIZE': str(int(cache_mb * swath_multiplier)),
        'GDAL_FORCE_CACHING': 'YES',
        'GDAL_NUM_THREADS': 'ALL_CPUS',
        'GDAL_HTTP_MULTIPLEX': 'YES',
        'GDAL_HTTP_VERSION': '2',
        'CPL_VSIL_CURL_CACHE_SIZE': str(cache_mb * 1024 * 1024),
        'GDAL_BAND_BLOCK_CACHE': 'HASHSET',  # 最新のキャッシュ機構
        'GDAL_CACHEMAX_MEMORY_OPTIMIZATION': 'YES',
    }
    
    for key, value in ultra_configs.items():
        os.environ[key] = value
        gdal.SetConfigOption(key, value)
    
    logging.getLogger(__name__).info(
        "GDAL超高速化（%s）: キャッシュ%dMB, プール%d, HTTP/2有効",
        gpu_name or "CPU",
        cache_mb,
        dataset_pool_size,
    )
