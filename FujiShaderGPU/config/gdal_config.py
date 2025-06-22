"""
FujiShaderGPU/config/gdal_config.py
"""
import os, logging
from osgeo import gdal
from ..config.gpu_config_manager import _gpu_config_manager

def _configure_gdal_ultra_performance(gpu_config: dict):
    """GDAL設定（システムメモリベース）"""
    sys_info = gpu_config["system_info"]
    
    # CPUメモリに基づいて設定（GPUではなく）
    cpu_memory_gb = sys_info["memory_gb"]
    
    if cpu_memory_gb >= 128:
        cache_mb = 32768  # 32GB
        dataset_pool_size = "5000"
    elif cpu_memory_gb >= 64:
        cache_mb = 16384  # 16GB
        dataset_pool_size = "3000"
    elif cpu_memory_gb >= 32:
        cache_mb = 8192   # 8GB
        dataset_pool_size = "2000"
    else:
        cache_mb = 4096   # 4GB
        dataset_pool_size = "1000"
    
    # スワスサイズは固定値で良い
    swath_multiplier = 1.0
    
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
        "GDAL高速化（%s）: キャッシュ%dMB, プール%d, HTTP/2有効",
        cache_mb,
        dataset_pool_size,
    )
