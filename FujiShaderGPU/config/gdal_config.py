"""
FujiShaderGPU/config/gdal_config.py
"""

import logging
import os

from osgeo import gdal


def _configure_gdal_ultra_performance(gpu_config: dict):
    """Tune GDAL I/O options based on available system RAM."""
    sys_info = gpu_config["system_info"]
    cpu_memory_gb = int(sys_info["memory_gb"])

    if cpu_memory_gb >= 128:
        cache_mb = 32768
        dataset_pool_size = 5000
    elif cpu_memory_gb >= 64:
        cache_mb = 16384
        dataset_pool_size = 3000
    elif cpu_memory_gb >= 32:
        cache_mb = 8192
        dataset_pool_size = 2000
    else:
        cache_mb = 4096
        dataset_pool_size = 1000

    swath_multiplier = 1.0
    cache_bytes = cache_mb * 1024 * 1024

    ultra_configs = {
        "GDAL_CACHEMAX": str(cache_mb),
        "GDAL_MAX_DATASET_POOL_SIZE": str(dataset_pool_size),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "YES",
        "VSI_CACHE_SIZE": str(cache_bytes),
        "GDAL_SWATH_SIZE": str(int(cache_mb * swath_multiplier)),
        "GDAL_FORCE_CACHING": "YES",
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "CPL_VSIL_CURL_CACHE_SIZE": str(cache_bytes),
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
        "GDAL_CACHEMAX_MEMORY_OPTIMIZATION": "YES",
    }

    for key, value in ultra_configs.items():
        os.environ[key] = value
        gdal.SetConfigOption(key, value)

    logging.getLogger(__name__).info(
        "GDAL settings applied: cache=%dMB, dataset_pool=%d, HTTP/2 enabled",
        cache_mb,
        dataset_pool_size,
    )
