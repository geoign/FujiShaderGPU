"""
FujiShaderGPU/config/gdal_config.py
"""

import logging
import os
from contextlib import contextmanager

from osgeo import gdal

from ..utils.cpu import container_cpu_count


@contextmanager
def gdal_local_no_exceptions():
    """Temporarily select GDAL's non-exception (None-returning) mode, restoring the
    caller's prior policy on exit.

    FujiShaderGPU's GDAL helpers check return values (``None`` on failure) rather
    than catching exceptions, and GDAL 4.0 emits a FutureWarning unless the policy
    is chosen explicitly.  Selecting it *globally at import time* (the previous
    ``gdal.DontUseExceptions()`` at module scope) silently changed GDAL's behaviour
    for any application that merely imported FujiShaderGPU.  Each GDAL entry point
    instead opts in locally via this context manager (usable as a decorator), so
    importing the package no longer mutates process-wide GDAL state.
    """
    get_use = getattr(gdal, "GetUseExceptions", None)
    prev = bool(get_use()) if callable(get_use) else False
    gdal.DontUseExceptions()
    try:
        yield
    finally:
        if prev:
            gdal.UseExceptions()
        else:
            gdal.DontUseExceptions()


def apply_gdal_io_config(cache_mb: int, *, dataset_pool_size: int = None, force: bool = True) -> None:
    """Apply the shared, container-aware GDAL I/O tuning env used by BOTH backends.

    Single source of truth so the dask and tile pipelines do not drift on GDAL
    settings.  ``force=True`` overwrites existing env (tile pipeline); ``force=
    False`` uses ``setdefault`` so a user-set env is respected (dask read path).
    ``GDAL_NUM_THREADS`` is cgroup-aware (``ALL_CPUS`` ignores the CFS quota and
    oversubscribes throttled containers)."""
    cache_mb = int(max(256, cache_mb))
    if dataset_pool_size is None:
        dataset_pool_size = 2000 if cache_mb >= 8192 else 1000
    cache_bytes = cache_mb * 1024 * 1024
    opts = {
        "GDAL_CACHEMAX": str(cache_mb),
        "GDAL_MAX_DATASET_POOL_SIZE": str(dataset_pool_size),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "YES",
        "VSI_CACHE_SIZE": str(cache_bytes),
        # Bytes, like the VSI caches (a bare MB figure here meant a 4-32 KB
        # swath, crippling GDALDatasetCopyWholeRaster during COG builds).
        "GDAL_SWATH_SIZE": str(cache_bytes),
        "GDAL_FORCE_CACHING": "YES",
        "GDAL_NUM_THREADS": str(max(1, container_cpu_count())),
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "CPL_VSIL_CURL_CACHE_SIZE": str(cache_bytes),
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
        "GDAL_CACHEMAX_MEMORY_OPTIMIZATION": "YES",
    }
    for key, value in opts.items():
        if force:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
        try:
            gdal.SetConfigOption(key, os.environ.get(key, value))
        except Exception:
            pass


def _configure_gdal_ultra_performance(gpu_config: dict):
    """Tune GDAL I/O options based on available system RAM (tile pipeline)."""
    sys_info = gpu_config["system_info"]
    cpu_memory_gb = int(sys_info["memory_gb"])

    if cpu_memory_gb >= 128:
        cache_mb, dataset_pool_size = 32768, 5000
    elif cpu_memory_gb >= 64:
        cache_mb, dataset_pool_size = 16384, 3000
    elif cpu_memory_gb >= 32:
        cache_mb, dataset_pool_size = 8192, 2000
    else:
        cache_mb, dataset_pool_size = 4096, 1000

    apply_gdal_io_config(cache_mb, dataset_pool_size=dataset_pool_size, force=True)

    logging.getLogger(__name__).info(
        "GDAL settings applied: cache=%dMB, dataset_pool=%d, HTTP/2 enabled",
        cache_mb,
        dataset_pool_size,
    )
