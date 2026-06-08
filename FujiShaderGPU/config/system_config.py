"""
FujiShaderGPU/config/system_config.py
"""

import logging
import math
from importlib.util import find_spec
from typing import List, Optional

import cupy as cp
from osgeo import gdal

from ..config.gpu_config_manager import _gpu_config_manager
from ..config.auto_tune import auto_tune
from ..utils.memory import container_memory_total_gb
from ..utils.cpu import container_cpu_count

logger = logging.getLogger(__name__)

# Keep current non-exception behavior and silence GDAL 4.0 future warning.
gdal.DontUseExceptions()


def get_gpu_config(
    gpu_type: str = "auto",
    sigma: float = 10.0,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5,
    target_distances: Optional[List[float]] = None,
    algorithm: str = "",
) -> dict:
    """GPU type and system specs aware runtime config."""
    sys_config = detect_optimal_system_config()

    if gpu_type == "auto":
        gpu_name = sys_config.get("gpu_name", "").upper()
        vram_gb = float(sys_config.get("vram_gb", 0.0))
        gpu_type = _gpu_config_manager.detect_gpu_type(vram_gb, gpu_name)
        logger.info("GPU auto-detected: %s (%.1fGB) -> %s", gpu_name, vram_gb, gpu_type)

    # Dynamic parameter computation from VRAM
    vram_gb = float(sys_config.get("vram_gb", 8.0))
    cpu_count = int(sys_config.get("cpu_count", 4))
    is_colab_env = bool(sys_config.get("is_colab", False))
    tuned = auto_tune(
        vram_gb,
        algorithm=algorithm,
        cpu_count=cpu_count,
        is_colab=is_colab_env,
    )

    if multiscale_mode:
        if target_distances:
            max_distance = max(target_distances)
        else:
            max_distance = max([5.0, 25.0, 100.0, 200.0])
        max_sigma = max_distance / pixel_size
        required_padding = int(math.ceil(max_sigma * 5.0))
    else:
        required_padding = int(math.ceil(sigma * 5.0))

    min_padding = 32
    calculated_padding = max(min_padding, ((required_padding + 31) // 32) * 32)

    return {
        "tile_size": tuned["tile_size"],
        "max_workers": tuned["max_workers"],
        "padding": calculated_padding,
        "vram_monitor": vram_gb < 40,
        "batch_size": tuned["batch_size"],
        "prefetch_tiles": tuned["prefetch_tiles"],
        "description": f"{gpu_type.upper()} dynamic optimization (VRAM {vram_gb:.0f}GB)",
        "system_info": sys_config,
    }


def detect_optimal_system_config() -> dict:
    """Detect hardware and derive an optimization level."""
    config = {
        # Container-aware: cgroup CPU quota, not the host core count (avoids
        # over-sizing worker pools on RunPod/Colab/k8s where the host dwarfs
        # the usable slice).  Mirrors the memory_gb treatment below.
        "cpu_count": container_cpu_count(),
        # Container-aware: cgroup cap, not host RAM (avoids over-sizing on
        # RunPod/Colab/k8s where the host total dwarfs the usable limit).
        "memory_gb": int(container_memory_total_gb()),
        "gpu_detected": False,
        "gpu_name": "Unknown",
        "vram_gb": 0.0,
        "platform": "unknown",
        "is_colab": False,
    }

    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            config["gpu_detected"] = True
            gpu_props = cp.cuda.runtime.getDeviceProperties(0)
            config["gpu_name"] = gpu_props["name"].decode()
            config["vram_gb"] = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
            config["gpu_compute_capability"] = f"{gpu_props['major']}.{gpu_props['minor']}"
            config["gpu_multiprocessors"] = gpu_props["multiProcessorCount"]
    except (cp.cuda.runtime.CUDARuntimeError, AttributeError, RuntimeError) as exc:
        logger.debug("GPU detection failed, continuing with CPU-only metadata: %s", exc)

    try:
        config["is_colab"] = find_spec("google.colab") is not None
    except ModuleNotFoundError:
        config["is_colab"] = False
    config["platform"] = "colab" if config["is_colab"] else "local"

    vram_gb = float(config["vram_gb"])
    if vram_gb >= 40:
        config["optimization_level"] = "ultra"
    elif vram_gb >= 20:
        config["optimization_level"] = "high"
    elif vram_gb >= 14:
        config["optimization_level"] = "medium"
    elif vram_gb >= 8:
        config["optimization_level"] = "medium_high"
    else:
        config["optimization_level"] = "standard"

    logger.info("System detection results:")
    logger.info("  CPU: %s cores, RAM: %sGB", config['cpu_count'], config['memory_gb'])
    if config["gpu_detected"]:
        logger.info("  GPU: %s, VRAM: %.1fGB", config['gpu_name'], config['vram_gb'])
    else:
        logger.info("  GPU: not detected (continuing with CPU info only)")
    logger.info("  Optimization level: %s", config['optimization_level'])
    return config


def check_gdal_environment():
    """
    GDAL environment check (QGIS-optimization aware)
    """
    logger.info("=== GDAL environment check ===")

    gdal_version = gdal.VersionInfo()
    logger.info("GDAL version: %s", gdal_version)

    cog_driver = gdal.GetDriverByName("COG")
    logger.info("COG driver: %s", 'available' if cog_driver else 'unavailable')

    gtiff_driver = gdal.GetDriverByName("GTiff")
    logger.info("GTiff driver: %s", 'available' if gtiff_driver else 'unavailable')

    logger.info("QGIS optimization: 512x512 blocks / multi-level overviews / AVERAGE resampling / ZSTD compression")

    sys_config = detect_optimal_system_config()
    logger.info(
        "Platform: %s, GPU detected: %s",
        sys_config['platform'], sys_config['gpu_detected'],
    )
