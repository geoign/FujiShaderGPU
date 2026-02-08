"""
FujiShaderGPU/config/system_config.py
"""

import logging
import math
import multiprocessing
from importlib.util import find_spec
from typing import List, Optional

import cupy as cp
import psutil
from osgeo import gdal

from ..config.gpu_config_manager import _gpu_config_manager

logger = logging.getLogger(__name__)

# Keep current non-exception behavior and silence GDAL 4.0 future warning.
gdal.DontUseExceptions()


def get_gpu_config(
    gpu_type: str = "auto",
    sigma: float = 10.0,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5,
    target_distances: Optional[List[float]] = None,
) -> dict:
    """GPU type and system specs aware runtime config."""
    sys_config = detect_optimal_system_config()

    if gpu_type == "auto":
        gpu_name = sys_config.get("gpu_name", "").upper()
        vram_gb = float(sys_config.get("vram_gb", 0.0))
        gpu_type = _gpu_config_manager.detect_gpu_type(vram_gb, gpu_name)
        print(f"GPU自動検出: {gpu_name} ({vram_gb:.1f}GB) -> {gpu_type}")

    preset = _gpu_config_manager.get_preset(gpu_type)

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
        "tile_size": preset["chunk_size"] * 2,
        "max_workers": min(6, sys_config["cpu_count"]),
        "padding": calculated_padding,
        "vram_monitor": gpu_type != "a100",
        "batch_size": 2 if gpu_type == "a100" else 1,
        "prefetch_tiles": 4 if gpu_type == "a100" else 2,
        "description": f"{preset.get('name', gpu_type.upper())} 最適化設定",
        "system_info": sys_config,
    }


def detect_optimal_system_config() -> dict:
    """Detect hardware and derive an optimization level."""
    config = {
        "cpu_count": multiprocessing.cpu_count(),
        "memory_gb": psutil.virtual_memory().total // (1024**3),
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

    print("システム検出結果:")
    print(f"  CPU: {config['cpu_count']}コア, RAM: {config['memory_gb']}GB")
    if config["gpu_detected"]:
        print(f"  GPU: {config['gpu_name']}, VRAM: {config['vram_gb']:.1f}GB")
    else:
        print("  GPU: 未検出 (CPU情報のみで継続)")
    print(f"  最適化レベル: {config['optimization_level']}")
    return config


def check_gdal_environment():
    """
    GDAL環境チェック (QGIS最適化対応)
    """
    print("=== GDAL環境チェック ===")

    gdal_version = gdal.VersionInfo()
    print(f"GDALバージョン: {gdal_version}")

    cog_driver = gdal.GetDriverByName("COG")
    print(f"COGドライバー: {'利用可能' if cog_driver else '利用不可'}")

    gtiff_driver = gdal.GetDriverByName("GTiff")
    print(f"GTiffドライバー: {'利用可能' if gtiff_driver else '利用不可'}")

    print("\nQGIS最適化:")
    print("  - 512x512ブロックサイズ")
    print("  - 多段階オーバービュー")
    print("  - AVERAGE リサンプリング")
    print("  - ZSTD圧縮")

    sys_config = detect_optimal_system_config()
    print(f"Platform: {sys_config['platform']}, GPU detected: {sys_config['gpu_detected']}")
    print("=" * 50)
