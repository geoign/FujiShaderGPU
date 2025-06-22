"""
FujiShaderGPU/config/system_config.py
"""

import math, multiprocessing, psutil, logging
import cupy as cp
from typing import Optional, List
from osgeo import gdal
from ..config.gpu_config_manager import _gpu_config_manager

# ロギング設定
logger = logging.getLogger(__name__)

def get_gpu_config(gpu_type: str = "auto", sigma: float = 10.0, multiscale_mode: bool = True, 
                   pixel_size: float = 0.5, target_distances: Optional[List[float]] = None) -> dict:
    """GPU種別に応じた最適設定を取得（T4/L4対応安定版）"""
    sys_config = detect_optimal_system_config()
    
    if gpu_type == "auto":
        gpu_name = sys_config.get("gpu_name", "").upper()
        vram_gb = sys_config["vram_gb"]
        gpu_type = _gpu_config_manager.detect_gpu_type(vram_gb, gpu_name)
        print(f"GPU自動検出: {gpu_name} ({vram_gb:.1f}GB) → {gpu_type}設定")
    
    # プリセットから設定を取得
    preset = _gpu_config_manager.get_preset(gpu_type)
    
    # σ値とマルチスケール設定に基づくpadding計算
    if multiscale_mode:
        if target_distances is not None:
            max_distance = max(target_distances)
            max_sigma = max_distance / pixel_size
        else:
            default_distances = [5.0, 25.0, 100.0, 200.0]
            max_distance = max(default_distances)
            max_sigma = max_distance / pixel_size
        required_padding = int(math.ceil(max_sigma * 5.0))
    else:
        required_padding = int(math.ceil(sigma * 5.0))
    
    min_padding = 32
    calculated_padding = max(min_padding, ((required_padding + 31) // 32) * 32)
    
    # 既存の形式に変換（互換性のため）
    config = {
        "tile_size": preset["chunk_size"] * 2,  # tile_sizeはchunk_sizeの2倍として扱う
        "max_workers": min(6, sys_config["cpu_count"]),
        "padding": calculated_padding,
        "vram_monitor": gpu_type != "a100",
        "batch_size": 2 if gpu_type == "a100" else 1,
        "prefetch_tiles": 4 if gpu_type == "a100" else 2,
        "description": f"{preset.get('name', gpu_type.upper())} 最適化設定",
        "system_info": sys_config,
    }
    
    return config


def detect_optimal_system_config() -> dict:
    """
    システム環境を詳細に分析して最適な設定を決定（安定版）
    """
    config = {
        "cpu_count": multiprocessing.cpu_count(),
        "memory_gb": psutil.virtual_memory().total // (1024**3),
        "gpu_detected": False,
        "gpu_name": "Unknown",
        "vram_gb": 0,
        "platform": "unknown"
    }
    
    # GPU詳細検出
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            config["gpu_detected"] = True
            gpu_props = cp.cuda.runtime.getDeviceProperties(0)
            config["gpu_name"] = gpu_props['name'].decode()
            config["vram_gb"] = cp.cuda.runtime.memGetInfo()[1] / (1024**3)
            config["gpu_compute_capability"] = f"{gpu_props['major']}.{gpu_props['minor']}"
            config["gpu_multiprocessors"] = gpu_props['multiProcessorCount']
    except (cp.cuda.runtime.CUDARuntimeError, AttributeError) as e:
        logger.debug(f"GPU detection failed: {e}")
        raise RuntimeError("GPU detection failed")
    
    # Google Colab検出
    try:
        import google.colab
        config["platform"] = "colab"
        config["is_colab"] = True
    except ImportError:
        config["platform"] = "local"
        config["is_colab"] = False
    
    # 最適化レベル決定（T4/L4対応）
    if config["vram_gb"] >= 40:  # A100クラス
        config["optimization_level"] = "ultra"
    elif config["vram_gb"] >= 20:  # L4クラス
        config["optimization_level"] = "high"  # very_high → high に下げる
    elif config["vram_gb"] >= 14:  # T4クラス
        config["optimization_level"] = "medium"  # high → medium に下げる
    elif config["vram_gb"] >= 8:   # RTX4070クラス
        config["optimization_level"] = "medium_high"
    else:
        config["optimization_level"] = "standard"
    
    print(f"システム構成検出:")
    print(f"  CPU: {config['cpu_count']}コア, RAM: {config['memory_gb']}GB")
    if config["gpu_detected"]:
        print(f"  GPU: {config['gpu_name']}, VRAM: {config['vram_gb']:.1f}GB")
        print(f"  最適化レベル: {config['optimization_level']}")
    
    return config


def check_gdal_environment():
    """
    GDAL環境チェック（QGIS最適化対応）
    """
    print("=== GDAL環境チェック ===")
    
    gdal_version = gdal.VersionInfo()
    print(f"GDALバージョン: {gdal_version}")
    
    cog_driver = gdal.GetDriverByName("COG")
    print(f"COGドライバー: {'✅ 利用可能' if cog_driver else '❌ 利用不可'}")
    
    gtiff_driver = gdal.GetDriverByName("GTiff")
    print(f"GTiffドライバー: {'✅ 利用可能' if gtiff_driver else '❌ 利用不可'}")
    
    # QGIS最適化情報
    print("\n🎯 QGIS最適化:")
    print("   - 512x512ブロックサイズ")
    print("   - 多段階オーバービュー（2-512レベル）")
    print("   - AVERAGE リサンプリング")
    print("   - ZSTD高速圧縮")
    
    # システム構成表示
    sys_config = detect_optimal_system_config()
    print("=" * 50)