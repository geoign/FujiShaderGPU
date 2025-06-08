"""
FujiShaderGPU/config/system_config.py
"""

import math, multiprocessing, psutil
import cupy as cp
from typing import Optional, List
from osgeo import gdal

def get_gpu_config(gpu_type: str = "auto", sigma: float = 10.0, multiscale_mode: bool = True, pixel_size: float = 0.5, target_distances: Optional[List[float]] = None) -> dict:
    """
    GPU種別に応じた最適設定を取得（T4/L4対応安定版）
    """
    sys_config = detect_optimal_system_config()
    
    if gpu_type == "auto":
        gpu_name = sys_config.get("gpu_name", "").upper()
        vram_gb = sys_config["vram_gb"]
        
        # GPU名による詳細判定
        if "A100" in gpu_name or vram_gb >= 40:
            gpu_type = "a100"
        elif "L4" in gpu_name or (vram_gb >= 20 and vram_gb < 32):
            gpu_type = "l4"
        elif "T4" in gpu_name or (vram_gb >= 14 and vram_gb < 20):
            gpu_type = "t4"
        elif "RTX 4070" in gpu_name or (vram_gb >= 8 and vram_gb < 14):
            gpu_type = "rtx4070"
        else:
            gpu_type = "rtx4070"  # 安全側設定
            
        print(f"GPU自動検出: {gpu_name} ({vram_gb:.1f}GB) → {gpu_type}設定")
    
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
    
    # 安定版設定（T4/L4のメモリ使用量を削減）
    configs = {
        "rtx4070": {
            "tile_size": 4096,      # 大幅増量（2048→4096）
            "max_workers": min(6, sys_config["cpu_count"]),  # CPU数に応じて調整
            "padding": calculated_padding,
            "vram_monitor": False,
            "batch_size": 2,        # 複数タイル同時処理
            "prefetch_tiles": 4,    # プリフェッチ数
            "description": "RTX 4070 超高速最適化（4K tiles）"
        },
        "t4": {
            "tile_size": 2048,      # T4: 安定性重視（3072→2048）
            "max_workers": min(6, sys_config["cpu_count"]),  # ワーカー数も削減
            "padding": calculated_padding,
            "vram_monitor": True,   # メモリ監視を有効化
            "batch_size": 1,        # バッチサイズを最小化
            "prefetch_tiles": 2,    # プリフェッチを大幅削減
            "description": "Tesla T4 安定版（16GB VRAM、2K tiles）"
        },
        "l4": {
            "tile_size": 4096,      # L4: 安定性重視（6144→4096）
            "max_workers": min(8, sys_config["cpu_count"]),  # ワーカー数も調整
            "padding": calculated_padding,
            "vram_monitor": True,   # メモリ監視を維持
            "batch_size": 2,        # バッチサイズを削減
            "prefetch_tiles": 4,    # プリフェッチを半減
            "description": "L4 安定版（24GB VRAM、4K tiles）"
        },
        "a100": {
            "tile_size": 8192,      # A100は調整済みで問題なし
            "max_workers": min(16, sys_config["cpu_count"]),
            "padding": calculated_padding,
            "vram_monitor": True,
            "batch_size": 4,        # より多くのタイル同時処理
            "prefetch_tiles": 8,    # 大量プリフェッチ
            "description": "A100 超高速最適化（8K tiles + batch処理）"
        }
    }
    
    config = configs.get(gpu_type, configs["rtx4070"])
    config["system_info"] = sys_config
    
    # Google Colab環境での追加調整
    if sys_config.get("is_colab", False):
        if gpu_type == "t4":
            config["tile_size"] = min(config["tile_size"], 2048)
            config["batch_size"] = 1
            print("⚠️ Google Colab T4環境: メモリ制限のため設定を調整")
        elif gpu_type == "l4":
            config["tile_size"] = min(config["tile_size"], 4096)
            config["batch_size"] = min(config["batch_size"], 2)
            print("⚠️ Google Colab L4環境: メモリ制限のため設定を調整")
    
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
    except:
        pass
    
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