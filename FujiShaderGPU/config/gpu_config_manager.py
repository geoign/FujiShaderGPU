import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GPUConfigManager:
    """GPU設定の一元管理（最小実装）"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """YAMLファイルから設定を読み込み"""
        config_path = Path(__file__).parent / "gpu_presets.yaml"
        with open(config_path, encoding="utf-8-sig") as f:
            self._config = yaml.safe_load(f)
    
    def detect_gpu_type(self, vram_gb: float, gpu_name: str = "") -> str:
        """GPU種別を判定"""
        gpu_name_upper = gpu_name.upper()
        
        # 名前による判定（優先）
        if "H100" in gpu_name_upper:
            return "h100"
        elif "A100" in gpu_name_upper:
            # VRAMサイズで40GBと80GBを区別
            if vram_gb >= 70:
                return "a100_80gb"
            else:
                return "a100_40gb"
        elif "L4" in gpu_name_upper:
            return "l4"
        elif "T4" in gpu_name_upper:
            return "t4"
        elif "4070" in gpu_name_upper and ("LAPTOP" in gpu_name_upper or "MOBILE" in gpu_name_upper):
            return "rtx4070_laptop"
        
        # VRAMサイズによる判定
        for gpu_type, preset in self._config["gpu_presets"].items():
            if "vram_gb_range" in preset:
                min_vram, max_vram = preset["vram_gb_range"]
                if min_vram <= vram_gb < max_vram:
                    return gpu_type
        
        # 最も近いVRAMサイズのGPUを選択
        return self._find_closest_gpu_by_vram(vram_gb)
    
    def _find_closest_gpu_by_vram(self, vram_gb: float) -> str:
        """最も近いVRAMサイズのGPU設定を見つける"""
        closest_gpu = "t4"  # デフォルト
        min_diff = float('inf')
        
        for gpu_type, preset in self._config["gpu_presets"].items():
            gpu_vram = preset["vram_gb"]
            diff = abs(gpu_vram - vram_gb)
            if diff < min_diff:
                min_diff = diff
                closest_gpu = gpu_type
        
        logger.info(f"Unknown GPU with {vram_gb}GB VRAM, using {closest_gpu} preset")
        return closest_gpu
    
    def is_colab(self) -> bool:
        """Google Colab環境かどうかを判定"""
        import sys
        # より確実なColab検出
        if 'google.colab' in sys.modules:
            return True
        # 環境変数でも確認
        import os
        if os.getenv('COLAB_GPU', None):
            return True
        # /content ディレクトリの存在確認（Colab特有）
        if os.path.exists('/content') and os.path.isdir('/content'):
            return True
        return False
    
    def get_preset(self, gpu_type: str) -> Dict[str, Any]:
        """GPU種別のプリセットを取得（環境変数でオーバーライド）"""
        if gpu_type not in self._config["gpu_presets"]:
            logger.warning(f"Unknown GPU type: {gpu_type}, falling back to t4")
            gpu_type = "t4"
        
        preset = self._config["gpu_presets"][gpu_type].copy()
        
        # 環境変数によるオーバーライド
        if chunk_size := os.getenv("FUJISHADER_CHUNK_SIZE"):
            preset["chunk_size"] = int(chunk_size)
            logger.info(f"Overriding chunk_size to {chunk_size} from env")
            
        if rmm_size := os.getenv("FUJISHADER_RMM_POOL_GB"):
            preset["rmm_pool_size_gb"] = int(rmm_size)
            logger.info(f"Overriding rmm_pool_size_gb to {rmm_size} from env")
        
        return preset
    
    def get_environment_config(self, env_type: str) -> Dict[str, Any]:
        """環境別の設定を取得"""
        env_configs = {
            "colab": {
                "memory_fraction_limit": 0.5,
                "death_timeout": "60s",
                "interface": "lo"
            },
            "chunk_threshold": {
                "colab_min": 20,
                "colab_max": 60,
                "local_min": 30,
                "local_max": 100
            }
        }
        return env_configs.get(env_type, {})
    
    def get_preset_for_unknown_gpu(self, vram_gb: float) -> Dict[str, Any]:
        """未知のGPU用の設定を動的に生成"""
        # 最も近い既知のGPUを基準にする
        base_gpu = self._find_closest_gpu_by_vram(vram_gb)
        base_preset = self._config["gpu_presets"][base_gpu].copy()
        
        # VRAMサイズに基づいてパラメータを調整
        vram_ratio = vram_gb / base_preset["vram_gb"]
        
        # チャンクサイズを調整（512の倍数に丸める）
        adjusted_chunk = int(base_preset["chunk_size"] * (vram_ratio ** 0.5))
        adjusted_chunk = max(512, (adjusted_chunk // 512) * 512)
        
        # RMMプールサイズを調整
        adjusted_rmm = int(base_preset["rmm_pool_size_gb"] * vram_ratio)
        
        # 新しいプリセットを作成
        new_preset = {
            "vram_gb": vram_gb,
            "chunk_size": adjusted_chunk,
            "rmm_pool_size_gb": adjusted_rmm,
            "rmm_pool_fraction": base_preset["rmm_pool_fraction"],
        }
        
        logger.info(f"Generated preset for unknown GPU ({vram_gb}GB): "
                   f"chunk_size={adjusted_chunk}, rmm_pool={adjusted_rmm}GB")
        
        return new_preset
    
    def get_algorithm_complexity(self, algorithm: str) -> float:
        """アルゴリズムの複雑度を取得"""
        return self._config["algorithm_complexity"].get(algorithm, 1.0)
    
    def get_system_preset(self, memory_gb: int) -> Dict[str, Any]:
        """システムメモリに基づいたGDAL設定を取得"""
        gdal_config = self._config["system_presets"]["gdal"]
        
        if memory_gb >= 64:
            return gdal_config["high_memory"]
        elif memory_gb >= 32:
            return gdal_config["medium_memory"]
        else:
            return gdal_config["low_memory"]

# グローバルインスタンス
_gpu_config_manager = GPUConfigManager()
