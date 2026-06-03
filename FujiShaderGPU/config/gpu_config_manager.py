import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    from .auto_tune import ALGORITHM_COMPLEXITY as _AUTO_COMPLEXITY
except ImportError:
    _AUTO_COMPLEXITY = None

class GPUConfigManager:
    """Centralized GPU configuration management (minimal implementation)."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load settings from the YAML file."""
        config_path = Path(__file__).parent / "gpu_presets.yaml"
        with open(config_path, encoding="utf-8-sig") as f:
            self._config = yaml.safe_load(f)
    
    def detect_gpu_type(self, vram_gb: float, gpu_name: str = "") -> str:
        """Determine the GPU type."""
        gpu_name_upper = gpu_name.upper()
        
        # Decide by name (preferred)
        if "H100" in gpu_name_upper:
            return "h100"
        elif "A100" in gpu_name_upper:
            # Distinguish 40GB and 80GB by VRAM size
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
        
        # Decide by VRAM size
        for gpu_type, preset in self._config["gpu_presets"].items():
            if "vram_gb_range" in preset:
                min_vram, max_vram = preset["vram_gb_range"]
                if min_vram <= vram_gb < max_vram:
                    return gpu_type
        
        # Select the GPU with the closest VRAM size
        return self._find_closest_gpu_by_vram(vram_gb)
    
    def _find_closest_gpu_by_vram(self, vram_gb: float) -> str:
        """Find the GPU config with the closest VRAM size."""
        closest_gpu = "t4"  # default
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
        """Determine whether this is a Google Colab environment."""
        import sys
        # More reliable Colab detection
        if 'google.colab' in sys.modules:
            return True
        # Also check environment variables
        import os
        if os.getenv('COLAB_GPU', None):
            return True
        # Check for the /content directory (Colab-specific)
        if os.path.exists('/content') and os.path.isdir('/content'):
            return True
        return False
    
    def get_preset(self, gpu_type: str) -> Dict[str, Any]:
        """Get the preset for the GPU type (overridable via environment variables)."""
        if gpu_type not in self._config["gpu_presets"]:
            logger.warning(f"Unknown GPU type: {gpu_type}, falling back to t4")
            gpu_type = "t4"
        
        preset = self._config["gpu_presets"][gpu_type].copy()
        
        # Override via environment variables
        if chunk_size := os.getenv("FUJISHADER_CHUNK_SIZE"):
            preset["chunk_size"] = int(chunk_size)
            logger.info(f"Overriding chunk_size to {chunk_size} from env")
            
        if rmm_size := os.getenv("FUJISHADER_RMM_POOL_GB"):
            preset["rmm_pool_size_gb"] = int(rmm_size)
            logger.info(f"Overriding rmm_pool_size_gb to {rmm_size} from env")
        
        return preset
    
    def get_environment_config(self, env_type: str) -> Dict[str, Any]:
        """Get environment-specific settings."""
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
        """Dynamically generate settings for an unknown GPU."""
        # Base it on the closest known GPU
        base_gpu = self._find_closest_gpu_by_vram(vram_gb)
        base_preset = self._config["gpu_presets"][base_gpu].copy()
        
        # Adjust parameters based on VRAM size
        vram_ratio = vram_gb / base_preset["vram_gb"]
        
        # Adjust chunk size (round to a multiple of 512)
        adjusted_chunk = int(base_preset["chunk_size"] * (vram_ratio ** 0.5))
        adjusted_chunk = max(512, (adjusted_chunk // 512) * 512)
        
        # Adjust the RMM pool size
        adjusted_rmm = int(base_preset["rmm_pool_size_gb"] * vram_ratio)
        
        # Create a new preset
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
        """Get algorithm complexity (delegated to auto_tune.py)."""
        if _AUTO_COMPLEXITY is not None:
            return _AUTO_COMPLEXITY.get(algorithm, 1.0)
        return self._config["algorithm_complexity"].get(algorithm, 1.0)
    
    def get_system_preset(self, memory_gb: int) -> Dict[str, Any]:
        """Get GDAL settings based on system memory."""
        gdal_config = self._config["system_presets"]["gdal"]
        
        if memory_gb >= 64:
            return gdal_config["high_memory"]
        elif memory_gb >= 32:
            return gdal_config["medium_memory"]
        else:
            return gdal_config["low_memory"]

# Global instance
_gpu_config_manager = GPUConfigManager()
