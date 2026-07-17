import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUConfigManager:
    """GPU type detection from VRAM / device name (feeds the run description label).

    Performance parameters are computed dynamically in ``config/auto_tune.py``;
    this class only classifies the detected GPU into a named preset bucket.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_config(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load settings from the YAML file."""
        config_path = Path(__file__).parent / "gpu_presets.yaml"
        with open(config_path, encoding="utf-8-sig") as f:
            self._config = yaml.safe_load(f)
    
    def detect_gpu_type(self, vram_gb: float, gpu_name: str = "") -> str:
        """Determine the GPU type."""
        self._ensure_config()
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
        self._ensure_config()
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

# Global instance
_gpu_config_manager = GPUConfigManager()
