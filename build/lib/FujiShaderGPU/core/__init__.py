"""
FujiShaderGPU/core/__init__.py
コア処理モジュールの初期化
"""

# プラットフォーム別のインポート
import platform

__all__ = []

# 共通モジュール
try:
    from .gpu_memory import gpu_memory_pool  # noqa: F401
    __all__.append("gpu_memory_pool")
except ImportError:
    pass

# Windows/macOS向けモジュール
if platform.system().lower() != "linux":
    try:
        from .tile_processor import process_dem_tiles, resume_cog_generation  # noqa: F401
        __all__.extend(["process_dem_tiles", "resume_cog_generation"])
    except ImportError:
        pass

# Linux向けモジュール
else:
    try:
        from .dask_processor import run_pipeline, make_cluster  # noqa: F401
        __all__.extend(["run_pipeline", "make_cluster"])
    except ImportError:
        pass
