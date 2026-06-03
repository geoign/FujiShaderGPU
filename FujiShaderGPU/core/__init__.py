"""
FujiShaderGPU/core/__init__.py
Core processing package initialization.
"""

# Platform-specific imports
import platform

__all__ = []

# Shared modules
try:
    from .gpu_memory import gpu_memory_pool  # noqa: F401
    __all__.append("gpu_memory_pool")
except ImportError:
    pass

# Windows/macOS modules
if platform.system().lower() != "linux":
    try:
        from .tile_processor import process_dem_tiles, resume_cog_generation  # noqa: F401
        __all__.extend(["process_dem_tiles", "resume_cog_generation"])
    except ImportError:
        pass

# Linux modules
else:
    try:
        from .dask_processor import run_pipeline, make_cluster  # noqa: F401
        __all__.extend(["run_pipeline", "make_cluster"])
    except ImportError:
        pass
