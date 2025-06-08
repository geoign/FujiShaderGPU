"""
FujiShaderGPU/algorithms/__init__.py
アルゴリズムパッケージの初期化
"""
__all__ = []

# プラットフォームによって利用可能なアルゴリズムを変える
import platform

if platform.system().lower() == "linux":
    # Linux: dask_algorithmsから全アルゴリズムをインポート
    try:
        from .dask_algorithms import ALGORITHMS, DaskAlgorithm
        __all__.extend(["ALGORITHMS", "DaskAlgorithm"])
    except ImportError:
        pass
else:
    # Windows/macOS: tile_algorithmsから個別アルゴリズムをインポート
    try:
        from .tile_algorithms import (
            TileAlgorithm,
            RVIGaussianAlgorithm,
            HillshadeAlgorithm,
            AtmosphericScatteringAlgorithm,
            CompositeTerrainAlgorithm,
            CurvatureAlgorithm,
            FrequencyEnhancementAlgorithm,
            VisualSaliencyAlgorithm
        )
        __all__.extend([
            "TileAlgorithm",
            "RVIGaussianAlgorithm",
            "HillshadeAlgorithm",
            "AtmosphericScatteringAlgorithm",
            "CompositeTerrainAlgorithm",
            "CurvatureAlgorithm",
            "FrequencyEnhancementAlgorithm",
            "VisualSaliencyAlgorithm"
        ])
    except ImportError:
        pass