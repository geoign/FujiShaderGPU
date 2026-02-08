"""Public algorithm package exports."""

__all__ = []

# Dask registry (canonical algorithm names)
try:
    from .dask_registry import ALGORITHMS, DaskAlgorithm  # noqa: F401
    __all__.extend(["ALGORITHMS", "DaskAlgorithm"])
except Exception:
    pass

# Tile-side classes (same canonical names as Dask)
try:
    from .tile_shared import TileAlgorithm  # noqa: F401
    from .tile.rvi import RVIAlgorithm  # noqa: F401
    from .tile.hillshade import HillshadeAlgorithm  # noqa: F401
    from .tile.slope import SlopeAlgorithm  # noqa: F401
    from .tile.specular import SpecularAlgorithm  # noqa: F401
    from .tile.atmospheric_scattering import AtmosphericScatteringAlgorithm  # noqa: F401
    from .tile.multiscale_terrain import MultiscaleDaskAlgorithm  # noqa: F401
    from .tile.curvature import CurvatureAlgorithm  # noqa: F401
    from .tile.visual_saliency import VisualSaliencyAlgorithm  # noqa: F401
    from .tile.npr_edges import NPREdgesAlgorithm  # noqa: F401
    from .tile.ambient_occlusion import AmbientOcclusionAlgorithm  # noqa: F401
    from .tile.lrm import LRMAlgorithm  # noqa: F401
    from .tile.openness import OpennessAlgorithm  # noqa: F401
    from .tile.fractal_anomaly import FractalAnomalyAlgorithm  # noqa: F401
    from .tile.scale_space_surprise import ScaleSpaceSurpriseAlgorithm  # noqa: F401
    from .tile.multi_light_uncertainty import MultiLightUncertaintyAlgorithm  # noqa: F401

    __all__.extend([
        "TileAlgorithm",
        "RVIAlgorithm",
        "HillshadeAlgorithm",
        "SlopeAlgorithm",
        "SpecularAlgorithm",
        "AtmosphericScatteringAlgorithm",
        "MultiscaleDaskAlgorithm",
        "CurvatureAlgorithm",
        "VisualSaliencyAlgorithm",
        "NPREdgesAlgorithm",
        "AmbientOcclusionAlgorithm",
        "LRMAlgorithm",
        "OpennessAlgorithm",
        "FractalAnomalyAlgorithm",
        "ScaleSpaceSurpriseAlgorithm",
        "MultiLightUncertaintyAlgorithm",
    ])
except Exception:
    pass
