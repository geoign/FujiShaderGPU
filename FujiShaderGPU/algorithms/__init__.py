"""Public algorithm package exports."""

import logging

logger = logging.getLogger(__name__)

__all__ = []

# Dask registry (canonical algorithm names).
# The catch stays broad because the backends pull in optional, environment-
# specific deps (cupy/CUDA, dask) that can fail in ways other than ImportError;
# importing the package must still succeed and expose whatever *did* load. But it
# now logs the cause instead of swallowing it silently (a typo'd import or a
# circular import used to look like "the name just doesn't exist").
try:
    from .dask_registry import ALGORITHMS, DaskAlgorithm  # noqa: F401
    __all__.extend(["ALGORITHMS", "DaskAlgorithm"])
except Exception as exc:
    logger.warning("FujiShaderGPU: Dask algorithm registry unavailable: %s", exc)

# Tile-side classes (same canonical names as Dask)
try:
    from .tile_shared import TileAlgorithm  # noqa: F401
    from .tile.topousm_fast import TopoUSMFastAlgorithm  # noqa: F401
    from .tile.hillshade import HillshadeAlgorithm  # noqa: F401
    from .tile.slope import SlopeAlgorithm  # noqa: F401
    from .tile.specular import SpecularAlgorithm  # noqa: F401
    from .tile.atmospheric_scattering import AtmosphericScatteringAlgorithm  # noqa: F401
    from .tile.multiscale_terrain import MultiscaleDaskAlgorithm  # noqa: F401
    from .tile.blur import BlurAlgorithm  # noqa: F401
    from .tile.curvature import CurvatureAlgorithm  # noqa: F401
    from .tile.visual_saliency import VisualSaliencyAlgorithm  # noqa: F401
    from .tile.npr_edges import NPREdgesAlgorithm  # noqa: F401
    from .tile.ambient_occlusion import AmbientOcclusionAlgorithm  # noqa: F401
    from .tile.openness import OpennessAlgorithm  # noqa: F401
    from .tile.fractal_anomaly import FractalAnomalyAlgorithm  # noqa: F401
    from .tile.scale_space_surprise import ScaleSpaceSurpriseAlgorithm  # noqa: F401
    from .tile.multi_light_uncertainty import MultiLightUncertaintyAlgorithm  # noqa: F401

    __all__.extend([
        "TileAlgorithm",
        "TopoUSMFastAlgorithm",
        "HillshadeAlgorithm",
        "SlopeAlgorithm",
        "SpecularAlgorithm",
        "AtmosphericScatteringAlgorithm",
        "MultiscaleDaskAlgorithm",
        "BlurAlgorithm",
        "CurvatureAlgorithm",
        "VisualSaliencyAlgorithm",
        "NPREdgesAlgorithm",
        "AmbientOcclusionAlgorithm",
        "OpennessAlgorithm",
        "FractalAnomalyAlgorithm",
        "ScaleSpaceSurpriseAlgorithm",
        "MultiLightUncertaintyAlgorithm",
    ])
except Exception as exc:
    logger.warning("FujiShaderGPU: tile algorithm classes unavailable: %s", exc)
