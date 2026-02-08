"""Central Dask algorithm registry with per-algorithm modules."""
from __future__ import annotations

from .dask_shared import DaskAlgorithm
from .dask.rvi import RVIAlgorithm
from .dask.hillshade import HillshadeAlgorithm
from .dask.slope import SlopeAlgorithm
from .dask.specular import SpecularAlgorithm
from .dask.atmospheric_scattering import AtmosphericScatteringAlgorithm
from .dask.multiscale_terrain import MultiscaleDaskAlgorithm
from .dask.curvature import CurvatureAlgorithm
from .dask.visual_saliency import VisualSaliencyAlgorithm
from .dask.npr_edges import NPREdgesAlgorithm
from .dask.ambient_occlusion import AmbientOcclusionAlgorithm
from .dask.lrm import LRMAlgorithm
from .dask.openness import OpennessAlgorithm
from .dask.fractal_anomaly import FractalAnomalyAlgorithm
from .dask.scale_space_surprise import ScaleSpaceSurpriseAlgorithm
from .dask.multi_light_uncertainty import MultiLightUncertaintyAlgorithm

ALGORITHMS = {
    'rvi': RVIAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    'slope': SlopeAlgorithm(),
    'specular': SpecularAlgorithm(),
    'atmospheric_scattering': AtmosphericScatteringAlgorithm(),
    'multiscale_terrain': MultiscaleDaskAlgorithm(),
    'curvature': CurvatureAlgorithm(),
    'visual_saliency': VisualSaliencyAlgorithm(),
    'npr_edges': NPREdgesAlgorithm(),
    'ambient_occlusion': AmbientOcclusionAlgorithm(),
    'lrm': LRMAlgorithm(),
    'openness': OpennessAlgorithm(),
    'fractal_anomaly': FractalAnomalyAlgorithm(),
    'scale_space_surprise': ScaleSpaceSurpriseAlgorithm(),
    'multi_light_uncertainty': MultiLightUncertaintyAlgorithm(),
}

__all__ = ["ALGORITHMS", "DaskAlgorithm"]
