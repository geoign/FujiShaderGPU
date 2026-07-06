"""Central Dask algorithm registry with per-algorithm modules."""
from __future__ import annotations

from .dask_shared import DaskAlgorithm
from .dask.topousm_fast import TopoUSMFastAlgorithm
from .dask.hillshade import HillshadeAlgorithm
from .dask.slope import SlopeAlgorithm
from .dask.specular import SpecularAlgorithm
from .dask.atmospheric_scattering import AtmosphericScatteringAlgorithm
from .dask.multiscale_terrain import MultiscaleDaskAlgorithm
from .dask.blur import BlurAlgorithm
from .dask.curvature import CurvatureAlgorithm
from .dask.visual_saliency import VisualSaliencyAlgorithm
from .dask.npr_edges import NPREdgesAlgorithm
from .dask.ambient_occlusion import AmbientOcclusionAlgorithm
from .dask.openness import OpennessAlgorithm
from .dask.fractal_anomaly import FractalAnomalyAlgorithm
from .dask.scale_space_surprise import ScaleSpaceSurpriseAlgorithm
from .dask.multi_light_uncertainty import MultiLightUncertaintyAlgorithm
from .dask.structure_tensor import StructureTensorAlgorithm
from .dask.frangi import FrangiAlgorithm
from .dask.lic import LICAlgorithm
from .dask.phase_congruency import PhaseCongruencyAlgorithm
from .dask.tv_decomposition import TVDecompositionAlgorithm
from .dask.scale_drift import ScaleDriftAlgorithm

ALGORITHMS = {
    'topousm_fast': TopoUSMFastAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    'slope': SlopeAlgorithm(),
    'specular': SpecularAlgorithm(),
    'atmospheric_scattering': AtmosphericScatteringAlgorithm(),
    'multiscale_terrain': MultiscaleDaskAlgorithm(),
    'blur': BlurAlgorithm(),
    'curvature': CurvatureAlgorithm(),
    'visual_saliency': VisualSaliencyAlgorithm(),
    'npr_edges': NPREdgesAlgorithm(),
    'ambient_occlusion': AmbientOcclusionAlgorithm(),
    'openness': OpennessAlgorithm(),
    'fractal_anomaly': FractalAnomalyAlgorithm(),
    'scale_space_surprise': ScaleSpaceSurpriseAlgorithm(),
    'multi_light_uncertainty': MultiLightUncertaintyAlgorithm(),
    'structure_tensor': StructureTensorAlgorithm(),
    'frangi': FrangiAlgorithm(),
    'lic': LICAlgorithm(),
    'phase_congruency': PhaseCongruencyAlgorithm(),
    'tv_decomposition': TVDecompositionAlgorithm(),
    'scale_drift': ScaleDriftAlgorithm(),
}

__all__ = ["ALGORITHMS", "DaskAlgorithm"]
