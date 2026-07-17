"""
FujiShaderGPU/algorithms/tile_shared.py
Minimal abstract interface for native tile algorithms.

Concrete classes live under ``algorithms.tile``. Most are lightweight adapters
around the canonical Dask implementations; importing this base module must not
pull that entire dependency graph into callers that only need the interface.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import cupy as cp

if TYPE_CHECKING:
    from .tile.multi_light_uncertainty import MultiLightUncertaintyAlgorithm
    from .tile.scale_space_surprise import ScaleSpaceSurpriseAlgorithm


class TileAlgorithm(ABC):
    """Base class for terrain analysis algorithms."""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return the default parameters."""
        pass

    @abstractmethod
    def process(self, dem_gpu: cp.ndarray, **params) -> cp.ndarray:
        """
        Run the algorithm on the GPU.

        Parameters
        ----------
        dem_gpu : cp.ndarray
            DEM data on the GPU
        **params : dict
            Algorithm-specific parameters

        Returns
        -------
        cp.ndarray
            Processing result (on GPU)
        """
        pass


def __getattr__(name):
    """Load legacy concrete re-exports only when explicitly requested."""
    if name == "ScaleSpaceSurpriseAlgorithm":
        from .tile.scale_space_surprise import ScaleSpaceSurpriseAlgorithm

        return ScaleSpaceSurpriseAlgorithm
    if name == "MultiLightUncertaintyAlgorithm":
        from .tile.multi_light_uncertainty import MultiLightUncertaintyAlgorithm

        return MultiLightUncertaintyAlgorithm
    raise AttributeError(name)


__all__ = [
    "TileAlgorithm",
    "ScaleSpaceSurpriseAlgorithm",
    "MultiLightUncertaintyAlgorithm",
]
