"""
FujiShaderGPU/algorithms/tile_shared.py
Shared foundation for tile-based processing (Windows/macOS).

NOTE: most algorithms use the dask_shared.py classes from the tile
path via tile/dask_bridge.py.
This file keeps only the tile-specific TileAlgorithm base class and the
lightweight algorithms that delegate directly to shared kernels.
"""
import cupy as cp
from abc import ABC, abstractmethod
from typing import Dict, Any
from .tile.scale_space_surprise import ScaleSpaceSurpriseAlgorithm
from .tile.multi_light_uncertainty import MultiLightUncertaintyAlgorithm


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


__all__ = [
    "TileAlgorithm",
    "ScaleSpaceSurpriseAlgorithm",
    "MultiLightUncertaintyAlgorithm",
]
