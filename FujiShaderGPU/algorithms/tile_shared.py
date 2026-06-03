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
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)


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


class ScaleSpaceSurpriseAlgorithm(TileAlgorithm):
    """Visualize local feature change across scales."""

    def get_default_params(self):
        return {
            "scales": [1.0, 2.0, 4.0, 8.0, 16.0],
            "enhancement": 2.0,
            "normalize": True,
        }

    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        return kernel_scale_space_surprise(
            dem_gpu,
            scales=p["scales"],
            enhancement=float(p["enhancement"]),
            normalize=bool(p["normalize"]),
        )


class MultiLightUncertaintyAlgorithm(TileAlgorithm):
    """Shading that overlays uncertainty from multiple light azimuths."""

    def get_default_params(self):
        return {
            "azimuths": [315.0, 45.0, 135.0, 225.0],
            "altitude": 45.0,
            "z_factor": 1.0,
            "uncertainty_weight": 0.7,
        }

    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        return kernel_multi_light_uncertainty(
            dem_gpu,
            azimuths=p["azimuths"],
            altitude=float(p["altitude"]),
            z_factor=float(p["z_factor"]),
            uncertainty_weight=float(p["uncertainty_weight"]),
            pixel_size=float(p.get("pixel_size", 1.0)),
            pixel_scale_x=p.get("pixel_scale_x", None),
            pixel_scale_y=p.get("pixel_scale_y", None),
        )
