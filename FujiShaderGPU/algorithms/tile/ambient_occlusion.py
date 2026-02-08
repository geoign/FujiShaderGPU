"""Tile algorithm module for ambient_occlusion (bridged from Dask shared implementation)."""
from ..dask_shared import AmbientOcclusionAlgorithm as _DaskAmbientOcclusionAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class AmbientOcclusionAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskAmbientOcclusionAlgorithm


__all__ = ["AmbientOcclusionAlgorithm"]
