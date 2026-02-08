"""Tile algorithm module for specular (bridged from Dask shared implementation)."""
from ..dask_shared import SpecularAlgorithm as _DaskSpecularAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class SpecularAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskSpecularAlgorithm


__all__ = ["SpecularAlgorithm"]
