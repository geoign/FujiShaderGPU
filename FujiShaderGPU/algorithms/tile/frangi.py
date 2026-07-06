"""Tile algorithm module for frangi (bridged from shared implementation)."""
from .._impl_frangi import FrangiAlgorithm as _DaskFrangiAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class FrangiAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskFrangiAlgorithm


__all__ = ["FrangiAlgorithm"]
