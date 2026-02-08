"""Tile algorithm module for lrm (bridged from Dask shared implementation)."""
from ..dask_shared import LRMAlgorithm as _DaskLRMAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class LRMAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskLRMAlgorithm


__all__ = ["LRMAlgorithm"]
