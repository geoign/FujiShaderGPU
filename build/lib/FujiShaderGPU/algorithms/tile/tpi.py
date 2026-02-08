"""Tile algorithm module for tpi (bridged from Dask shared implementation)."""
from ..dask_shared import TPIAlgorithm as _DaskTPIAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class TPIAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskTPIAlgorithm


__all__ = ["TPIAlgorithm"]
