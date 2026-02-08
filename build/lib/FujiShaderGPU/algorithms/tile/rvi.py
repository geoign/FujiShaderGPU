"""Tile algorithm module for rvi (bridged from Dask shared implementation)."""
from ..dask_shared import RVIAlgorithm as _DaskRVIAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class RVIAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskRVIAlgorithm


__all__ = ["RVIAlgorithm"]
