"""Tile algorithm module for lic (bridged from shared implementation)."""
from .._impl_lic import LICAlgorithm as _DaskLICAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class LICAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskLICAlgorithm


__all__ = ["LICAlgorithm"]
