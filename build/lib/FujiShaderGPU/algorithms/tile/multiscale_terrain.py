"""Tile algorithm module for multiscale_terrain (bridged from Dask shared implementation)."""
from ..dask_shared import MultiscaleDaskAlgorithm as _DaskMultiscaleDaskAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class MultiscaleDaskAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskMultiscaleDaskAlgorithm


__all__ = ["MultiscaleDaskAlgorithm"]
