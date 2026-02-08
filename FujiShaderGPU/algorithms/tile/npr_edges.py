"""Tile algorithm module for npr_edges (bridged from Dask shared implementation)."""
from ..dask_shared import NPREdgesAlgorithm as _DaskNPREdgesAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class NPREdgesAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskNPREdgesAlgorithm


__all__ = ["NPREdgesAlgorithm"]
