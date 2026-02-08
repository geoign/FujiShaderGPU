"""Tile algorithm module for slope (bridged from Dask shared implementation)."""
from ..dask_shared import SlopeAlgorithm as _DaskSlopeAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class SlopeAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskSlopeAlgorithm


__all__ = ["SlopeAlgorithm"]
