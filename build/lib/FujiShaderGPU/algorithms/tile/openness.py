"""Tile algorithm module for openness (bridged from Dask shared implementation)."""
from ..dask_shared import OpennessAlgorithm as _DaskOpennessAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class OpennessAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskOpennessAlgorithm


__all__ = ["OpennessAlgorithm"]
