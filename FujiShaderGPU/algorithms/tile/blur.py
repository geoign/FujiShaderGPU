"""Tile algorithm module for blur (bridged from Dask shared implementation)."""
from ..dask_shared import BlurAlgorithm as _DaskBlurAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class BlurAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskBlurAlgorithm


__all__ = ["BlurAlgorithm"]
