"""Tile algorithm module for multi_light_uncertainty (bridged from Dask shared implementation).

Previously delegated to the lightweight ``tile_shared`` single-light kernel, which
had no spatial/radii/weights support.  Routing through the Dask implementation gives
the tile backend the same unified local/spatial multi-radius behavior as Dask.
"""
from ..dask_shared import MultiLightUncertaintyAlgorithm as _DaskMultiLightUncertaintyAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class MultiLightUncertaintyAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskMultiLightUncertaintyAlgorithm


__all__ = ["MultiLightUncertaintyAlgorithm"]
