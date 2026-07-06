"""Tile algorithm module for tv_decomposition (bridged from shared implementation)."""
from .._impl_tv_decomposition import TVDecompositionAlgorithm as _DaskTVDecompositionAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class TVDecompositionAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskTVDecompositionAlgorithm


__all__ = ["TVDecompositionAlgorithm"]
