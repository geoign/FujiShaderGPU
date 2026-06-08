"""Tile algorithm module for topousm_fast (bridged from Dask shared implementation)."""
from ..dask_shared import TopoUSMFastAlgorithm as _DaskTopoUSMFastAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class TopoUSMFastAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskTopoUSMFastAlgorithm


__all__ = ["TopoUSMFastAlgorithm"]
