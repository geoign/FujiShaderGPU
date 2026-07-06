"""Tile algorithm module for structure_tensor (bridged from shared implementation)."""
from .._impl_structure_tensor import StructureTensorAlgorithm as _DaskStructureTensorAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class StructureTensorAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskStructureTensorAlgorithm


__all__ = ["StructureTensorAlgorithm"]
