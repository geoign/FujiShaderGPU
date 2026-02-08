"""Tile algorithm module for rvi_glossy (bridged from Dask shared implementation)."""
from ..dask_shared import RVIGlossyAlgorithm as _DaskRVIGlossyAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class RVIGlossyAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskRVIGlossyAlgorithm


__all__ = ["RVIGlossyAlgorithm"]

