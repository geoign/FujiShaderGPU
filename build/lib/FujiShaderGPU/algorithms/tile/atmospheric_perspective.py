"""Tile algorithm module for atmospheric_perspective (bridged from Dask shared implementation)."""
from ..dask_shared import AtmosphericPerspectiveAlgorithm as _DaskAtmosphericPerspectiveAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class AtmosphericPerspectiveAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskAtmosphericPerspectiveAlgorithm


__all__ = ["AtmosphericPerspectiveAlgorithm"]
