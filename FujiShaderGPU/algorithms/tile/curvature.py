"""Tile algorithm module for curvature (bridged from Dask shared implementation)."""
from ..dask_shared import CurvatureAlgorithm as _DaskCurvatureAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class CurvatureAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskCurvatureAlgorithm


__all__ = ["CurvatureAlgorithm"]
