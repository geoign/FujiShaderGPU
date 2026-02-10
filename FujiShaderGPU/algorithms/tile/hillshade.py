"""Tile algorithm module for hillshade (bridged from Dask shared implementation)."""
from ..dask_shared import HillshadeAlgorithm as _DaskHillshadeAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class HillshadeAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskHillshadeAlgorithm


__all__ = ["HillshadeAlgorithm"]
