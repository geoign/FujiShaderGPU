"""Tile algorithm module for scale_drift (bridged from shared implementation)."""
from .._impl_scale_drift import ScaleDriftAlgorithm as _DaskScaleDriftAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class ScaleDriftAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskScaleDriftAlgorithm


__all__ = ["ScaleDriftAlgorithm"]
