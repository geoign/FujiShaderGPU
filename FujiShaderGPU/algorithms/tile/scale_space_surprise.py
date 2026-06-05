"""Tile algorithm module for scale_space_surprise (bridged from Dask shared implementation).

Previously delegated to the lightweight ``tile_shared`` kernel, which skipped the
global-stat normalization and ignored ``--weights``.  Routing through the Dask
implementation gives the tile backend parity with the Dask backend (global p99
normalization + weighted scale mixing).
"""
from ..dask_shared import ScaleSpaceSurpriseAlgorithm as _DaskScaleSpaceSurpriseAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class ScaleSpaceSurpriseAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskScaleSpaceSurpriseAlgorithm


__all__ = ["ScaleSpaceSurpriseAlgorithm"]
