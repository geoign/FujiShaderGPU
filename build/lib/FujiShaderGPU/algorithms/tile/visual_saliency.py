"""Tile algorithm module for visual_saliency (bridged from Dask shared implementation)."""
from ..dask_shared import VisualSaliencyAlgorithm as _DaskVisualSaliencyAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class VisualSaliencyAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskVisualSaliencyAlgorithm


__all__ = ["VisualSaliencyAlgorithm"]
