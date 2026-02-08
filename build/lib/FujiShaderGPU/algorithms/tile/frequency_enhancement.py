"""Tile algorithm module for frequency_enhancement (bridged from Dask shared implementation)."""
from ..dask_shared import FrequencyEnhancementAlgorithm as _DaskFrequencyEnhancementAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class FrequencyEnhancementAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskFrequencyEnhancementAlgorithm


__all__ = ["FrequencyEnhancementAlgorithm"]
