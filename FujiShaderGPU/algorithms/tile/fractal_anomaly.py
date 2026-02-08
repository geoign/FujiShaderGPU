"""Tile algorithm module for fractal_anomaly (bridged from Dask shared implementation)."""
from ..dask_shared import FractalAnomalyAlgorithm as _DaskFractalAnomalyAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class FractalAnomalyAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskFractalAnomalyAlgorithm


__all__ = ["FractalAnomalyAlgorithm"]
