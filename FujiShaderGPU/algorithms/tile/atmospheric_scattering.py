"""Tile algorithm module for atmospheric_scattering (bridged from Dask shared implementation)."""
from ..dask_shared import AtmosphericScatteringAlgorithm as _DaskAtmosphericScatteringAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class AtmosphericScatteringAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskAtmosphericScatteringAlgorithm


__all__ = ["AtmosphericScatteringAlgorithm"]
