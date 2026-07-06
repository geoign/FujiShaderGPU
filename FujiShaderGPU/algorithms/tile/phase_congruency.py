"""Tile algorithm module for phase_congruency (bridged from shared implementation)."""
from .._impl_phase_congruency import PhaseCongruencyAlgorithm as _DaskPhaseCongruencyAlgorithm
from .dask_bridge import DaskSharedTileAdapter


class PhaseCongruencyAlgorithm(DaskSharedTileAdapter):
    dask_algorithm_cls = _DaskPhaseCongruencyAlgorithm


__all__ = ["PhaseCongruencyAlgorithm"]
