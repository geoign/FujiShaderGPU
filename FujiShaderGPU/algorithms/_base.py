"""
FujiShaderGPU/algorithms/_base.py

Shared constants, base classes, and resolution-classification helpers.
Module split out from dask_shared.py (Phase 1).
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import cupy as cp
import numpy as np
import dask.array as da


class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 150
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6


def classify_resolution(pixel_size: float) -> str:
    """
    Classify the resolution (finer classification).
    Returns: 'ultra_high', 'very_high', 'high', 'medium', 'low', 'very_low', 'ultra_low'
    """
    if pixel_size <= 0.5:
        return 'ultra_high'
    elif pixel_size <= 1.0:
        return 'very_high'
    elif pixel_size <= 2.5:
        return 'high'
    elif pixel_size <= 5.0:
        return 'medium'
    elif pixel_size <= 15.0:
        return 'low'
    elif pixel_size <= 30.0:
        return 'very_low'
    else:
        return 'ultra_low'


def get_gradient_scale_factor(pixel_size: float, algorithm: str = 'default') -> float:
    """
    Return a gradient scaling factor based on resolution.
    Lower resolution returns a larger factor to correct the gradient.
    """
    if algorithm == 'npr_edges':
        # Factor for NPR edges (more aggressive scaling)
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.5
        elif pixel_size <= 10.0:
            return 2.5
        elif pixel_size <= 30.0:
            return 4.0
        else:
            return 6.0
    elif algorithm == 'visual_saliency':
        # Factor for Visual Saliency (more conservative scaling)
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.2
        elif pixel_size <= 10.0:
            return 1.5
        elif pixel_size <= 30.0:
            return 2.0
        else:
            return 2.5
    else:
        # Default factor
        return cp.sqrt(max(1.0, pixel_size))


class DaskAlgorithm(ABC):
    """Base class for terrain analysis algorithms."""

    @abstractmethod
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        """Main processing of the algorithm."""
        pass

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return the default parameters."""
        pass


__all__ = [
    "Constants",
    "DaskAlgorithm",
    "classify_resolution",
    "get_gradient_scale_factor",
]
