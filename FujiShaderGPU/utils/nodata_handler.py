"""NoData preprocessing helpers for tile-based computation."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_propagation

logger = logging.getLogger(__name__)


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    """Return masked cells connected to the current block edge."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    seeds = np.zeros_like(mask, dtype=bool)
    seeds[0, :] = mask[0, :]
    seeds[-1, :] = mask[-1, :]
    seeds[:, 0] = mask[:, 0]
    seeds[:, -1] = mask[:, -1]
    if not np.any(seeds):
        return np.zeros_like(mask, dtype=bool)
    return binary_propagation(seeds, mask=mask).astype(bool, copy=False)
