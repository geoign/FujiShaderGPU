"""Shared auto-parameter helpers used by multiple backends."""
from __future__ import annotations

from typing import List, Tuple


def determine_optimal_radii(terrain_stats: dict) -> Tuple[List[int], List[float]]:
    """Estimate RVI radii and weights from sampled terrain statistics."""
    complexity = terrain_stats.get('complexity_score', 0.5)
    if complexity > 0.7:
        radii = [3, 8, 20, 50]
        weights = [0.35, 0.30, 0.20, 0.15]
    elif complexity > 0.4:
        radii = [4, 16, 64, 256]
        weights = [0.30, 0.30, 0.25, 0.15]
    else:
        radii = [6, 24, 96, 384]
        weights = [0.25, 0.30, 0.30, 0.15]
    return radii, weights
