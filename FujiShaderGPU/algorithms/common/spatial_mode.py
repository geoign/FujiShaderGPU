"""Common helpers for local/spatial algorithm modes."""
from __future__ import annotations

from typing import Iterable, List, Optional

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter


def determine_spatial_radii(
    pixel_size: float,
    target_distances: Iterable[float] = (5.0, 20.0, 80.0, 320.0),
    min_radius: int = 2,
    max_radius: int = 256,
    max_count: int = 4,
) -> List[int]:
    """Derive stable pixel radii from target physical distances."""
    px = float(pixel_size) if pixel_size and pixel_size > 0 else 1.0
    radii: List[int] = []
    for dist in target_distances:
        try:
            value = float(dist)
        except (TypeError, ValueError):
            continue
        r = int(round(value / px))
        r = max(min_radius, min(max_radius, r))
        radii.append(r)
    radii = sorted(list(dict.fromkeys(radii)))
    if not radii:
        return [min_radius]
    if len(radii) <= max_count:
        return radii
    idx = np.linspace(0, len(radii) - 1, max_count).astype(int)
    return [radii[int(i)] for i in idx]


def _normalize_weights(n: int, weights: Optional[Iterable[float]]) -> cp.ndarray:
    if n <= 0:
        return cp.asarray([], dtype=cp.float32)
    if weights is None:
        return cp.full((n,), 1.0 / n, dtype=cp.float32)
    arr = cp.asarray(list(weights), dtype=cp.float32)
    if arr.size != n:
        return cp.full((n,), 1.0 / n, dtype=cp.float32)
    arr = cp.where(cp.isfinite(arr) & (arr > 0), arr, 0)
    s = float(arr.sum())
    if s <= 0:
        return cp.full((n,), 1.0 / n, dtype=cp.float32)
    return arr / s


def _nan_aware_uniform(block: cp.ndarray, size: int) -> cp.ndarray:
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return uniform_filter(block, size=size, mode="reflect")
    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)
    sum_values = uniform_filter(filled * valid, size=size, mode="reflect")
    sum_weights = uniform_filter(valid, size=size, mode="reflect")
    mask = sum_weights > 1e-6
    denom = cp.where(mask, sum_weights, 1.0)
    out = sum_values / denom
    return cp.where(mask, out, cp.nan).astype(cp.float32)


def _nan_aware_gaussian(block: cp.ndarray, sigma: float) -> cp.ndarray:
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return gaussian_filter(block, sigma=sigma, mode="nearest")
    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)
    sum_values = gaussian_filter(filled * valid, sigma=sigma, mode="nearest")
    sum_weights = gaussian_filter(valid, sigma=sigma, mode="nearest")
    mask = sum_weights > 1e-6
    denom = cp.where(mask, sum_weights, 1.0)
    out = sum_values / denom
    return cp.where(mask, out, cp.nan).astype(cp.float32)


def smooth_with_radius(block: cp.ndarray, radius: int, method: str = "uniform") -> cp.ndarray:
    """Apply NaN-aware smoothing with a given spatial radius."""
    if radius <= 1:
        return block
    if method == "gaussian":
        sigma = max(0.5, float(radius) * 0.5)
        return _nan_aware_gaussian(block, sigma=sigma)
    size = int(2 * radius + 1)
    return _nan_aware_uniform(block, size=size)


def combine_multiscale_responses(
    responses: List[cp.ndarray],
    weights: Optional[Iterable[float]] = None,
    agg: str = "mean",
) -> cp.ndarray:
    """Combine per-radius responses robustly with NaN handling."""
    if not responses:
        raise ValueError("responses must not be empty")
    if len(responses) == 1:
        return responses[0].astype(cp.float32, copy=False)

    if agg in ("max", "min"):
        stack = cp.stack(responses, axis=0)
        return (cp.nanmax(stack, axis=0) if agg == "max" else cp.nanmin(stack, axis=0)).astype(cp.float32)

    w = _normalize_weights(len(responses), weights)
    result = cp.zeros_like(responses[0], dtype=cp.float32)
    weight_sum = cp.zeros_like(responses[0], dtype=cp.float32)
    for i, resp in enumerate(responses):
        valid = ~cp.isnan(resp)
        result[valid] += resp[valid] * w[i]
        weight_sum[valid] += w[i]
    mask = weight_sum > 1e-6
    denom = cp.where(mask, weight_sum, 1.0)
    out = result / denom
    return cp.where(mask, out, cp.nan).astype(cp.float32)
