"""Shared CuPy kernels used by both Dask and tile backends."""
from __future__ import annotations

from typing import Iterable
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter


def scale_space_surprise(
    block: cp.ndarray,
    *,
    scales: Iterable[float],
    enhancement: float = 2.0,
    normalize: bool = True,
    nan_mask: cp.ndarray | None = None,
    weights: Iterable[float] | None = None,
) -> cp.ndarray:
    """Scale-Space Surprise Map kernel on a single CuPy block.

    ``weights`` (the unified ``--weights``, length-matching ``scales``) weights
    each consecutive-scale surprise term by the mean of its two scales' weights;
    absent/mismatched weights keep the original equal averaging.
    """
    if nan_mask is None:
        nan_mask = cp.isnan(block)
    work = cp.where(nan_mask, cp.nanmean(block), block) if nan_mask.any() else block

    scale_list = [float(s) for s in scales]
    weight_list = None
    if weights is not None:
        wl = list(weights)
        if len(wl) == len(scale_list):
            weight_list = [float(w) for w in wl]
    # Keep only positive scales, carrying weights along, then sort by scale.
    kept = [(s, (weight_list[i] if weight_list is not None else None))
            for i, s in enumerate(scale_list) if s > 0]
    kept.sort(key=lambda t: t[0])
    if len(kept) < 2:
        kept = [(1.0, None), (2.0, None), (4.0, None)]
    sorted_scales = [s for s, _ in kept]
    sorted_w = [w for _, w in kept] if all(w is not None for _, w in kept) else None

    responses = []
    for sigma in sorted_scales:
        blur = gaussian_filter(work, sigma=sigma, mode='reflect')
        responses.append(work - blur)

    n_pair = max(1, len(responses) - 1)
    pair_w = None
    if sorted_w is not None and len(responses) >= 2:
        pw = [0.5 * (sorted_w[i] + sorted_w[i + 1]) for i in range(len(responses) - 1)]
        psum = float(sum(pw))
        if psum > 1e-12:
            pair_w = [p / psum for p in pw]

    surprise = cp.zeros_like(work, dtype=cp.float32)
    for i in range(len(responses) - 1):
        term = cp.abs(responses[i + 1] - responses[i])
        surprise += term * pair_w[i] if pair_w is not None else term
    if pair_w is None:
        surprise /= n_pair

    if normalize:
        valid = surprise[~nan_mask] if nan_mask.any() else surprise.ravel()
        if valid.size > 0:
            lo = cp.percentile(valid, 5)
            hi = cp.percentile(valid, 95)
            if hi > lo:
                surprise = cp.clip((surprise - lo) / (hi - lo), 0, 1)
        surprise = cp.power(surprise, 1.0 / max(1e-3, enhancement))

    if nan_mask.any():
        surprise[nan_mask] = cp.nan
    return surprise.astype(cp.float32)


def multi_light_uncertainty(
    block: cp.ndarray,
    *,
    azimuths: Iterable[float],
    altitude: float = 45.0,
    z_factor: float = 1.0,
    uncertainty_weight: float = 0.7,
    pixel_size: float = 1.0,
    pixel_scale_x: float | None = None,
    pixel_scale_y: float | None = None,
    nan_mask: cp.ndarray | None = None,
) -> cp.ndarray:
    """Multi-light uncertainty shading kernel on a single CuPy block."""
    if nan_mask is None:
        nan_mask = cp.isnan(block)
    work = cp.where(nan_mask, cp.nanmean(block), block) if nan_mask.any() else block

    step_y = float(pixel_scale_y if pixel_scale_y is not None else pixel_size)
    step_x = float(pixel_scale_x if pixel_scale_x is not None else pixel_size)
    if abs(step_y) < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if abs(step_x) < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    dy, dx = cp.gradient(work * float(z_factor), step_y, step_x, edge_order=2)
    slope = cp.sqrt(dx**2 + dy**2)
    denom = cp.sqrt(slope**2 + 1.0)
    nx = -dx / denom
    ny = -dy / denom
    nz = 1.0 / denom

    alt_rad = cp.radians(cp.asarray(float(altitude), dtype=cp.float32))
    light_values = []
    for az in azimuths:
        az_rad = cp.radians(cp.asarray(float(az), dtype=cp.float32))
        lx = cp.sin(az_rad) * cp.cos(alt_rad)
        ly = cp.cos(az_rad) * cp.cos(alt_rad)
        lz = cp.sin(alt_rad)
        hs = cp.maximum(0.0, lx * nx + ly * ny + lz * nz)
        light_values.append(hs)

    stack = cp.stack(light_values, axis=0)
    mean_light = cp.mean(stack, axis=0)
    std_light = cp.std(stack, axis=0)
    uncertainty = cp.clip(std_light / (mean_light + 1e-6), 0.0, 1.0)

    shaded = cp.clip(mean_light + float(uncertainty_weight) * uncertainty, 0.0, 1.0)
    if nan_mask.any():
        shaded[nan_mask] = cp.nan
    return shaded.astype(cp.float32)

