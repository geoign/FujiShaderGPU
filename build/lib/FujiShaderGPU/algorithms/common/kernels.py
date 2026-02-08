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
) -> cp.ndarray:
    """Scale-Space Surprise Map kernel on a single CuPy block."""
    if nan_mask is None:
        nan_mask = cp.isnan(block)
    work = cp.where(nan_mask, cp.nanmean(block), block) if nan_mask.any() else block

    sorted_scales = sorted(float(s) for s in scales if float(s) > 0)
    if len(sorted_scales) < 2:
        sorted_scales = [1.0, 2.0, 4.0]

    responses = []
    for sigma in sorted_scales:
        blur = gaussian_filter(work, sigma=sigma, mode='reflect')
        responses.append(work - blur)

    surprise = cp.zeros_like(work, dtype=cp.float32)
    for i in range(len(responses) - 1):
        surprise += cp.abs(responses[i + 1] - responses[i])
    surprise /= max(1, len(responses) - 1)

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
    nan_mask: cp.ndarray | None = None,
) -> cp.ndarray:
    """Multi-light uncertainty shading kernel on a single CuPy block."""
    if nan_mask is None:
        nan_mask = cp.isnan(block)
    work = cp.where(nan_mask, cp.nanmean(block), block) if nan_mask.any() else block

    dy, dx = cp.gradient(work * float(z_factor), float(pixel_size), edge_order=2)
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

