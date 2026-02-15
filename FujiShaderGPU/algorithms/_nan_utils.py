"""
FujiShaderGPU/algorithms/_nan_utils.py

NaN処理・空間スムージング・ダウンサンプリング・リストア関数群。
dask_shared.py からの分離モジュール (Phase 1)。
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, zoom

from .common.spatial_mode import determine_spatial_radii, determine_spatial_profile


def handle_nan_with_gaussian(block: cp.ndarray, sigma: float, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaNを考慮したガウシアンフィルタ処理"""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return gaussian_filter(block, sigma=sigma, mode=mode), nan_mask

    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)

    smoothed_values = gaussian_filter(filled * valid, sigma=sigma, mode=mode)
    smoothed_weights = gaussian_filter(valid, sigma=sigma, mode=mode)
    smoothed = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)

    return smoothed, nan_mask


def handle_nan_with_uniform(block: cp.ndarray, size: int, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaNを考慮したuniform_filter処理"""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return uniform_filter(block, size=size, mode=mode), nan_mask

    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)

    sum_values = uniform_filter(filled * valid, size=size, mode=mode)
    sum_weights = uniform_filter(valid, size=size, mode=mode)
    mean = cp.where(sum_weights > 0, sum_values / sum_weights, 0)

    return mean, nan_mask


def handle_nan_for_gradient(block: cp.ndarray, scale: float = 1.0,
                          pixel_size: float = 1.0,
                          pixel_scale_x: float = None,
                          pixel_scale_y: float = None) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """NaNを考慮した勾配計算"""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block

    # Use metric spacing magnitude only. Sign carries geotransform orientation,
    # which can unintentionally flip illumination direction in shading algorithms.
    step_y = abs(float(pixel_scale_y if pixel_scale_y is not None else pixel_size))
    step_x = abs(float(pixel_scale_x if pixel_scale_x is not None else pixel_size))
    if step_y < 1e-9:
        step_y = float(pixel_size if pixel_size else 1.0)
    if step_x < 1e-9:
        step_x = float(pixel_size if pixel_size else 1.0)
    dy, dx = cp.gradient(filled * scale, step_y, step_x, edge_order=2)
    return dy, dx, nan_mask


def _normalize_spatial_radii(radii: Optional[List[int]], pixel_size: float) -> List[int]:
    """Normalize user-provided radii or auto-derive stable defaults."""
    if radii is None:
        return determine_spatial_radii(pixel_size=pixel_size)
    out: List[int] = []
    for r in radii:
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        if rv > 0:
            out.append(rv)
    if not out:
        return determine_spatial_radii(pixel_size=pixel_size)
    # Keep user order while dropping duplicates.
    seen = set()
    ordered = []
    for v in out:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _resolve_spatial_radii_weights(
    radii: Optional[List[int]],
    weights: Optional[List[float]],
    pixel_size: float,
) -> Tuple[List[int], Optional[List[float]]]:
    """Resolve radii/weights with YAML presets when user values are omitted."""
    if radii is None:
        auto_radii, auto_weights = determine_spatial_profile(pixel_size=pixel_size)
        return auto_radii, auto_weights if weights is None else weights

    resolved_radii = _normalize_spatial_radii(radii, pixel_size)
    if not isinstance(weights, (list, tuple)) or len(weights) != len(resolved_radii):
        return resolved_radii, None

    cleaned: List[float] = []
    for w in weights:
        try:
            fv = float(w)
        except (TypeError, ValueError):
            return resolved_radii, None
        cleaned.append(fv if np.isfinite(fv) and fv > 0 else 0.0)
    s = float(sum(cleaned))
    if s <= 0:
        return resolved_radii, None
    return resolved_radii, [v / s for v in cleaned]


def _combine_multiscale_dask(
    responses: List[da.Array],
    *,
    weights: Optional[List[float]] = None,
    agg: str = "mean",
) -> da.Array:
    """Combine per-radius dask responses with optional weighted mean."""
    if not responses:
        raise ValueError("responses must not be empty")
    if len(responses) == 1:
        return responses[0]

    stacked = da.stack(responses, axis=0)
    agg_norm = str(agg or "mean").lower()
    if agg_norm == "stack":
        return stacked
    if agg_norm == "max":
        return da.max(stacked, axis=0)
    if agg_norm == "min":
        return da.min(stacked, axis=0)
    if agg_norm == "sum":
        return da.sum(stacked, axis=0)

    if isinstance(weights, (list, tuple)) and len(weights) == len(responses):
        w = np.asarray(weights, dtype=np.float32)
        if np.isfinite(w).all() and w.sum() > 0:
            w = w / w.sum()
            out = responses[0] * float(w[0])
            for i in range(1, len(responses)):
                out = out + responses[i] * float(w[i])
            return out
    return da.mean(stacked, axis=0)


def _smooth_for_radius(
    block: cp.ndarray,
    radius: float,
    *,
    pixel_size: float = 1.0,
    algorithm_name: str = "default",
) -> cp.ndarray:
    """NaN-aware gaussian smoothing controlled by spatial radius."""
    r = max(1.0, float(radius))
    if r <= 1.0:
        return block
    factor = _radius_to_downsample_factor(
        r,
        block_shape=block.shape,
        pixel_size=pixel_size,
        algorithm_name=algorithm_name,
    )
    if factor <= 1:
        sigma = max(0.5, r / 2.0)
        smoothed, _ = handle_nan_with_gaussian(block, sigma=sigma, mode="nearest")
        return smoothed

    reduced = _downsample_nan_aware(block, factor)
    sigma_small = max(0.5, (r / factor) / 2.0)
    smoothed_small, _ = handle_nan_with_gaussian(reduced, sigma=sigma_small, mode="nearest")
    return _upsample_to_shape(smoothed_small, block.shape)


def _radius_to_downsample_factor(
    radius: float,
    *,
    block_shape: Optional[Tuple[int, int]] = None,
    pixel_size: float = 1.0,
    algorithm_name: str = "default",
    base_radius: float = 24.0,
    max_factor: int = 16,
) -> int:
    """
    Dynamic downsample factor from radius + workload context.
    Returns power-of-two factors: 1,2,4,8,...
    """
    r = max(1.0, float(radius))
    px = max(1e-3, float(pixel_size) if pixel_size else 1.0)

    algo_factor_map = {
        "rvi": 1.15,
        "hillshade": 1.0,
        "slope": 1.0,
        "specular": 1.4,
        "atmospheric_scattering": 1.05,
        "curvature": 1.1,
        "ambient_occlusion": 1.5,
        "openness": 1.4,
        "multi_light_uncertainty": 1.25,
    }
    algo_factor = float(algo_factor_map.get(str(algorithm_name), 1.0))

    block_factor = 1.0
    if block_shape is not None and len(block_shape) >= 2:
        h = max(1, int(block_shape[0]))
        w = max(1, int(block_shape[1]))
        block_pixels = float(h * w)
        # Mild scaling by block area to avoid over-aggressive shrink on small chunks.
        block_factor = max(1.0, (block_pixels / 1_000_000.0) ** 0.5)

    # 0.5m should be somewhat more aggressive than 1m.
    resolution_factor = max(1.0, 1.0 / px)

    score = (r / max(1.0, base_radius)) * algo_factor * block_factor * (resolution_factor ** 0.35)
    if score <= 1.0:
        return 1

    # Convert to power-of-two scaling for stable kernels.
    factor = 2 ** int(np.floor(np.log2(score)))
    factor = int(max(1, min(factor, max_factor)))
    return factor


def _downsample_nan_aware(block: cp.ndarray, factor: int) -> cp.ndarray:
    if factor <= 1:
        return block
    nan_mask = cp.isnan(block)
    h, w = block.shape[:2]
    out_h = max(1, (int(h) + int(factor) - 1) // int(factor))
    out_w = max(1, (int(w) + int(factor) - 1) // int(factor))
    if nan_mask.any():
        fill = cp.nanmean(block)
        fill = cp.where(cp.isfinite(fill), fill, 0.0)
        work = cp.where(nan_mask, fill, block).astype(cp.float32)
    else:
        work = block.astype(cp.float32, copy=False)
    sy = out_h / max(1, h)
    sx = out_w / max(1, w)
    return zoom(work, zoom=(sy, sx), order=1, mode="nearest").astype(cp.float32)


def _upsample_to_shape(block: cp.ndarray, target_shape: Tuple[int, int]) -> cp.ndarray:
    th, tw = int(target_shape[0]), int(target_shape[1])
    h, w = block.shape[:2]
    if h == th and w == tw:
        return block.astype(cp.float32, copy=False)
    sy = th / max(1, h)
    sx = tw / max(1, w)
    out = zoom(block, zoom=(sy, sx), order=1, mode="nearest").astype(cp.float32)
    return out[:th, :tw]


def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """NaN位置を復元"""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result


__all__ = [
    "handle_nan_with_gaussian",
    "handle_nan_with_uniform",
    "handle_nan_for_gradient",
    "_normalize_spatial_radii",
    "_resolve_spatial_radii_weights",
    "_combine_multiscale_dask",
    "_smooth_for_radius",
    "_radius_to_downsample_factor",
    "_downsample_nan_aware",
    "_upsample_to_shape",
    "restore_nan",
]
