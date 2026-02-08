"""Tile pipeline compute helpers."""
from __future__ import annotations

import math
from typing import Dict, Any, Iterable, Optional, Tuple, List
import cupy as cp


def _normalize_rvi_radii_and_weights(
    target_distances,
    weights,
    pixel_size: float,
    manual_radii: Optional[Iterable] = None,
    manual_weights: Optional[Iterable] = None,
) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """Convert tile-scale parameters into Dask-shared RVI radii/weights."""
    if pixel_size <= 0:
        pixel_size = 1.0

    if manual_radii is not None:
        radii_raw = manual_radii
    else:
        radii_raw = target_distances

    if radii_raw is None:
        return None, None

    resolved: List[int] = []
    for value in radii_raw:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        if manual_radii is None:
            radius = int(round(numeric / pixel_size))
        else:
            radius = int(round(numeric))
        if radius < 2:
            radius = 2
        if radius > 256:
            radius = 256
        resolved.append(radius)

    if not resolved:
        return None, None

    # Deduplicate while preserving first appearance order.
    deduped_radii = list(dict.fromkeys(resolved))

    weights_raw = manual_weights if manual_weights is not None else weights
    if weights_raw is None:
        return deduped_radii, None

    cleaned_weights: List[float] = []
    for value in weights_raw:
        try:
            w = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(w) and w > 0:
            cleaned_weights.append(w)
        else:
            cleaned_weights.append(0.0)

    # If lengths differ (often after radius dedupe), let algorithm use uniform weights.
    if len(cleaned_weights) != len(deduped_radii):
        return deduped_radii, None

    total = sum(cleaned_weights)
    if total <= 0:
        return deduped_radii, None

    normalized_weights = [w / total for w in cleaned_weights]
    return deduped_radii, normalized_weights


def run_tile_algorithm(algo_instance, algorithm: str, dem_gpu: cp.ndarray, sigma: float, multiscale_mode: bool,
                       target_distances, weights, pixel_size: float, algo_params: Dict[str, Any]):
    if algorithm == 'rvi':
        radii, rvi_weights = _normalize_rvi_radii_and_weights(
            target_distances=target_distances,
            weights=weights,
            pixel_size=pixel_size,
            manual_radii=algo_params.get("radii"),
            manual_weights=algo_params.get("weights"),
        )
        params = {
            'multiscale_mode': multiscale_mode,
            'radii': radii,
            'weights': rvi_weights,
            'pixel_size': pixel_size,
            'sigma': sigma,
        }
        for key in ("global_stats", "downsample_factor"):
            if key in algo_params and algo_params[key] is not None:
                params[key] = algo_params[key]
        return algo_instance.process(dem_gpu, **params)

    if algorithm == 'scale_space_surprise':
        # Avoid per-tile percentile normalization (causes seam artifacts).
        # Tile pipeline will apply one global normalization pass afterwards.
        params = {'sigma': sigma, 'pixel_size': pixel_size, **algo_params}
        params['normalize'] = False
        return algo_instance.process(dem_gpu, **params)

    params = {'sigma': sigma, 'pixel_size': pixel_size, **algo_params}
    return algo_instance.process(dem_gpu, **params)


def apply_nodata_mask(result_gpu: cp.ndarray, mask_nodata, nodata):
    if mask_nodata is not None:
        mask_gpu = cp.asarray(mask_nodata)
        fill_value = cp.float32(nodata if nodata is not None else 0)
        if result_gpu.ndim == 2:
            result_gpu[mask_gpu] = fill_value
        elif result_gpu.ndim == 3:
            # HxWxC output (e.g. hillshade color)
            result_gpu[mask_gpu, :] = fill_value
        else:
            raise ValueError(f"Unsupported result ndim for nodata masking: {result_gpu.ndim}")
    return result_gpu
