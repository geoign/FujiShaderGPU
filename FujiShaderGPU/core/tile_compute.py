"""Tile pipeline compute helpers."""
from __future__ import annotations

import math
from typing import Dict, Any, Iterable, Optional, Tuple, List
import cupy as cp


def _normalize_topousm_fast_radii_and_weights(
    target_distances,
    weights,
    pixel_size: float,
    manual_radii: Optional[Iterable] = None,
    manual_weights: Optional[Iterable] = None,
) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """Convert tile-scale parameters into Dask-shared TopoUSM Fast radii/weights."""
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
        # Keep radius 1 as-is: --mode local injects radii=[1] and the Dask
        # backend computes it as a sigma-1 gaussian; clamping it to 2 here made
        # the tile output a radius-2 box mean -- a different kernel whose ~2x
        # larger magnitude no longer matched the shared normalization stats.
        if radius < 1:
            radius = 1
        # NOTE: The previous hard cap (radius <= 256) that limited the per-tile
        # halo on the Windows/tile backend has been removed so user-specified
        # radii are honoured as-is.  Large radii enlarge the per-tile window and
        # its GPU memory footprint; process_dem_tiles emits an explicit VRAM
        # warning when that footprint is likely to exceed available VRAM.
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
    # The DEM is fed to the algorithms as-is (raw elevation), exactly like the
    # Dask backend: metric handling lives entirely in pixel_scale_x/y (signed
    # meters per pixel).  The old elevation_scale pre-multiply scaled the
    # output magnitude of elevation-based algorithms on geographic DEMs away
    # from the shared (raw-elevation) normalization stats and washed them out.
    if algorithm == 'topousm_fast':
        radii, topousm_fast_weights = _normalize_topousm_fast_radii_and_weights(
            target_distances=target_distances,
            weights=weights,
            pixel_size=pixel_size,
            manual_radii=algo_params.get("radii"),
            manual_weights=algo_params.get("weights"),
        )
        params = {
            'multiscale_mode': multiscale_mode,
            'radii': radii,
            'weights': topousm_fast_weights,
            'pixel_size': pixel_size,
            'sigma': sigma,
        }
        for key in (
            "global_stats",
            # Large-radius-from-overview fast path (set by process_dem_tiles /
            # process_single_tile); TopoUSMFastAlgorithm uses these instead of `radii`.
            "_topousm_fast_coarse_field", "_topousm_fast_small_radii", "_topousm_fast_small_weights",
            "_topousm_fast_w_large", "_topousm_fast_full_shape", "_topousm_fast_field_offset",
        ):
            if key in algo_params and algo_params[key] is not None:
                params[key] = algo_params[key]
        return algo_instance.process(dem_gpu, **params)

    # scale_space_surprise normalizes internally with the injected global stats
    # (seam-free), exactly like the Dask backend -- no tile-side special case.
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
