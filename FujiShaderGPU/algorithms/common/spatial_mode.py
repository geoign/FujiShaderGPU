"""Common helpers for local/spatial algorithm modes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import yaml


DEFAULT_SPATIAL_PRESETS = [
    {"max_pixel_size_m": 5.0, "radii": [2, 4, 16, 64], "weights": [0.36, 0.30, 0.22, 0.12]},
    {"min_pixel_size_m": 5.0, "max_pixel_size_m": 25.0, "radii": [2, 4, 8, 16], "weights": [0.34, 0.30, 0.22, 0.14]},
    {"min_pixel_size_m": 25.0, "max_pixel_size_m": 50.0, "radii": [2, 3, 6, 12], "weights": [0.33, 0.29, 0.23, 0.15]},
    {"min_pixel_size_m": 50.0, "max_pixel_size_m": 250.0, "radii": [2, 3, 4, 8], "weights": [0.32, 0.28, 0.23, 0.17]},
    {"min_pixel_size_m": 250.0, "max_pixel_size_m": 1250.0, "radii": [2, 3, 4, 6], "weights": [0.31, 0.28, 0.24, 0.17]},
    {"min_pixel_size_m": 1250.0, "max_pixel_size_m": 5000.0, "radii": [2, 3, 4, 5], "weights": [0.30, 0.28, 0.24, 0.18]},
    {"min_pixel_size_m": 5000.0, "radii": [2, 3, 4, 6], "weights": [0.30, 0.27, 0.23, 0.20]},
]


def _sanitize_radii(values: Iterable[float], min_radius: int = 2, max_radius: int = 256, max_count: int = 4) -> List[int]:
    out: List[int] = []
    for value in values:
        try:
            rv = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if rv <= 0:
            continue
        rv = max(min_radius, min(max_radius, rv))
        out.append(rv)
    out = list(dict.fromkeys(out))
    if not out:
        out = [min_radius]
    if len(out) > max_count:
        idx = np.linspace(0, len(out) - 1, max_count).astype(int)
        out = [out[int(i)] for i in idx]
    return out


def _sanitize_weights(values: Optional[Iterable[float]], n: int) -> Optional[List[float]]:
    if values is None or n <= 0:
        return None
    out: List[float] = []
    for value in values:
        try:
            w = float(value)
        except (TypeError, ValueError):
            continue
        out.append(w if np.isfinite(w) and w > 0 else 0.0)
    if len(out) != n:
        return None
    s = float(sum(out))
    if s <= 0:
        return None
    return [float(w / s) for w in out]


@lru_cache(maxsize=1)
def _load_spatial_presets() -> List[dict]:
    config_path = Path(__file__).resolve().parents[2] / "config" / "spatial_presets.yaml"
    presets = None
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            presets = data.get("spatial_presets")
        except Exception:
            presets = None
    if not isinstance(presets, list) or len(presets) == 0:
        presets = DEFAULT_SPATIAL_PRESETS
    return presets


def _select_preset(pixel_size: float) -> dict:
    px = float(pixel_size) if pixel_size and pixel_size > 0 else 1.0
    for preset in _load_spatial_presets():
        min_px = preset.get("min_pixel_size_m", None)
        max_px = preset.get("max_pixel_size_m", None)
        if min_px is not None and px < float(min_px):
            continue
        if max_px is not None and px >= float(max_px):
            continue
        return preset
    return DEFAULT_SPATIAL_PRESETS[-1]


def determine_spatial_profile(
    pixel_size: float,
    min_radius: int = 2,
    max_radius: int = 256,
    max_count: int = 4,
) -> Tuple[List[int], Optional[List[float]]]:
    preset = _select_preset(pixel_size)
    radii = _sanitize_radii(preset.get("radii", []), min_radius=min_radius, max_radius=max_radius, max_count=max_count)
    weights = _sanitize_weights(preset.get("weights"), len(radii))
    return radii, weights


def determine_spatial_radii(
    pixel_size: float,
    target_distances: Iterable[float] = (5.0, 20.0, 80.0, 320.0),
    min_radius: int = 2,
    max_radius: int = 256,
    max_count: int = 4,
) -> List[int]:
    """Derive spatial radii from YAML preset by pixel-size class."""
    radii, _ = determine_spatial_profile(
        pixel_size=pixel_size,
        min_radius=min_radius,
        max_radius=max_radius,
        max_count=max_count,
    )
    return radii
