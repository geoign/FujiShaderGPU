"""
FujiShaderGPU/algorithms/_normalization.py

各アルゴリズム用の統計・正規化関数群。
dask_shared.py からの分離モジュール (Phase 1)。
"""
from __future__ import annotations
from typing import Tuple
import cupy as cp

NORMAL_PERCENTILE = 80.0
OVERFLOW_LIMIT = 1.5


# --- RVI用 ---

def rvi_stat_func(data: cp.ndarray) -> Tuple[float]:
    """RVI normalization scale from robust absolute percentile."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        abs_valid = cp.abs(valid_data)
        scale = float(cp.percentile(abs_valid, NORMAL_PERCENTILE))
        if scale > 1e-9:
            return (scale,)
        # Fallback for near-constant tiles/arrays.
        return (float(cp.std(valid_data)) if float(cp.std(valid_data)) > 1e-9 else 1.0,)
    return (1.0,)


def rvi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """RVI用の正規化"""
    scale_global = stats[0]
    if scale_global > 0:
        normalized = block / scale_global
        return cp.clip(normalized, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
    return cp.zeros_like(block)


# --- NPREdges用 ---

def npr_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """NPRエッジ用の統計量計算（勾配のパーセンタイル）"""
    # 簡易的に勾配を計算
    dy, dx = cp.gradient(data)
    gradient_mag = cp.sqrt(dx**2 + dy**2)
    valid_grad = gradient_mag[~cp.isnan(gradient_mag)]

    if len(valid_grad) > 0:
        return (float(cp.percentile(valid_grad, 70)),
                float(cp.percentile(valid_grad, 90)))
    return (0.1, 0.3)


# --- LRM用 ---

def lrm_stat_func(data: cp.ndarray) -> Tuple[float]:
    """Return robust signed scale: p80(abs(value)) maps to +/-1."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) == 0:
        return (1.0,)

    scale = float(cp.percentile(cp.abs(valid_data), NORMAL_PERCENTILE))
    if scale > 1e-9:
        return (scale,)
    return (1.0,)


# --- TPI/LRM共通 ---

def tpi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """TPI/LRM用の正規化"""
    max_abs = stats[0]
    if max_abs > 0:
        return cp.clip(block / max_abs, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
    return cp.zeros_like(block)


__all__ = [
    "rvi_stat_func",
    "rvi_norm_func",
    "npr_stat_func",
    "lrm_stat_func",
    "tpi_norm_func",
    "NORMAL_PERCENTILE",
    "OVERFLOW_LIMIT",
]
