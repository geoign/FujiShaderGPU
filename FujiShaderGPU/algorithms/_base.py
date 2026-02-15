"""
FujiShaderGPU/algorithms/_base.py

共通定数・基底クラス・解像度分類関数。
dask_shared.py からの分離モジュール (Phase 1)。
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import cupy as cp
import numpy as np
import dask.array as da


class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 150
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6


def classify_resolution(pixel_size: float) -> str:
    """
    解像度を分類（より精密な分類）。
    Returns: 'ultra_high', 'very_high', 'high', 'medium', 'low', 'very_low', 'ultra_low'
    """
    if pixel_size <= 0.5:
        return 'ultra_high'
    elif pixel_size <= 1.0:
        return 'very_high'
    elif pixel_size <= 2.5:
        return 'high'
    elif pixel_size <= 5.0:
        return 'medium'
    elif pixel_size <= 15.0:
        return 'low'
    elif pixel_size <= 30.0:
        return 'very_low'
    else:
        return 'ultra_low'


def get_gradient_scale_factor(pixel_size: float, algorithm: str = 'default') -> float:
    """
    解像度に応じた勾配スケーリング係数を返す。
    低解像度ほど大きな係数を返し、勾配を補正する。
    """
    if algorithm == 'npr_edges':
        # NPRエッジ用の係数（より積極的なスケーリング）
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.5
        elif pixel_size <= 10.0:
            return 2.5
        elif pixel_size <= 30.0:
            return 4.0
        else:
            return 6.0
    elif algorithm == 'visual_saliency':
        # Visual Saliency用の係数（より控えめなスケーリング）
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.2
        elif pixel_size <= 10.0:
            return 1.5
        elif pixel_size <= 30.0:
            return 2.0
        else:
            return 2.5
    else:
        # デフォルトの係数
        return cp.sqrt(max(1.0, pixel_size))


class DaskAlgorithm(ABC):
    """地形解析アルゴリズムの基底クラス"""

    @abstractmethod
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        """アルゴリズムのメイン処理"""
        pass

    @abstractmethod
    def get_default_params(self) -> dict:
        """デフォルトパラメータを返す"""
        pass


__all__ = [
    "Constants",
    "DaskAlgorithm",
    "classify_resolution",
    "get_gradient_scale_factor",
]
