"""
FujiShaderGPU/algorithms/_global_stats.py

グローバル統計ユーティリティ。
ダウンサンプリングした代表データでの統計量計算・正規化適用を行う。
dask_shared.py からの分離モジュール (Phase 1)。
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import cupy as cp
import numpy as np
import dask.array as da

from ._base import Constants
from ._nan_utils import restore_nan


def determine_optimal_downsample_factor(
    data_shape: Tuple[int, int],
    algorithm_name: str = None,
    target_pixels: int = 500000,  # 目標ピクセル数（1000x1000）
    min_factor: int = 5,
    max_factor: int = 100,
    algorithm_complexity: Dict[str, float] = None) -> int:
    """
    データサイズとアルゴリズムの特性に基づいて最適なダウンサンプル係数を決定。

    Parameters:
    -----------
    data_shape : Tuple[int, int]
        入力データの形状 (height, width)
    algorithm_name : str
        アルゴリズム名（複雑度の調整用）
    target_pixels : int
        ダウンサンプル後の目標ピクセル数
    min_factor : int
        最小ダウンサンプル係数
    max_factor : int
        最大ダウンサンプル係数
    algorithm_complexity : Dict[str, float]
        アルゴリズムごとの複雑度係数（デフォルトは内蔵辞書）

    Returns:
    --------
    int : 最適なダウンサンプル係数
    """
    # アルゴリズムの複雑度係数（計算コストが高いほど大きい値）
    if algorithm_complexity is None:
        algorithm_complexity = {
            'rvi': 1.2,                    # マルチスケール処理
            'hillshade': 0.8,              # 単純な勾配計算
            'slope': 0.8,                  # 単純な勾配計算
            'specular': 1.5,               # ラフネス計算が重い
            'atmospheric_scattering': 0.9,
            'multiscale_terrain': 1.5,     # マルチスケール処理
            'curvature': 1.0,              # 2次微分
            'visual_saliency': 1.4,        # マルチスケール特徴抽出
            'npr_edges': 1.1,              # エッジ検出
            'ambient_occlusion': 2.0,      # 最も計算コストが高い
            'lrm': 1.1,                    # ガウシアンフィルタ
            'openness': 1.8,               # 多方向探索
            'fractal_anomaly': 1.6,        # マルチスケール回帰計算
        }

    # 現在のピクセル数
    current_pixels = data_shape[0] * data_shape[1]

    # 基本のダウンサンプル係数（平方根で計算）
    base_factor = cp.sqrt(current_pixels / target_pixels).get()

    # アルゴリズムの複雑度で調整
    complexity = algorithm_complexity.get(algorithm_name, 1.0)
    adjusted_factor = base_factor * complexity

    # 整数化して範囲内に収める
    downsample_factor = int(cp.clip(adjusted_factor, min_factor, max_factor))

    # データが小さい場合は係数を小さくする
    if current_pixels < 1_000_000:  # 1Mピクセル未満
        downsample_factor = min(downsample_factor, 2)
    elif current_pixels < 10_000_000:  # 10Mピクセル未満
        downsample_factor = min(downsample_factor, 4)
    return downsample_factor


def compute_global_stats(gpu_arr: da.Array,
                        stat_func: callable,
                        algorithm_func: callable,
                        algorithm_params: dict,
                        downsample_factor: int = None,  # Noneの場合は自動決定
                        depth: int = None,
                        algorithm_name: str = None) -> Tuple[Any, ...]:
    """
    ダウンサンプリングしたデータで統計量を計算する共通関数。

    Parameters:
    -----------
    gpu_arr : da.Array
        入力データ
    stat_func : callable
        統計量を計算する関数。CuPy配列を受け取り、統計量のタプルを返す
    algorithm_func : callable
        アルゴリズムの処理関数（正規化なしバージョン）
    algorithm_params : dict
        アルゴリズムのパラメータ
    downsample_factor : int
        ダウンサンプリング係数
    depth : int
        map_overlapのdepth（Noneの場合は自動計算）

    Returns:
    --------
    統計量のタプル
    """
    # downsample_factorが指定されていない場合は自動決定
    if downsample_factor is None:
        downsample_factor = determine_optimal_downsample_factor(
            gpu_arr.shape,
            algorithm_name=algorithm_name
        )

    # ダウンサンプリング
    downsampled = gpu_arr[::downsample_factor, ::downsample_factor]

    # depthの調整
    if depth is not None:
        depth_small = max(1, depth // downsample_factor)
    else:
        depth_small = 1

    # ダウンサンプル版でアルゴリズムを実行（正規化なし）
    params_small = algorithm_params.copy()

    # スケール系パラメータの調整が必要な場合
    for key in ['scale', 'sigma', 'radius', 'kernel_size']:
        if key in params_small and params_small[key] is not None:
            if isinstance(params_small[key], list):
                if key in ['radius', 'kernel_size']:
                    params_small[key] = [max(1, int(s/downsample_factor)) for s in params_small[key]]
                else:
                    params_small[key] = [max(1, s/downsample_factor) for s in params_small[key]]
            else:
                if key in ['radius', 'kernel_size']:
                    params_small[key] = max(1, int(params_small[key]/downsample_factor))
                else:
                    params_small[key] = max(1, params_small[key]/downsample_factor)

    result_small = downsampled.map_overlap(
        algorithm_func,
        depth=depth_small,
        boundary='reflect',
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        **params_small
    ).compute()

    # 統計量を計算
    stats = stat_func(result_small)

    return stats


def apply_global_normalization(block: cp.ndarray,
                              norm_func: callable,
                              stats: Tuple[Any, ...],
                              nan_mask: cp.ndarray = None) -> cp.ndarray:
    """
    グローバル統計量を使って正規化を適用する共通関数。

    Parameters:
    -----------
    block : cp.ndarray
        処理するブロック
    norm_func : callable
        正規化関数。(block, stats, nan_mask)を受け取り、正規化されたブロックを返す
    stats : tuple
        グローバル統計量
    nan_mask : cp.ndarray
        NaNマスク（オプション）

    Returns:
    --------
    正規化されたブロック
    """
    if nan_mask is None:
        nan_mask = cp.isnan(block)

    normalized = norm_func(block, stats, nan_mask)

    # ガンマ補正（0-1の範囲の場合のみ）
    valid_normalized = normalized[~nan_mask]
    if len(valid_normalized) > 0 and cp.min(valid_normalized) >= 0 and cp.max(valid_normalized) <= 1:
        normalized = cp.power(normalized, Constants.DEFAULT_GAMMA)

    # NaN位置を復元
    normalized = restore_nan(normalized, nan_mask)

    return normalized.astype(cp.float32)


__all__ = [
    "determine_optimal_downsample_factor",
    "compute_global_stats",
    "apply_global_normalization",
]
