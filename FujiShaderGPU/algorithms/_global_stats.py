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
                        downsample_factor: int = None,  # 後方互換のため保持（未使用）
                        depth: int = None,
                        algorithm_name: str = None) -> Tuple[Any, ...]:
    """
    境界付き中央クロップ上でアルゴリズムを実行し統計量を計算する共通関数。

    フル解像度配列をストライドスライス（``gpu_arr[::n, ::n]``）すると、間引きの
    ために全チャンク＝データセット全体が読み込まれGPUへ転送され、書き込み進捗が
    出る前に巨大ラスタで停止してしまう。連続した中央ウィンドウを読むことで、重なる
    チャンクのみを実体化し、小さく有界なコストでグローバルスケールを推定する。
    アルゴリズムはそのウィンドウ上でフル解像度のブロック関数として直接実行するため、
    スケール系パラメータの縮小は不要。

    Parameters:
    -----------
    gpu_arr : da.Array
        入力データ
    stat_func : callable
        統計量を計算する関数。CuPy配列を受け取り、統計量のタプルを返す
    algorithm_func : callable
        アルゴリズムの処理関数（正規化なしバージョン、ブロック関数）
    algorithm_params : dict
        アルゴリズムのパラメータ
    depth : int
        アルゴリズムのハロー幅（中央ウィンドウサイズの目安に使用）

    Returns:
    --------
    統計量のタプル
    """
    h, w = gpu_arr.shape
    halo = int(depth) if depth else 1
    # ウィンドウはアルゴリズムのフットプリントを十分含みつつ、フルラスタサイズに
    # かかわらず有界に保つ。
    win = int(min(int(h), int(w), max(4096, halo * 4)))
    win = max(256, win)
    y0 = max(0, (int(h) - win) // 2)
    x0 = max(0, (int(w) - win) // 2)
    y1 = min(int(h), y0 + win)
    x1 = min(int(w), x0 + win)

    sample_block = gpu_arr[y0:y1, x0:x1].compute()
    if getattr(sample_block, "size", 0) == 0:
        return stat_func(cp.zeros((1, 1), dtype=cp.float32))

    result_small = algorithm_func(
        sample_block.astype(cp.float32, copy=False), **algorithm_params
    )

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

    # NaN位置を復元
    normalized = restore_nan(normalized, nan_mask)

    return normalized.astype(cp.float32)


__all__ = [
    "determine_optimal_downsample_factor",
    "compute_global_stats",
    "apply_global_normalization",
]
