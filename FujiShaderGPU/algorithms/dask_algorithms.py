"""
FujiShaderGPU/algorithms/dask_algorithms.py
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter, minimum_filter, convolve, binary_dilation
from tqdm.auto import tqdm

class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 200
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6

# 1. より詳細な解像度分類関数（既存のclassify_resolutionを置き換え）
def classify_resolution(pixel_size: float) -> str:
    """
    解像度を分類（より詳細な分類）
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

# 2. 解像度に応じた勾配スケーリング係数を計算する関数（新規追加）
def get_gradient_scale_factor(pixel_size: float, algorithm: str = 'default') -> float:
    """
    解像度に応じた勾配スケーリング係数を返す
    低解像度ほど大きな係数を返し、勾配を補正する
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
    
###############################################################################
# アルゴリズム基底クラスと共通インターフェース
###############################################################################

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

###############################################################################
# NaN処理のユーティリティ関数
###############################################################################

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
                          pixel_size: float = 1.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """NaNを考慮した勾配計算"""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    
    dy, dx = cp.gradient(filled * scale, pixel_size, edge_order=2)
    return dy, dx, nan_mask
    
def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """NaN位置を復元"""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result

###############################################################################
# グローバル統計ユーティリティ
###############################################################################
def determine_optimal_downsample_factor(
    data_shape: Tuple[int, int],
    algorithm_name: str = None,
    target_pixels: int = 1000000,  # 目標ピクセル数（1000x1000）
    min_factor: int = 5,
    max_factor: int = 100,
    algorithm_complexity: Dict[str, float] = None) -> int:
    """
    データサイズとアルゴリズムの特性に基づいて最適なダウンサンプル係数を決定
    
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
            'frequency_enhancement': 1.3,   # FFT処理
            'curvature': 1.0,              # 2次微分
            'visual_saliency': 1.4,        # マルチスケール特徴抽出
            'npr_edges': 1.1,              # エッジ検出
            'atmospheric_perspective': 0.9,
            'ambient_occlusion': 2.0,      # 最も計算コストが高い
            'tpi': 1.0,                    # 畳み込み処理
            'lrm': 1.1,                    # ガウシアンフィルタ
            'openness': 1.8,               # 多方向探索
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
    ダウンサンプリングしたデータで統計量を計算する共通関数
    
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
    
    # CuPyのメモリプールをクリア
    result_small = None  # 参照を切る
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()  # GPU処理の完了を待つ
    
    # Pythonのガベージコレクションも実行
    import gc
    gc.collect()
    
    return stats

def apply_global_normalization(block: cp.ndarray, 
                              norm_func: callable,
                              stats: Tuple[Any, ...],
                              nan_mask: cp.ndarray = None) -> cp.ndarray:
    """
    グローバル統計量を使って正規化を適用する共通関数
    
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


###############################################################################
# 各アルゴリズム用の統計・正規化関数
###############################################################################

# RVI用
def rvi_stat_func(data: cp.ndarray) -> Tuple[float]:
    """RVI用の統計量計算（標準偏差）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.std(valid_data)),)
    return (1.0,)

def rvi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """RVI用の正規化"""
    std_global = stats[0]
    if std_global > 0:
        normalized = block / (3 * std_global)
        return cp.clip(normalized, -1, 1)
    return cp.zeros_like(block)

# FrequencyEnhancement用
def freq_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """周波数強調用の統計量計算（最小値・最大値）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def freq_norm_func(block: cp.ndarray, stats: Tuple[float, float], nan_mask: cp.ndarray) -> cp.ndarray:
    """周波数強調用の正規化"""
    min_val, max_val = stats
    if max_val > min_val:
        return (block - min_val) / (max_val - min_val)
    return cp.full_like(block, 0.5)

# NPREdges用
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

# TPI/LRM用
def tpi_lrm_stat_func(data: cp.ndarray) -> Tuple[float]:
    """TPI/LRM用の統計量計算（最大絶対値）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.maximum(cp.abs(cp.min(valid_data)), 
                                cp.abs(cp.max(valid_data)))),)
    return (1.0,)

def tpi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """TPI/LRM用の正規化"""
    max_abs = stats[0]
    if max_abs > 0:
        return cp.clip(block / max_abs, -1, 1)
    return cp.zeros_like(block)

# Visual Saliency用
def visual_saliency_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Visual Saliency用の統計量計算（解像度適応型）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # より狭い範囲でコントラストを強調
        # 低い値を切り捨てて、特徴的な部分を強調
        low_p = float(cp.percentile(valid_data, 25))   # 5→25に変更
        high_p = float(cp.percentile(valid_data, 85))  # 95→85に変更
        
        # 範囲が狭すぎる場合でも、あまり広げすぎない
        if (high_p - low_p) < cp.std(valid_data) * 0.3:
            low_p = float(cp.percentile(valid_data, 15))   # 5→15
            high_p = float(cp.percentile(valid_data, 90))  # 95→90
            
        return (low_p, high_p)
    return (0.0, 1.0)

###############################################################################
# 2.1. RVI (Ridge-Valley Index) アルゴリズム
###############################################################################

def high_pass(block: cp.ndarray, *, sigma: float) -> cp.ndarray:
    """CuPy でガウシアンぼかし後に差分を取る高‑pass フィルタ（NaN対応）"""
    # NaNマスク処理
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        # NaNを周囲の値で埋める（一時的に）
        filled = cp.where(nan_mask, 0, block)
        # 有効なピクセルのマスクも作成
        valid_mask = (~nan_mask).astype(cp.float32)
        
        # ガウシアンフィルタを値と有効マスクの両方に適用
        blurred_values = gaussian_filter(filled * valid_mask, sigma=sigma, mode="nearest", truncate=4.0)
        blurred_weights = gaussian_filter(valid_mask, sigma=sigma, mode="nearest", truncate=4.0)
        
        # 重み付き平均でNaN領域を考慮したぼかし
        blurred = cp.where(blurred_weights > 0, blurred_values / blurred_weights, 0)
    else:
        blurred = gaussian_filter(block, sigma=sigma, mode="nearest", truncate=4.0)
    
    result = block - blurred
    
    # NaN位置を復元
    result = restore_nan(result, nan_mask)
        
    return result

def compute_rvi_efficient_block(block: cp.ndarray, *, 
                               radii: List[int] = [4, 16, 64], 
                               weights: Optional[List[float]] = None) -> cp.ndarray:
    """効率的なRVI計算（メモリ最適化版）"""
    nan_mask = cp.isnan(block)
    
    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    else:
        if not isinstance(weights, cp.ndarray):
            weights = cp.array(weights, dtype=cp.float32)
        if len(weights) != len(radii):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of radii ({len(radii)})")
    
    # 結果をインプレースで累積（メモリ効率向上）
    rvi_combined = cp.zeros_like(block, dtype=cp.float32)
    
    for radius, weight in zip(radii, weights):
        if radius <= 1:
            # 小さな半径の場合
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=1.0, mode='nearest')
        else:
            # 大きな半径の場合
            kernel_size = 2 * radius + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='reflect')
        
        # インプレース演算でメモリ効率向上
        rvi_combined += weight * (block - mean_elev)
        
        # 中間結果のメモリを明示的に解放（追加）
        del mean_elev
    
    # NaN処理
    rvi_combined = restore_nan(rvi_combined, nan_mask)
    
    return rvi_combined


def multiscale_rvi_efficient(gpu_arr: da.Array, *, 
                            radii: List[int], 
                            weights: Optional[List[float]] = None) -> da.Array:
    """効率的なマルチスケールRVI（Dask版）"""
    
    if not radii:
        raise ValueError("At least one radius value is required")
    
    # 最大半径に基づいてdepthを設定（Gaussianよりも大幅に小さい）
    max_radius = max(radii)
    depth = max_radius * 2 + 1  # 半径の2倍+1に変更
    
    # 単一のmap_overlapで全スケールを計算（効率的）
    result = gpu_arr.map_overlap(
        compute_rvi_efficient_block,
        depth=depth,
        boundary="reflect",
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        radii=radii,
        weights=weights
    )
    
    return result


class RVIAlgorithm(DaskAlgorithm):
    """Ridge-Valley Indexアルゴリズム（効率的実装）"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:        
        # 新しい効率的な半径ベース
        radii = params.get('radii', None)
        weights = params.get('weights', None)
        
        # 自動決定
        if radii is None:
            pixel_size = params.get('pixel_size', 1.0)
            radii = self._determine_optimal_radii(pixel_size)
        max_radius = max(radii)
        rvi = multiscale_rvi_efficient(gpu_arr, radii=radii, weights=weights)
        
        # グローバル統計を使用 統計量を計算
        stats = compute_global_stats(
            rvi,  # 正規化前のRVI結果
            rvi_stat_func,
            compute_rvi_efficient_block,
            {'radii': radii, 'weights': weights},
            params.get('downsample_factor', None),
            depth=max_radius * 2 + 1
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        # 正規化を適用
        return rvi.map_blocks(
            lambda block: apply_global_normalization(block, rvi_norm_func, stats),
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
    
    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """ピクセルサイズに基づいて最適な半径を決定"""
        # 実世界の距離（メートル）をピクセルに変換
        target_distances = [5, 20, 80, 320]  # メートル単位
        radii = []
        
        for dist in target_distances:
            radius = int(dist / pixel_size)
            # 現実的な範囲に制限
            radius = max(2, min(radius, 256))
            radii.append(radius)
        
        # 重複を削除してソート
        radii = sorted(list(set(radii)))
        
        # 最大4つまでに制限
        if len(radii) > 4:
            # 対数的に分布するように選択
            # CuPy配列をPythonリストに変換してからアクセス
            indices = cp.logspace(0, cp.log10(len(radii)-1), 4).astype(int)
            indices_list = indices.get()  # CuPy配列をホストメモリ（NumPy配列）に転送
            radii = [radii[int(i)] for i in indices_list]
        
        return radii
    
    def get_default_params(self) -> dict:
        return {
            'mode': 'radius',  # デフォルトは効率的な半径モード
            'radii': None,     # Noneの場合は自動決定
            'weights': None,   # Noneの場合は均等重み
            'sigmas': None,    # 従来モード用（互換性）
            'agg': 'mean',     # 従来モード用（互換性）
            'auto_sigma': False,  # 従来のsigma自動決定は無効化
        }

###############################################################################
# 4. マルチスケール計算 (sigma 複数 → 集約) - 改善版
###############################################################################

def multiscale_rvi(gpu_arr: da.Array, *, sigmas: List[float], agg: str, 
                   show_progress: bool = True) -> da.Array:
    """複数 σ の RVI を計算し、集約 (メモリ効率版)"""
    
    # 最初のsigmaで初期化
    if not sigmas:
        raise ValueError("At least one sigma value is required")
    
    iterator = tqdm(sigmas, desc="Computing scales") if show_progress else sigmas
    
    result = None
    
    for i, sigma in enumerate(iterator):
        # 大きなsigmaに対してdepthを制限
        depth = min(int(4 * sigma), Constants.MAX_DEPTH)  # 最大depth=200に制限
        
        # 各sigmaを個別に計算（メモリ効率向上）
        hp = da.map_overlap(
            high_pass,
            gpu_arr,
            depth=depth,
            boundary="reflect",
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            sigma=sigma,
        )
        
        if i == 0:
            # 最初のsigma
            if agg == "stack":
                result = da.expand_dims(hp, axis=0)
            else:
                result = hp
        else:
            # 2番目以降のsigma
            if agg == "stack":
                result = da.concatenate([result, da.expand_dims(hp, axis=0)], axis=0)
            elif agg == "mean":
                result = (result * i + hp) / (i + 1)  # 累積平均
            elif agg == "min":
                result = da.minimum(result, hp)
            elif agg == "max":
                result = da.maximum(result, hp)
            elif agg == "sum":
                result = result + hp
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")
        
        # メモリクリーンアップ
        del hp
    
    return result
    
###############################################################################
# 2.2. Hillshade アルゴリズム
###############################################################################

def compute_hillshade_block(block: cp.ndarray, *, azimuth: float = Constants.DEFAULT_AZIMUTH, 
                           altitude: float = Constants.DEFAULT_ALTITUDE, z_factor: float = 1.0,
                           pixel_size: float = 1.0) -> cp.ndarray:
    """1ブロックに対するHillshade計算"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 方位角と高度角をラジアンに変換
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    
    # 勾配計算（中央差分）- NaNを含む場合は隣接値で補間
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=z_factor, pixel_size=pixel_size)
    
    # 勾配と傾斜角
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    
    # アスペクト（斜面方位）
    aspect = cp.arctan2(-dy, dx)  # 北を0°とする座標系
    
    # 光源ベクトルとの角度差
    aspect_diff = aspect - azimuth_rad
    
    # Hillshade計算（Lambertian reflectance model）
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect_diff)
    
    # 0-255の範囲に正規化（Hillshadeは例外的に0-255出力）
    hillshade = cp.clip(hillshade, -1, 1)
    hillshade = ((hillshade + 1) / 2 * 255).astype(cp.float32)
    
    # NaN処理
    hillshade = restore_nan(hillshade, nan_mask)
    
    return hillshade

class HillshadeAlgorithm(DaskAlgorithm):
    """Hillshadeアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        azimuth = params.get('azimuth', Constants.DEFAULT_AZIMUTH)
        altitude = params.get('altitude', Constants.DEFAULT_ALTITUDE)
        z_factor = params.get('z_factor', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        multiscale = params.get('multiscale', False)
        sigmas = params.get('sigmas', [1])  # Hillshadeでのマルチスケール用
        agg = params.get('agg', 'mean')
        
        if multiscale and len(sigmas) > 1:
            # マルチスケールHillshade
            results = []
            for sigma in sigmas:
                # まずスムージング
                if sigma > 1:
                    depth = int(4 * sigma)
                    smoothed = gpu_arr.map_overlap(
                        lambda x, *, sigma=sigma: gaussian_filter(x, sigma=sigma, mode='nearest'),
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32)
                    )
                else:
                    smoothed = gpu_arr
                
                # Hillshade計算
                hs = smoothed.map_overlap(
                    compute_hillshade_block,
                    depth=1,
                    boundary='reflect',
                    dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    azimuth=azimuth,
                    altitude=altitude,
                    z_factor=z_factor,
                    pixel_size=pixel_size
                )
                results.append(hs)

            # 大規模データの場合、定期的にGCを実行（追加）
            if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
                import gc
                gc.collect()
            
            # 集約
            stacked = da.stack(results, axis=0)
            if agg == "stack":
                return stacked
            elif agg == "mean":
                return da.mean(stacked, axis=0)
            elif agg == "min":
                return da.min(stacked, axis=0)
            elif agg == "max":
                return da.max(stacked, axis=0)
            else:
                return da.mean(stacked, axis=0)
        else:
            # 大規模データの場合、定期的にGCを実行（追加）
            if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
                import gc
                gc.collect()

            # 単一スケールHillshade
            return gpu_arr.map_overlap(
                compute_hillshade_block,
                depth=1,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                azimuth=azimuth,
                altitude=altitude,
                z_factor=z_factor,
                pixel_size=pixel_size
            )
    
    def get_default_params(self) -> dict:
        return {
            'azimuth': Constants.DEFAULT_AZIMUTH,
            'altitude': Constants.DEFAULT_ALTITUDE,
            'z_factor': 1.0,
            'pixel_size': 1.0,
            'multiscale': False,
            'sigmas': [1],
            'agg': 'mean'
        }

###############################################################################
# 2.3. Slope アルゴリズム（拡張例）
###############################################################################

###############################################################################
# 2.8. Visual Saliency (視覚的顕著性) アルゴリズム
###############################################################################
def visual_saliency_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Visual Saliency用の統計量計算（解像度適応型）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # データの分布の広がりを確認
        std_val = float(cp.std(valid_data))
        mean_val = float(cp.mean(valid_data))
        
        # 分布が狭い場合（低解像度データでよくある）はパーセンタイルを調整
        if std_val < mean_val * 0.1:  # 変動係数が小さい場合
            return (float(cp.percentile(valid_data, 20)), 
                    float(cp.percentile(valid_data, 80)))
        else:
            return (float(cp.percentile(valid_data, 5)), 
                    float(cp.percentile(valid_data, 95)))
    return (0.0, 1.0)

def compute_visual_saliency_block(block: cp.ndarray, *, scales: List[float] = [2, 4, 8, 16],
                                pixel_size: float = 1.0,
                                normalize: bool = True,
                                norm_min: float = None,
                                norm_max: float = None) -> cp.ndarray:
    """視覚的顕著性の計算（改善版）"""
    nan_mask = cp.isnan(block)
    resolution_class = classify_resolution(pixel_size)
    
    # 解像度に基づいてスケールを完全に再計算（改善版）
    if scales == [2, 4, 8, 16]:  # デフォルト値の場合のみ
        if resolution_class == 'ultra_high':
            scales = [2, 4, 8, 16]
        elif resolution_class == 'very_high':
            scales = [2, 4, 8, 12]
        elif resolution_class == 'high':
            scales = [1.5, 3, 6, 10]
        elif resolution_class == 'medium':
            scales = [1, 2, 4, 8]
        elif resolution_class == 'low':
            # 10m解像度はここに該当
            scales = [0.5, 1, 2, 4]
        elif resolution_class == 'very_low':
            scales = [0.3, 0.6, 1.2, 2.4]
        else:  # ultra_low
            scales = [0.2, 0.4, 0.8, 1.6]
    
    # 基本の勾配を最初に計算
    dy_orig, dx_orig, _ = handle_nan_for_gradient(block, scale=1.0, pixel_size=pixel_size)
    gradient_mag_base = cp.sqrt(dx_orig**2 + dy_orig**2)
    
    # 解像度に応じた勾配スケーリング（改善版）
    gradient_scale = get_gradient_scale_factor(pixel_size, 'visual_saliency')
    gradient_mag_base = gradient_mag_base * gradient_scale
    
    gradient_mag_base = cp.where(cp.isnan(gradient_mag_base), 0, gradient_mag_base)
    
    # マルチスケール処理
    saliency_maps = []
    
    for scale in scales:
        # 局所的なコントラスト
        center_sigma = scale
        surround_sigma = scale * 2
        
        if nan_mask.any():
            center, _ = handle_nan_with_gaussian(block, sigma=center_sigma, mode='nearest')
            surround, _ = handle_nan_with_gaussian(block, sigma=surround_sigma, mode='nearest')
        else:
            center = gaussian_filter(block, sigma=center_sigma, mode='nearest')
            surround = gaussian_filter(block, sigma=surround_sigma, mode='nearest')
        
        # 差分の絶対値（改善版：スケールで正規化）
        contrast = cp.abs(center - surround) / (scale + 1.0)
        
        # 勾配の強度
        if scale > 1:
            if nan_mask.any():
                filled_grad = cp.where(nan_mask, 0, gradient_mag_base)
                valid = (~nan_mask).astype(cp.float32)
                
                smoothed_values = gaussian_filter(filled_grad * valid, sigma=scale/2, mode='nearest')
                smoothed_weights = gaussian_filter(valid, sigma=scale/2, mode='nearest')
                gradient_mag = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)
            else:
                gradient_mag = gaussian_filter(gradient_mag_base, sigma=scale/2, mode='nearest')
        else:
            gradient_mag = gradient_mag_base
        
        # 解像度に応じた特徴の組み合わせ（改善版）
        if resolution_class in ['ultra_high', 'very_high']:
            feature = contrast * 0.6 + gradient_mag * 0.4
        elif resolution_class in ['high', 'medium']:
            feature = contrast * 0.5 + gradient_mag * 0.5
        elif resolution_class == 'low':
            # 10m解像度 - コントラストの重みを増やす
            feature = contrast * 0.7 + gradient_mag * 0.3
        else:  # very_low, ultra_low
            feature = contrast * 0.8 + gradient_mag * 0.2
            
        feature = cp.where(cp.isnan(feature), 0, feature)
        saliency_maps.append(feature)
    
    # スケール間での正規化と統合（改善版）
    combined_saliency = cp.zeros_like(block)
    
    # 各マップを正規化して統合
    for i, smap in enumerate(saliency_maps):
        if nan_mask.any():
            valid_smap = smap[~nan_mask]
        else:
            valid_smap = smap.ravel()
            
        if len(valid_smap) > 0:
            # ロバストな正規化（外れ値に強い）
            p5 = cp.percentile(valid_smap, 5)
            p95 = cp.percentile(valid_smap, 95)
            if p95 > p5:
                normalized = (smap - p5) / (p95 - p5)
                normalized = cp.clip(normalized, 0, 1)
                # スケールに応じた重み付け
                scale_weight = 1.0 / len(saliency_maps)  # 1.0 / (i + 1) から変更
                combined_saliency += smap * scale_weight
            else:
                combined_saliency += 0.5
    
    # 重みの合計で正規化
    weight_sum = sum(1.0 / (i + 1) for i in range(len(saliency_maps)))
    combined_saliency /= weight_sum
    valid_count = 0
    
    for smap in saliency_maps:
        # 各マップを正規化
        if nan_mask.any():
            valid_smap = smap[~nan_mask]
        else:
            valid_smap = smap.ravel()
            
        if len(valid_smap) > 0:
            min_val = cp.min(valid_smap)
            max_val = cp.max(valid_smap)
            if max_val > min_val:
                normalized = (smap - min_val) / (max_val - min_val)
                combined_saliency += normalized
                valid_count += 1
    
    # スケール間での正規化と統合（改善版）
    combined_saliency = cp.zeros_like(block)
    
    # 各マップを統合（正規化はnormalizeフラグで制御）
    for i, smap in enumerate(saliency_maps):
        scale_weight = 1.0 / (i + 1)
        combined_saliency += smap * scale_weight
    
    # 重みの合計で正規化
    weight_sum = sum(1.0 / (i + 1) for i in range(len(saliency_maps)))
    combined_saliency /= weight_sum
    
    # 正規化処理を条件分岐に変更
    if normalize:
        if norm_min is not None and norm_max is not None:
            # 提供された統計量で正規化
            if norm_max > norm_min:
                result = (combined_saliency - norm_min) / (norm_max - norm_min)
                result = cp.clip(result, 0, 1)
            else:
                result = cp.full_like(block, 0.5)
        else:
            # 従来のローカル正規化（互換性のため残す）
            valid_result = combined_saliency[~nan_mask] if nan_mask.any() else combined_saliency.ravel()
            if len(valid_result) > 0:
                min_val = cp.min(valid_result)
                max_val = cp.max(valid_result)
                if max_val > min_val:
                    result = (combined_saliency - min_val) / (max_val - min_val)
                else:
                    result = cp.full_like(combined_saliency, 0.5)
            else:
                result = cp.full_like(combined_saliency, 0.5)
        
        # ガンマ補正（正規化された場合のみ）
        result = cp.power(result, Constants.DEFAULT_GAMMA)
    else:
        # 正規化なし
        result = combined_saliency
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class VisualSaliencyAlgorithm(DaskAlgorithm):
    """視覚的顕著性アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [2, 4, 8, 16])
        max_scale = max(scales)

        # 統計量を計算（新しい共通関数を使用）
        stats = compute_global_stats(
            gpu_arr,
            visual_saliency_stat_func,
            compute_visual_saliency_block,
            {
                'scales': scales,
                'pixel_size': params.get('pixel_size', 1.0),
                'normalize': False
            },
            downsample_factor=params.get('downsample_factor', None),
            depth=int(max_scale * 8)
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        # フルサイズで処理
        return gpu_arr.map_overlap(
            compute_visual_saliency_block,
            depth=int(max_scale * 8),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales,
            pixel_size=params.get('pixel_size', 1.0),
            normalize=True,
            norm_min=stats[0],
            norm_max=stats[1]
        )
    
    def get_default_params(self) -> dict:
        return {
            'scales': [2, 4, 8, 16],
            'pixel_size': 1.0,
            'downsample_factor': None,       # ダウンサンプル係数
            'verbose': False               # デバッグ出力
        }

###############################################################################
# 2.9. NPR Edges (非写実的レンダリング輪郭線) アルゴリズム
###############################################################################

def compute_npr_edges_block(block: cp.ndarray, *, edge_sigma: float = 1.0,
                          threshold_low: float = 0.1, threshold_high: float = 0.3,
                          pixel_size: float = 1.0) -> cp.ndarray:
    """NPRスタイルの輪郭線抽出（改良版v2）"""
    nan_mask = cp.isnan(block)
    resolution_class = classify_resolution(pixel_size)
    
    # 解像度に応じたスムージング
    if resolution_class in ['ultra_high', 'very_high']:
        adaptive_sigma = 0.5
    elif resolution_class in ['high', 'medium']:
        adaptive_sigma = 1.0
    elif resolution_class == 'low':  # 10m
        adaptive_sigma = 0.5  # スムージングを弱める
    else:  # very_low, ultra_low
        adaptive_sigma = 0.3
    
    # edge_sigmaが明示的に指定されている場合はそれを使用
    if edge_sigma != 1.0:
        adaptive_sigma = edge_sigma
    
    # ノイズ除去（最小限に）
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(filled, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = filled
    else:
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(block, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = block
    
    # 方法1: Sobelフィルタを使用した勾配計算（pixel_sizeに依存しない）
    # Sobelカーネル
    sobel_x = cp.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=cp.float32) / 8.0
    sobel_y = cp.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=cp.float32) / 8.0
    
    # 畳み込みによる勾配計算
    dx = convolve(smoothed, sobel_x, mode='nearest')
    dy = convolve(smoothed, sobel_y, mode='nearest')
    
    # 勾配の大きさ（標高差として扱う）
    gradient_mag = cp.sqrt(dx**2 + dy**2)
    
    # 方法2: 解像度適応型の勾配強調
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        # 低解像度では局所的な標高差を直接計算
        # 3x3近傍での最大標高差を計算
        local_max = maximum_filter(smoothed, size=3, mode='nearest')
        local_min = minimum_filter(smoothed, size=3, mode='nearest')
        local_range = local_max - local_min
        
        # 勾配と局所範囲の組み合わせ
        gradient_mag = cp.maximum(gradient_mag, local_range * 0.3)
    
    gradient_dir = cp.arctan2(dy, dx)
    
    # 適応的な閾値設定
    valid_grad = gradient_mag[~nan_mask] if nan_mask.any() else gradient_mag.ravel()
    if len(valid_grad) > 0:
        # 統計量に基づく閾値
        grad_std = cp.std(valid_grad)
        grad_mean = cp.mean(valid_grad)
        
        # 解像度に応じた閾値戦略
        if resolution_class in ['low', 'very_low', 'ultra_low']:
            # 低解像度：平均値を基準に
            base_threshold = grad_mean
            threshold_range = grad_std * 1.5
        else:
            # 高解像度：パーセンタイルベース
            base_threshold = cp.percentile(valid_grad, 50)
            threshold_range = cp.percentile(valid_grad, 90) - base_threshold
        
        # ユーザー指定の閾値で調整
        actual_threshold_low = base_threshold + threshold_range * threshold_low * 0.5
        actual_threshold_high = base_threshold + threshold_range * threshold_high
        
        # 最小閾値を保証（完全に白い画像を防ぐ）
        min_threshold = grad_mean * 0.1
        actual_threshold_low = cp.maximum(actual_threshold_low, min_threshold)
        actual_threshold_high = cp.maximum(actual_threshold_high, min_threshold * 2)
    else:
        actual_threshold_low = 0.1
        actual_threshold_high = 0.3
    
    # 非最大値抑制（簡易版）
    angle = gradient_dir * 180.0 / cp.pi
    angle[angle < 0] += 180
    
    nms = gradient_mag.copy()
    
    # 8方向での非最大値抑制（元のコードと同じ）
    # 0度と180度方向
    shifted_pos = cp.roll(gradient_mag, 1, axis=1)
    shifted_neg = cp.roll(gradient_mag, -1, axis=1)
    mask = ((angle < 22.5) | (angle >= 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 45度方向
    shifted_pos = cp.roll(cp.roll(gradient_mag, 1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, -1, axis=0), 1, axis=1)
    mask = ((angle >= 22.5) & (angle < 67.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 90度方向
    shifted_pos = cp.roll(gradient_mag, 1, axis=0)
    shifted_neg = cp.roll(gradient_mag, -1, axis=0)
    mask = ((angle >= 67.5) & (angle < 112.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 135度方向
    shifted_pos = cp.roll(cp.roll(gradient_mag, -1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, 1, axis=0), 1, axis=1)
    mask = ((angle >= 112.5) & (angle < 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # ダブルスレッショルド
    strong = nms > actual_threshold_high
    weak = (nms > actual_threshold_low) & (nms <= actual_threshold_high)
    
    # エッジの強調（NPRスタイル）
    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5
    
    # ヒステリシス処理（接続性の改善）
    for _ in range(3):  # 3回に増やして接続を強化
        dilated = cp.maximum(
            cp.maximum(
                cp.roll(edges, 1, axis=0),
                cp.roll(edges, -1, axis=0)
            ),
            cp.maximum(
                cp.roll(edges, 1, axis=1),
                cp.roll(edges, -1, axis=1)
            )
        )
        
        # 対角方向も追加
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, 1, axis=0), 1, axis=1))
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, -1, axis=0), -1, axis=1))
        
        edges = cp.where(weak & (dilated > 0.5), 1.0, edges)
    
    # 後処理：解像度に応じたエッジの太さ調整
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        # 低解像度ではエッジを少し太くする
        structure = cp.ones((3, 3))
        edges_binary = edges > 0.5
        edges_dilated = binary_dilation(edges_binary, structure=structure).astype(cp.float32)
        edges = cp.where(edges_dilated, cp.maximum(edges, 0.8), edges)
    
    # エッジ強度を調整
    edges = edges * 0.8
    
    # 輪郭線を反転（黒線で描画）
    result = 1.0 - edges
    
    # コントラスト調整（エッジをより見やすく）
    result = cp.clip(result, 0.2, 1.0)
    
    # ガンマ補正
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR輪郭線アルゴリズム（改良版）"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        edge_sigma = params.get('edge_sigma', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        threshold_low = params.get('threshold_low', 0.1)
        threshold_high = params.get('threshold_high', 0.3)
        
        # depthを3に増やす（Sobelフィルタと膨張処理のため）
        depth = 3
        
        if edge_sigma != 1.0:
            depth = max(depth, int(edge_sigma * 4 + 2))
                        
        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            edge_sigma=edge_sigma,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            pixel_size=pixel_size,
        )
    
    def get_default_params(self) -> dict:
        return {
            'edge_sigma': 1.0,
            'threshold_low': 0.1,
            'threshold_high': 0.3,
            'pixel_size': 1.0
        }

###############################################################################
# 2.10. Atmospheric Perspective (大気遠近法) アルゴリズム
###############################################################################

def compute_atmospheric_perspective_block(block: cp.ndarray, *, 
                                        depth_scale: float = 1000.0,
                                        haze_strength: float = 0.7,
                                        pixel_size: float = 1.0) -> cp.ndarray:
    """大気遠近法による深度表現"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 標高を深度として使用
    depth = block / depth_scale
    depth = cp.clip(depth, 0, 1)
    
    # Hillshadeの計算
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    aspect = cp.arctan2(-dy, dx)
    
    azimuth_rad = cp.radians(Constants.DEFAULT_AZIMUTH)
    altitude_rad = cp.radians(Constants.DEFAULT_ALTITUDE)
    
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad)
    hillshade = (hillshade + 1) / 2
    
    # 深度に応じたヘイズ（霞）効果
    haze = 1.0 - cp.exp(-haze_strength * depth)
    
    # ヘイズカラー（薄い青みがかった白）
    haze_color = 0.85  # 単一チャンネルなので明度のみ
    
    # 深度に応じてヘイズとブレンド
    result = hillshade * (1 - haze) + haze_color * haze
    
    # コントラストの調整（遠いほどコントラストが低下）
    contrast = 1.0 - depth * 0.7
    result = (result - 0.5) * contrast + 0.5
    result = cp.clip(result, 0, 1)
    
    # ガンマ補正
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class AtmosphericPerspectiveAlgorithm(DaskAlgorithm):
    """大気遠近法アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        depth_scale = params.get('depth_scale', 1000.0)
        haze_strength = params.get('haze_strength', 0.7)
        pixel_size = params.get('pixel_size', 1.0)

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_atmospheric_perspective_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            depth_scale=depth_scale,
            haze_strength=haze_strength,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'depth_scale': 1000.0,  # 最大標高値に応じて調整
            'haze_strength': 0.7,
            'pixel_size': 1.0
        }

###############################################################################
# 2.11. Ambient Occlusion (環境光遮蔽) アルゴリズム
###############################################################################

def compute_ambient_occlusion_block(block: cp.ndarray, *, 
                                    num_samples: int = 16,
                                    radius: float = 10.0,
                                    intensity: float = 1.0,
                                    pixel_size: float = 1.0) -> cp.ndarray:
    """スクリーン空間環境光遮蔽（SSAO）の地形版（高速ベクトル化版）"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # サンプリング方向を事前計算
    angles = cp.linspace(0, 2 * cp.pi, num_samples, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    # 距離のサンプリング
    r_factors = cp.array([0.25, 0.5, 0.75, 1.0])
    
    # 全サンプル点の座標を事前計算（ベクトル化）
    occlusion_total = cp.zeros((h, w), dtype=cp.float32)
    sample_count = cp.zeros((h, w), dtype=cp.float32)  # 有効なサンプル数をカウント
    
    # バッチ処理で高速化
    for r_factor in r_factors:
        r = radius * r_factor
        
        # 全方向の変位を一度に計算（ピクセル単位に変換）
        # 修正: radiusはピクセル単位として扱う（pixel_sizeで除算しない）
        dx_all = cp.round(r * directions[:, 0]).astype(int)
        dy_all = cp.round(r * directions[:, 1]).astype(int)
        
        for i in range(num_samples):
            # CuPy配列から個別の値を取得して明示的にintに変換
            dx = int(dx_all[i])
            dy = int(dy_all[i])
            
            if dx == 0 and dy == 0:
                continue
            
            # 必要な方向のみパディング
            pad_left = max(0, -dx)
            pad_right = max(0, dx)
            pad_top = max(0, -dy)
            pad_bottom = max(0, dy)
            
            # パディング（edge modeを使用）
            padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='edge')
            
            # シフト
            start_y = pad_top + dy
            start_x = pad_left + dx
            shifted = padded[start_y:start_y+h, start_x:start_x+w]
            
            # 高さの差と遮蔽角度
            height_diff = shifted - block
            # 修正: 実際の距離（メートル）を使用
            distance = r * pixel_size
            occlusion_angle = cp.arctan(height_diff / distance)
            
            # 正の角度のみを遮蔽として扱う（修正: より適切な遮蔽の計算）
            # 角度を0-1の範囲に正規化（最大45度を1とする）
            max_angle = cp.pi / 4  # 45度
            occlusion = cp.maximum(0, occlusion_angle) / max_angle
            occlusion = cp.minimum(occlusion, 1.0)  # 1を超えないようにクリップ
            
            # 距離による減衰（修正: より緩やかな減衰）
            distance_factor = 1.0 - (r_factor * 0.3)  # 0.5から0.3に変更
            
            # 遮蔽の累積（NaNを除外）
            valid = ~(cp.isnan(shifted) | nan_mask)
            occlusion_total += cp.where(valid, 
                                      occlusion * distance_factor,
                                      0)
            sample_count += cp.where(valid, 1.0, 0)
    
    # 正規化（修正: 有効なサンプル数で除算）
    # ゼロ除算を防ぐ
    sample_count = cp.maximum(sample_count, 1.0)
    
    # 平均遮蔽を計算（修正: すでに0-1の範囲）
    mean_occlusion = occlusion_total / sample_count
    
    # AOの計算（修正: より直接的な計算）
    # 遮蔽が多いほど暗くなる（0に近づく）
    ao = 1.0 - mean_occlusion * intensity
    ao = cp.clip(ao, 0, 1)
    
    # スムージング（NaN考慮）
    if nan_mask.any():
        filled_ao = cp.where(nan_mask, 1.0, ao)  # NaN領域は明るく（遮蔽なし）
        ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
    else:
        ao = gaussian_filter(ao, sigma=1.0, mode='nearest')
    
    # ガンマ補正
    result = cp.power(ao, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)


class AmbientOcclusionAlgorithm(DaskAlgorithm):
    """環境光遮蔽アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        num_samples = params.get('num_samples', 16)
        radius = params.get('radius', 10.0)
        intensity = params.get('intensity', 1.0)
        pixel_size = params.get('pixel_size', 1.0)

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        # 修正: radiusをピクセル単位として扱うので、pixel_sizeで除算しない
        # ユーザーが指定するradiusは既にピクセル単位
        return gpu_arr.map_overlap(
            compute_ambient_occlusion_block,
            depth=int(radius + 1),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            num_samples=num_samples,
            radius=radius,
            intensity=intensity,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'num_samples': 16,
            'radius': 10.0,     # ピクセル単位の探索半径
            'intensity': 1.0,
            'pixel_size': 1.0
        }

###############################################################################
# 2.12. TPI (Topographic Position Index) アルゴリズム
###############################################################################

def compute_tpi_block(block: cp.ndarray, *, radius: int = 10,
                      std_global: float = None) -> cp.ndarray:
    """地形位置指数の計算（尾根・谷・斜面の分類）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 円形カーネルの作成
    y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = (x**2 + y**2) <= radius**2
    kernel = kernel.astype(cp.float32)
    kernel /= kernel.sum()
    
    # 周囲の平均標高を計算（NaN考慮）
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        
        sum_values = convolve(filled * valid, kernel, mode='reflect')
        sum_weights = convolve(valid, kernel, mode='reflect')
        mean_elev = cp.where(sum_weights > 0, sum_values / sum_weights, 0)
    else:
        mean_elev = convolve(block, kernel, mode='reflect')
    
    # TPIは中心と周囲の差
    tpi = block - mean_elev
    
    # グローバル統計量で正規化
    if std_global is not None:
        if std_global > 0:
            tpi = tpi / (3 * std_global)
    
    # ガンマ補正は行わない（正負の値を保持）
    # 結果は-1から+1の範囲
    result = cp.clip(tpi, -1, 1)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class TPIAlgorithm(DaskAlgorithm):
    """地形位置指数アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radius = params.get('radius', 10)
        
        # 統計量を計算
        stats = compute_global_stats(
            gpu_arr,
            lambda data: tpi_lrm_stat_func(compute_tpi_block(data, radius=radius, std_global=None)),
            compute_tpi_block,
            {'radius': radius, 'std_global': None},
            downsample_factor=None,
            depth=radius+1
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_tpi_block,
            depth=radius+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radius=radius,
            std_global=stats[0]
        )
    
    def get_default_params(self) -> dict:
        return {
            'radius': 10,  # 解析半径（ピクセル）
        }

###############################################################################
# 2.13. LRM (Local Relief Model) アルゴリズム
###############################################################################

def compute_lrm_block(block: cp.ndarray, *, kernel_size: int = 25,
                     pixel_size: float = 1.0,
                     std_global: float = None) -> cp.ndarray:  # 追加
    """局所起伏モデルの計算（微地形の強調）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 大規模な地形トレンドを計算（NaN考慮）
    sigma = kernel_size / 3.0
    trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
    
    # 微地形の抽出
    lrm = block - trend
    
    if std_global is not None:
        # グローバル統計量で正規化
        if std_global > 0:
            lrm = lrm / (3 * std_global)
    
    # 結果は-1から+1の範囲
    result = cp.clip(lrm, -1, 1)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class LRMAlgorithm(DaskAlgorithm):
    """局所起伏モデルアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        kernel_size = params.get('kernel_size', 25)
        
        # 統計量を計算
        stats = compute_global_stats(
            gpu_arr,
            lambda data: tpi_lrm_stat_func(compute_lrm_block(data, kernel_size=kernel_size)),
            compute_lrm_block,
            {'kernel_size': kernel_size},
            downsample_factor=None,
            depth=int(kernel_size * 2)
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_lrm_block,
            depth=int(kernel_size * 2),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            kernel_size=kernel_size,
            std_global=stats[0]
        )
    
    def get_default_params(self) -> dict:
        return {
            'kernel_size': 25,  # トレンド除去のカーネルサイズ
        }

###############################################################################
# 2.14. Openness (開度) アルゴリズム - 簡易高速版
###############################################################################
# 効率的なベクトル化版
def compute_openness_vectorized(block: cp.ndarray, *, 
                              openness_type: str = 'positive',
                              num_directions: int = 16,
                              max_distance: int = 50,
                              pixel_size: float = 1.0) -> cp.ndarray:
    """開度の計算（最適化版）"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # 方向ベクトルの事前計算
    angles = cp.linspace(0, 2 * cp.pi, num_directions, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    # 初期化
    init_val = -cp.pi/2 if openness_type == 'positive' else cp.pi/2
    max_angles = cp.full((h, w), init_val, dtype=cp.float32)
    
    # 距離サンプルを事前計算（整数値に）
    distances = cp.unique(cp.linspace(0.1, 1.0, 10) * max_distance).astype(int)
    distances = distances[distances > 0]  # 0を除外
    
    # パディング値の決定
    pad_value = Constants.NAN_FILL_VALUE_POSITIVE if openness_type == 'positive' else Constants.NAN_FILL_VALUE_NEGATIVE
    
    for r in distances:
        # 全方向のオフセットを一度に計算
        offsets = cp.round(r * directions).astype(int)
        
        for offset in offsets:
            offset_x, offset_y = offset
            # CuPy配列要素をPython intに明示的に変換
            offset_x = int(offset_x)
            offset_y = int(offset_y)
            
            if offset_x == 0 and offset_y == 0:
                continue
            
            # パディングサイズの計算（簡潔に）
            pad_left = max(0, -offset_x)
            pad_right = max(0, offset_x)
            pad_top = max(0, -offset_y)
            pad_bottom = max(0, offset_y)
            
            # パディング
            if nan_mask.any():
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='constant', constant_values=pad_value)
            else:
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='edge')
            
            # 以下は変更なし
            # シフト（簡潔な記述）
            start_y = pad_top + offset_y
            start_x = pad_left + offset_x
            shifted = padded[start_y:start_y+h, start_x:start_x+w]
            
            # 角度計算と更新
            angle = cp.arctan((shifted - block) / (r * pixel_size))
            
            if openness_type == 'positive':
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.maximum(max_angles, angle), max_angles)
            else:
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.minimum(max_angles, angle), max_angles)
    
    # 開度の計算と正規化
    openness = (cp.pi/2 - max_angles if openness_type == 'positive' 
                else cp.pi/2 + max_angles)
    openness = cp.clip(openness / (cp.pi/2), 0, 1)
    
    # ガンマ補正とNaN処理
    result = cp.power(openness, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class OpennessAlgorithm(DaskAlgorithm):
    """開度アルゴリズム（簡易高速版）"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        openness_type = params.get('openness_type', 'positive')
        num_directions = params.get('num_directions', 16)
        pixel_size = params.get('pixel_size', 1.0)

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()

        return gpu_arr.map_overlap(
            compute_openness_vectorized,  # ベクトル化版を使用
            depth=max_distance+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            openness_type=openness_type,
            num_directions=num_directions,
            max_distance=max_distance,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'openness_type': 'positive',  # 'positive' or 'negative'
            'num_directions': 16,  # 探索方向数（少なくして高速化）
            'max_distance': 50,    # 最大探索距離（ピクセル）
            'pixel_size': 1.0
        }

def compute_slope_block(block: cp.ndarray, *, unit: str = 'degree',
                       pixel_size: float = 1.0) -> cp.ndarray:
    """1ブロックに対する勾配計算"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 勾配計算（NaN考慮）
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    # 勾配の大きさ
    slope_rad = cp.arctan(cp.sqrt(dx**2 + dy**2))
    
    # 単位変換
    if unit == 'degree':
        slope = cp.degrees(slope_rad)
    elif unit == 'percent':
        slope = cp.tan(slope_rad) * 100
    else:  # radians
        slope = slope_rad
    
    # NaN処理
    slope = restore_nan(slope, nan_mask)
    
    return slope.astype(cp.float32)

class SlopeAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        unit = params.get('unit', 'degree')
        pixel_size = params.get('pixel_size', 1.0)
        normalize = params.get('normalize', False)  # 正規化オプションを追加
        
        if normalize and unit == 'degree':
            # グローバル統計を計算
            stats = compute_global_stats(
                gpu_arr,
                lambda data: (float(cp.min(data[~cp.isnan(data)])), 
                             float(cp.max(data[~cp.isnan(data)]))),
                compute_slope_block,
                {'unit': unit, 'pixel_size': pixel_size},
                downsample_factor=None,
                depth=1
            )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_slope_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            unit=unit,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'unit': 'degree',  # 'degree', 'percent', 'radians'
            'pixel_size': 1.0
        }

###############################################################################
# 2.3. Specular (金属光沢効果) アルゴリズム
###############################################################################

def compute_specular_block(block: cp.ndarray, *, roughness_scale: float = 50.0,
                          shininess: float = 20.0, pixel_size: float = 1.0,
                          light_azimuth: float = Constants.DEFAULT_AZIMUTH, light_altitude: float = Constants.DEFAULT_ALTITUDE) -> cp.ndarray:
    """金属光沢効果の計算（Cook-Torranceモデルの簡略版）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 法線ベクトルの計算
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    normal = cp.stack([-dx, -dy, cp.ones_like(dx)], axis=-1)
    normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
    
    # ラフネスの計算（局所的な標高の分散）
    kernel_size = max(3, int(roughness_scale))
    
    # NaN対応のラフネス計算
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        
        # 平均と平均二乗を計算（NaN考慮）
        mean_values = uniform_filter(filled * valid, size=kernel_size, mode='constant')
        mean_weights = uniform_filter(valid, size=kernel_size, mode='constant')
        mean_filter = cp.where(mean_weights > 0, mean_values / mean_weights, 0)
        
        sq_values = uniform_filter((filled**2) * valid, size=kernel_size, mode='constant')
        mean_sq_filter = cp.where(mean_weights > 0, sq_values / mean_weights, 0)
    else:
        # 平均と平均二乗を計算
        mean_filter = uniform_filter(block, size=kernel_size, mode='constant')
        mean_sq_filter = uniform_filter(block**2, size=kernel_size, mode='constant')
    
    # 標準偏差 = sqrt(E[X^2] - E[X]^2)
    roughness = cp.sqrt(cp.maximum(mean_sq_filter - mean_filter**2, 0))
    
    # ラフネスを正規化（より適切な範囲に）
    roughness_valid = roughness[~nan_mask] if nan_mask.any() else roughness
    if len(roughness_valid) > 0 and cp.max(roughness_valid) > 0:
        roughness = roughness / cp.max(roughness_valid)
        # 最小値を設定して完全な鏡面反射を防ぐ
        roughness = cp.clip(roughness, 0.1, 1.0)
    else:
        roughness = cp.full_like(block, 0.5)
    
    # 光源方向
    light_az_rad = cp.radians(light_azimuth)
    light_alt_rad = cp.radians(light_altitude)
    light_dir = cp.array([
        cp.sin(light_az_rad) * cp.cos(light_alt_rad),
        -cp.cos(light_az_rad) * cp.cos(light_alt_rad),
        cp.sin(light_alt_rad)
    ])
    
    # 視線方向（真上から）
    view_dir = cp.array([0, 0, 1])
    
    # ハーフベクトル
    half_vec = (light_dir + view_dir) / cp.linalg.norm(light_dir + view_dir)
    
    # スペキュラー計算（ドット積を正しく計算）
    n_dot_h = cp.sum(normal * half_vec.reshape(1, 1, 3), axis=-1)
    n_dot_h = cp.clip(n_dot_h, 0, 1)
    
    # より穏やかな指数を使用
    exponent = shininess * (1.0 - roughness * 0.8)  # roughnessが高いほど指数を下げる
    specular = cp.power(n_dot_h, exponent)
    
    # ディフューズ成分も追加（完全な黒を防ぐ）
    n_dot_l = cp.sum(normal * light_dir.reshape(1, 1, 3), axis=-1)
    n_dot_l = cp.clip(n_dot_l, 0, 1)
    diffuse = n_dot_l * 0.3  # ディフューズ成分を30%
    
    # 合成
    result = diffuse + specular * 0.7
    result = cp.clip(result, 0, 1)
    
    # ガンマ補正（より明るくするため、ガンマ値を調整）
    result = cp.power(result, 0.7)  # Constants.DEFAULT_GAMMAの代わりに0.7を使用
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class SpecularAlgorithm(DaskAlgorithm):
    """金属光沢効果アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        roughness_scale = params.get('roughness_scale', 50.0)
        shininess = params.get('shininess', 20.0)
        pixel_size = params.get('pixel_size', 1.0)
        light_azimuth = params.get('light_azimuth', Constants.DEFAULT_AZIMUTH)
        light_altitude = params.get('light_altitude', Constants.DEFAULT_ALTITUDE)
        
        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()

        return gpu_arr.map_overlap(
            compute_specular_block,
            depth=int(roughness_scale),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            roughness_scale=roughness_scale,
            shininess=shininess,
            pixel_size=pixel_size,
            light_azimuth=light_azimuth,
            light_altitude=light_altitude
        )
    
    def get_default_params(self) -> dict:
        return {
            'roughness_scale': 20.0,
            'shininess': 10.0,
            'light_azimuth': Constants.DEFAULT_AZIMUTH,
            'light_altitude': Constants.DEFAULT_ALTITUDE,
            'pixel_size': 1.0
        }

###############################################################################
# 2.4. Atmospheric Scattering (大気散乱光) アルゴリズム
###############################################################################

def compute_atmospheric_scattering_block(block: cp.ndarray, *, 
                                       scattering_strength: float = 0.5,
                                       intensity: float | None = None,
                                       pixel_size: float = 1.0) -> cp.ndarray:
    """大気散乱によるシェーディング（Rayleigh散乱の簡略版）"""
    # intensity は scattering_strength のエイリアス（後方互換）
    if intensity is not None:
        scattering_strength = intensity
        
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 法線計算
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    slope = cp.sqrt(dx**2 + dy**2)
    
    # 天頂角（法線と垂直方向のなす角）
    zenith_angle = cp.arctan(slope)
    
    # 大気の厚さ（簡略化：天頂角に比例）
    air_mass = 1.0 / (cp.cos(zenith_angle) + 0.001)  # ゼロ除算回避
    
    # Rayleigh散乱の近似
    scattering = 1.0 - cp.exp(-scattering_strength * air_mass)
    
    # 青みがかった散乱光を表現（単一チャンネルなので明度のみ）
    ambient = 0.4 + 0.6 * scattering
    
    # 通常のHillshadeと組み合わせ
    azimuth_rad = cp.radians(Constants.DEFAULT_AZIMUTH)
    altitude_rad = cp.radians(Constants.DEFAULT_ALTITUDE)
    aspect = cp.arctan2(-dy, dx)
    
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad)
    
    # 散乱光と直接光の合成
    result = ambient * 0.3 + hillshade * 0.7
    result = cp.clip(result, 0, 1)
    
    # ガンマ補正
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class AtmosphericScatteringAlgorithm(DaskAlgorithm):
    """大気散乱光アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scattering_strength = params.get('scattering_strength', 0.5)
        intensity = params.get('intensity', None)
        pixel_size = params.get('pixel_size', 1.0)

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            compute_atmospheric_scattering_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scattering_strength=scattering_strength,
            intensity=intensity,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'scattering_strength': 0.5,
            'pixel_size': 1.0
        }

###############################################################################
# 2.5. Multiscale Terrain (マルチスケール地形) アルゴリズム
###############################################################################

class MultiscaleDaskAlgorithm(DaskAlgorithm):
    """マルチスケール地形アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [1, 10, 50, 100])
        weights = params.get('weights', None)
        
        downsample_factor = params.get('downsample_factor', None)
        if downsample_factor is None:
            # 自動決定
            downsample_factor = determine_optimal_downsample_factor(
                gpu_arr.shape,
                algorithm_name='multiscale_terrain'
            )
            
        if weights is None:
            # デフォルト：スケールに反比例する重み
            weights = [1.0 / s for s in scales]
        
        # 重みを正規化
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        
        # 最大スケールに基づいて共通のdepthを決定
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)
        
        # 縮小版で統計量を計算
        downsampled = gpu_arr[::downsample_factor, ::downsample_factor]
        
        # 縮小版でマルチスケール処理
        results_small = []

        # 共通のdepthを計算（追加）
        max_scale_small = max([max(1, s // downsample_factor) for s in scales])
        common_depth_small = min(int(4 * max_scale_small), Constants.MAX_DEPTH)

        def create_weighted_combiner(weights):
            def weighted_combine_for_stats(*blocks):
                nan_mask = cp.isnan(blocks[0])
                result = cp.zeros_like(blocks[0])
                
                for i, block in enumerate(blocks):
                    valid = ~cp.isnan(block)
                    result[valid] += block[valid] * weights[i]
                
                result[nan_mask] = cp.nan
                return result
            return weighted_combine_for_stats
        
        for i, scale in enumerate(scales):
            scale_small = max(1, scale // downsample_factor)
            
            def compute_detail_small(block, *, scale):
                # scale=1でも最小限のスムージングを適用
                smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            
            detail_small = downsampled.map_overlap(
                compute_detail_small,
                depth=common_depth_small,  # 共通のdepthを使用
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                scale=scale_small
            )
            results_small.append(detail_small)
            
        # ループ後に一度だけ合成
        combined_small = da.map_blocks(
            create_weighted_combiner(weights),
            *results_small,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        ).compute()

        # 統計量を計算
        valid_data = combined_small[~cp.isnan(combined_small)]
        if len(valid_data) > 0:
            norm_min = float(cp.percentile(valid_data, 5))
            norm_max = float(cp.percentile(valid_data, 95))
        else:
            norm_min, norm_max = 0.0, 1.0
        
        if params.get('verbose', False):
            print(f"Multiscale Terrain global stats: min={norm_min:.3f}, max={norm_max:.3f}")
        
        # Step 2: フルサイズで処理
        results = []
        for scale in scales:
            def compute_detail_with_smooth(block, *, scale):
                # scale=1でも最小限のスムージングを適用
                smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            
            detail = gpu_arr.map_overlap(
                compute_detail_with_smooth,
                depth=common_depth,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                scale=scale
            )
            results.append(detail)
        
        # 重み付き合成とグローバル正規化
        def weighted_combine_and_normalize(*blocks):
            """グローバル統計量を使用して正規化"""
            nan_mask = cp.isnan(blocks[0])
            result = cp.zeros_like(blocks[0])
            
            for i, block in enumerate(blocks):
                valid = ~cp.isnan(block)
                result[valid] += block[valid] * weights[i]
            
            # グローバル統計量で正規化
            if norm_max > norm_min:
                result = (result - norm_min) / (norm_max - norm_min)
                result = cp.clip(result, 0, 1)
            else:
                result = cp.full_like(result, 0.5)
            
            # ガンマ補正
            result = cp.power(result, Constants.DEFAULT_GAMMA)
            
            # NaN位置を復元
            result[nan_mask] = cp.nan
            
            return result.astype(cp.float32)
        
        combined = da.map_blocks(
            weighted_combine_and_normalize,
            *results,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()

        return combined
    
    def get_default_params(self) -> dict:
        return {
            'scales': [1, 10, 50, 100],
            'weights': None,
            'downsample_factor': None,       # ダウンサンプル係数
            'verbose': False               # デバッグ出力
        }

###############################################################################
# 2.6. Frequency Enhancement (周波数強調) アルゴリズム
###############################################################################

def enhance_frequency_block(block: cp.ndarray, *, target_frequency: float = 0.1,
                          bandwidth: float = 0.05, enhancement: float = 2.0,
                          normalize: bool = True,  # 追加
                          norm_min: float = None,   # 追加
                          norm_max: float = None) -> cp.ndarray:  # 追加
    """特定周波数成分の強調"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # NaNを平均値で一時的に埋める
    if nan_mask.any():
        block_filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        block_filled = block
    
    # 窓関数を適用（境界での不連続性を軽減）
    window_y = cp.hanning(block.shape[0])[:, None]
    window_x = cp.hanning(block.shape[1])[None, :]
    window = window_y * window_x
    windowed_block = block_filled * window
    
    # 2D FFT
    fft = cp.fft.fft2(windowed_block)
    freq_x = cp.fft.fftfreq(block.shape[0])
    freq_y = cp.fft.fftfreq(block.shape[1])
    freq_grid = cp.sqrt(freq_x[:, None]**2 + freq_y[None, :]**2)
    
    # バンドパスフィルタ
    filter_mask = cp.exp(-((freq_grid - target_frequency)**2) / (2 * bandwidth**2))
    filter_mask = 1 + (enhancement - 1) * filter_mask
    
    # フィルタ適用
    filtered_fft = fft * filter_mask
    enhanced = cp.real(cp.fft.ifft2(filtered_fft))
    
    if normalize and norm_min is not None and norm_max is not None:
        # グローバル統計量で正規化
        if norm_max > norm_min:
            enhanced = (enhanced - norm_min) / (norm_max - norm_min)
        else:
            enhanced = cp.full_like(enhanced, 0.5)
        # 正規化後もガンマ補正を適用
        result = cp.power(enhanced, Constants.DEFAULT_GAMMA)
    else:
        # 正規化なしの場合はそのまま返す
        result = enhanced
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class FrequencyEnhancementAlgorithm(DaskAlgorithm):
    """周波数強調アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        # 統計量を計算
        stats = compute_global_stats(
            gpu_arr,
            freq_stat_func,
            enhance_frequency_block,
            {
                'target_frequency': params.get('target_frequency', 0.1),
                'bandwidth': params.get('bandwidth', 0.05),
                'enhancement': params.get('enhancement', 2.0),
                'normalize': False
            },
            downsample_factor=None,
            depth=32
        )

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        return gpu_arr.map_overlap(
            enhance_frequency_block,
            depth=32,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            target_frequency=params.get('target_frequency', 0.1),
            bandwidth=params.get('bandwidth', 0.05),
            enhancement=params.get('enhancement', 2.0),
            normalize=True,
            norm_min=stats[0],
            norm_max=stats[1]
        )
    
    def get_default_params(self) -> dict:
        return {
            'target_frequency': 0.1,  # 0-0.5の範囲
            'bandwidth': 0.05,
            'enhancement': 2.0
        }

###############################################################################
# 2.7. Curvature (曲率) アルゴリズム
###############################################################################

def compute_curvature_block(block: cp.ndarray, *, curvature_type: str = 'mean',
                          pixel_size: float = 1.0) -> cp.ndarray:
    """曲率計算（平均曲率、ガウス曲率、平面・断面曲率）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # NaNを隣接値で一時的に埋める
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    
    # 1次微分
    dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    
    # 2次微分
    dyy, dyx = cp.gradient(dy, pixel_size, edge_order=2)
    dxy, dxx = cp.gradient(dx, pixel_size, edge_order=2)
    
    if curvature_type == 'mean':
        # 平均曲率
        p = dx
        q = dy
        r = dxx
        s = (dxy + dyx) / 2  # 対称性のため平均を取る
        t = dyy
        
        denominator = cp.power(1 + p**2 + q**2, 1.5)
        numerator = (1 + q**2) * r - 2 * p * q * s + (1 + p**2) * t
        
        curvature = -numerator / (2 * denominator + 1e-10)
        
    elif curvature_type == 'gaussian':
        # ガウス曲率
        curvature = (dxx * dyy - dxy**2) / cp.power(1 + dx**2 + dy**2, 2)
        
    elif curvature_type == 'planform':
        # 平面曲率（等高線の曲率）
        curvature = -2 * (dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / \
                   (cp.power(dx**2 + dy**2, 1.5) + 1e-10)
                   
    else:  # profile
        # 断面曲率（最大傾斜方向の曲率）
        curvature = -2 * (dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) / \
                   ((dx**2 + dy**2) * cp.power(1 + dx**2 + dy**2, 0.5) + 1e-10)
    
    # 曲率の可視化（正負で色分け）
    # 正の曲率（凸）を明るく、負の曲率（凹）を暗く
    curvature_normalized = cp.tanh(curvature * 100)  # 感度調整
    result = (curvature_normalized + 1) / 2  # 0-1に正規化
    
    # ガンマ補正
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class CurvatureAlgorithm(DaskAlgorithm):
    """曲率アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        curvature_type = params.get('curvature_type', 'mean')
        pixel_size = params.get('pixel_size', 1.0)

        # 大規模データの場合、定期的にGCを実行（追加）
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()

        return gpu_arr.map_overlap(
            compute_curvature_block,
            depth=2,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            curvature_type=curvature_type,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'curvature_type': 'mean',  # 'mean', 'gaussian', 'planform', 'profile'
            'pixel_size': 1.0
        }

###############################################################################
# 2.15. Fractal Anomaly (フラクタル異常検出) アルゴリズム
###############################################################################

def compute_roughness_multiscale(block: cp.ndarray, radii: List[int]) -> cp.ndarray:
    """複数スケールでの局所的な標準偏差（roughness）を計算"""
    nan_mask = cp.isnan(block)
    sigmas = []
    
    for r in radii:
        if r <= 1:
            # 小さな半径の場合はガウシアンフィルタ
            sigma = 1.0
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
            # 分散計算のために二乗の平均も必要
            block_sq = block ** 2
            mean_sq, _ = handle_nan_with_gaussian(block_sq, sigma=sigma, mode='nearest')
        else:
            # 大きな半径の場合はuniform filter
            kernel_size = 2 * r + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='reflect')
            # 分散計算
            block_sq = block ** 2
            mean_sq, _ = handle_nan_with_uniform(block_sq, size=kernel_size, mode='reflect')
        
        # 標準偏差 = sqrt(E[X^2] - E[X]^2)
        variance = mean_sq - mean_elev ** 2
        sigma = cp.sqrt(cp.maximum(variance, 0.0))
        sigmas.append(sigma)
    
    # shape (H, W, n_scales)にスタック
    return cp.stack(sigmas, axis=-1)

def compute_fractal_dimension_block(block: cp.ndarray, *, 
                                  radii: List[int] = [2, 4, 8, 16, 32],
                                  normalize: bool = True,
                                  mu_global: float = None,
                                  sigma_global: float = None) -> cp.ndarray:
    """ブロックごとのフラクタル次元計算"""
    nan_mask = cp.isnan(block)
    
    # 複数スケールでroughnessを計算
    sigmas = compute_roughness_multiscale(block, radii)
    
    # log-log回帰の準備
    log_radii = cp.log(cp.asarray(radii, dtype=cp.float32))
    n_scales = len(radii)
    
    # 各ピクセルでlog-log回帰
    # log(sigma) = H * log(radius) + intercept の形で回帰
    log_sigmas = cp.log(sigmas + 1e-10)  # 数値安定性のため小さな値を追加
    
    # 回帰係数の計算（ベクトル化）
    mean_log_r = cp.mean(log_radii)
    mean_log_r2 = cp.mean(log_radii ** 2)
    
    # 各ピクセルでの平均
    mean_log_sigma = cp.mean(log_sigmas, axis=2)
    
    # 共分散の計算
    log_radii_broadcast = log_radii.reshape(1, 1, -1)
    mean_log_r_log_sigma = cp.mean(log_sigmas * log_radii_broadcast, axis=2)
    
    # 傾き（Hurst指数）
    denominator = mean_log_r2 - mean_log_r ** 2
    if cp.abs(denominator) > 1e-10:  # cp.absを使用
        H = (mean_log_r_log_sigma - mean_log_r * mean_log_sigma) / denominator
    else:
        H = cp.zeros_like(mean_log_sigma)
    
    # フラクタル次元（2D表面の場合）
    D = 3.0 - H
    
    if normalize and mu_global is not None and sigma_global is not None:
        # グローバル統計量でZ-score計算
        Z = (D - mu_global) / (sigma_global + 1e-10)
        
        # -3から3の範囲にクリップして-1から1に正規化
        Z = cp.clip(Z, -3.0, 3.0) / 3.0
        
        # ガンママッピング
        sign = cp.sign(Z)
        result = sign * (cp.abs(Z) ** Constants.DEFAULT_GAMMA)
    else:
        # 正規化なし（統計計算用）
        result = D
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

def fractal_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """フラクタル次元の統計量計算（平均と標準偏差）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # ロバスト統計（外れ値の影響を減らす）
        # 中央値周辺のデータのみ使用
        median = cp.median(valid_data)
        mad = cp.median(cp.abs(valid_data - median))  # Median Absolute Deviation
        
        # 3MAD以内のデータのみ使用
        mask = cp.abs(valid_data - median) < 3 * mad
        filtered_data = valid_data[mask]
        
        if len(filtered_data) > 0:
            # CuPy配列から明示的にPython floatに変換
            mean_val = cp.mean(filtered_data).item()
            std_val = cp.std(filtered_data).item()
            return (mean_val, std_val)
        else:
            # CuPy配列から明示的にPython floatに変換
            mean_val = cp.mean(valid_data).item()
            std_val = cp.std(valid_data).item()
            return (mean_val, std_val)
    return (2.0, 0.5)  # デフォルト値（フラクタル次元2.0が平坦な表面）

class FractalAnomalyAlgorithm(DaskAlgorithm):
    """フラクタル異常検出アルゴリズム
    
    地形のフラクタル次元を計算し、統計的に異常な領域を検出します。
    - 正の値（明るい）: フラクタル次元が高い = 異常に複雑な地形
    - 負の値（暗い）: フラクタル次元が低い = 異常に平滑な地形
    - 0付近（中間色）: 典型的な地形パターン
    """
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radii = params.get('radii', None)
        pixel_size = params.get('pixel_size', 1.0)
        
        # 半径の自動決定
        if radii is None:
            radii = self._determine_optimal_radii(pixel_size)
        
        max_radius = max(radii)
        depth = max_radius * 2 + 1
        
        # グローバル統計量を計算
        print("🔍 Computing global fractal statistics...")
        stats = compute_global_stats(
            gpu_arr,
            fractal_stat_func,
            compute_fractal_dimension_block,
            {'radii': radii, 'normalize': False},
            downsample_factor=params.get('downsample_factor', None),
            depth=depth,
            algorithm_name='fractal_anomaly'
        )
        
        mu_global, sigma_global = stats
        print(f"📊 Fractal dimension: μ={mu_global:.3f}, σ={sigma_global:.3f}")
        
        # 大規模データの場合、定期的にGCを実行
        if gpu_arr.nbytes > 10 * 1024**3:  # 10GB以上
            import gc
            gc.collect()
        
        # フルサイズで処理（正規化あり）
        return gpu_arr.map_overlap(
            compute_fractal_dimension_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radii=radii,
            normalize=True,
            mu_global=mu_global,
            sigma_global=sigma_global
        )
    
    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """解像度に基づいて最適な半径を決定"""
        resolution_class = classify_resolution(pixel_size)
        
        if resolution_class == 'ultra_high':
            # 0.5m以下
            base_radii = [2, 4, 8, 16, 32, 64]
        elif resolution_class == 'very_high':
            # 1m
            base_radii = [2, 4, 8, 16, 32]
        elif resolution_class == 'high':
            # 2.5m
            base_radii = [2, 4, 8, 16, 24]
        elif resolution_class == 'medium':
            # 5m
            base_radii = [2, 4, 8, 12]
        elif resolution_class == 'low':
            # 10-15m
            base_radii = [1, 2, 4, 8]
        else:
            # 30m以上
            base_radii = [1, 2, 3, 4]
        
        # メモリ制約を考慮して最大5つまでに制限
        if len(base_radii) > 5:
            # 対数的に分布するように選択
            indices = cp.linspace(0, len(base_radii)-1, 5).astype(int).get()
            base_radii = [base_radii[int(i)] for i in indices]
        
        return base_radii
    
    def get_default_params(self) -> dict:
        return {
            'radii': None,  # Noneの場合は自動決定
            'pixel_size': 1.0,
            'downsample_factor': None,  # 統計計算時のダウンサンプル係数
        }
###############################################################################
# 2.13. アルゴリズムレジストリ
###############################################################################

ALGORITHMS = {
    'rvi': RVIAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    'slope': SlopeAlgorithm(),
    'specular': SpecularAlgorithm(),
    'atmospheric_scattering': AtmosphericScatteringAlgorithm(),
    'multiscale_terrain': MultiscaleDaskAlgorithm(),
    'frequency_enhancement': FrequencyEnhancementAlgorithm(),
    'curvature': CurvatureAlgorithm(),
    'visual_saliency': VisualSaliencyAlgorithm(),
    'npr_edges': NPREdgesAlgorithm(),
    'atmospheric_perspective': AtmosphericPerspectiveAlgorithm(),
    'ambient_occlusion': AmbientOcclusionAlgorithm(),
    'tpi': TPIAlgorithm(),
    'lrm': LRMAlgorithm(),
    'openness': OpennessAlgorithm(),
    'fractal_anomaly': FractalAnomalyAlgorithm(),
}

# 新しいアルゴリズムの追加例:
# class AspectAlgorithm(DaskAlgorithm):
#     def process(self, gpu_arr: da.Array, **params) -> da.Array:
#         # 斜面方位の計算
#         pass
#     def get_default_params(self) -> dict:
#         return {'unit': 'degree', 'north_up': True}
# 
# ALGORITHMS['aspect'] = AspectAlgorithm()
