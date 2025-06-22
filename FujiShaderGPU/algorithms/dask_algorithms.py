"""
FujiShaderGPU/algorithms/dask_algorithms.py
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter, minimum_filter, convolve, binary_dilation
from ..config.gpu_config_manager import _gpu_config_manager

# ロギング設定
logger = logging.getLogger(__name__)

class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 150
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6
    EPSILON = 1e-8 # ゼロ除算防止用の小さな値

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
        return np.sqrt(max(1.0, pixel_size))
    
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
    
    dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    dy = dy * scale
    dx = dx * scale
    return dy, dx, nan_mask
    
def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """NaN位置を復元"""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result

###############################################################################
# int16変換用の共通関数
###############################################################################

def compute_gamma_corrected_range(data: cp.ndarray, apply_gamma: bool = True, 
                                 signed_range: bool = False) -> Tuple[float, float]:
    """ガンマ補正後の最小値・最大値を計算"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) == 0:
        return (-1.0, 1.0) if signed_range else (0.0, 1.0)
    
    if apply_gamma:
        if signed_range:
            # 符号を保持したガンマ補正
            sign = cp.sign(valid_data)
            gamma_corrected = sign * cp.power(cp.abs(valid_data), Constants.DEFAULT_GAMMA)
        else:
            gamma_corrected = cp.power(cp.clip(valid_data, 0, 1), Constants.DEFAULT_GAMMA)
        return (float(cp.min(gamma_corrected)), float(cp.max(gamma_corrected)))
    else:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))

def normalize_and_convert_to_int16(block: cp.ndarray, 
                                   global_min: float, 
                                   global_max: float,
                                   apply_gamma: bool = True,
                                   signed_input: bool = False) -> cp.ndarray:
    """グローバル統計に基づいて正規化し、int16に変換"""
    nan_mask = cp.isnan(block)
    
    # ガンマ補正
    if apply_gamma:
        if signed_input:
            # NaNやInfを考慮した安全な処理
            valid_mask = cp.isfinite(block)
            sign = cp.sign(block)
            abs_values = cp.abs(block)
            # 0や非常に小さい値でのpower演算エラーを防ぐ
            abs_values = cp.maximum(abs_values, Constants.EPSILON)
            processed = cp.where(valid_mask, 
                                 sign * cp.power(abs_values, Constants.DEFAULT_GAMMA),
                                 block)
        else:
            processed = cp.power(cp.clip(block, 0, 1), Constants.DEFAULT_GAMMA)
    else:
        processed = block
    
    # グローバル範囲で正規化してint16にマッピング
    if global_max > global_min:
        normalized = (processed - global_min) / (global_max - global_min)
        result = (normalized * 65534 - 32767).astype(cp.int16)
    else:
        result = cp.zeros_like(block, dtype=cp.int16)
    
    result[nan_mask] = -32768
    return result

###############################################################################
# グローバル統計ユーティリティ
###############################################################################
def determine_optimal_downsample_factor(
    data_shape: Tuple[int, int],
    algorithm_name: str = None,
    target_pixels: int = 500000,  # 目標ピクセル数（1000x1000）
    min_factor: int = 5,
    max_factor: int = 100) -> int:
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
    
    Returns:
    --------
    int : 最適なダウンサンプル係数
    """    
    # 現在のピクセル数
    current_pixels = data_shape[0] * data_shape[1]
    
    # 基本のダウンサンプル係数（平方根で計算）
    base_factor = float(cp.sqrt(current_pixels / target_pixels))
    
    # アルゴリズムの複雑度で調整
    complexity = _gpu_config_manager.get_algorithm_complexity(algorithm_name)
    adjusted_factor = base_factor * complexity
    
    # 整数化して範囲内に収める
    downsample_factor = int(max(min_factor, min(adjusted_factor, max_factor)))
    
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
def rvi_stat_func_with_gamma(data: cp.ndarray) -> Tuple[float, float, float]:
    """RVI用の統計量計算（ガンマ補正後の範囲も含む）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        std = float(cp.std(valid_data))
        # 正規化
        normalized = valid_data / (3 * std) if std > 0 else valid_data
        normalized = cp.clip(normalized, -1, 1)
        # ガンマ補正後の範囲
        min_gamma, max_gamma = compute_gamma_corrected_range(normalized, apply_gamma=True, signed_range=True)
        return (std, min_gamma, max_gamma)
    return (1.0, -1.0, 1.0)

def rvi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """RVI用の正規化"""
    std_global = stats[0]
    if std_global > 0:
        normalized = block / (3 * std_global)
        return cp.clip(normalized, -1, 1)
    return cp.zeros_like(block)

# FrequencyEnhancement用
def freq_stat_func_with_gamma(data: cp.ndarray) -> Tuple[float, float, float, float]:
    """周波数強調用の統計量計算（ガンマ補正後の範囲も含む）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        raw_min = float(cp.min(valid_data))
        raw_max = float(cp.max(valid_data))
        
        # 正規化とガンマ補正をシミュレート
        if raw_max > raw_min:
            normalized = (valid_data - raw_min) / (raw_max - raw_min)
            gamma_corrected = cp.power(normalized, Constants.DEFAULT_GAMMA)
            gamma_min = float(cp.min(gamma_corrected))
            gamma_max = float(cp.max(gamma_corrected))
        else:
            gamma_min, gamma_max = 0.5, 0.5
            
        return (raw_min, raw_max, gamma_min, gamma_max)
    return (0.0, 1.0, 0.0, 1.0)

def freq_norm_func(block: cp.ndarray, stats: Tuple[float, float], nan_mask: cp.ndarray) -> cp.ndarray:
    """周波数強調用の正規化"""
    min_val, max_val = stats
    if max_val > min_val:
        return (block - min_val) / (max_val - min_val)
    return cp.full_like(block, 0.5)

# TPI/LRM用
def tpi_lrm_stat_func_with_gamma(data: cp.ndarray) -> Tuple[float, float, float]:
    """TPI/LRM用の統計量計算（ガンマ補正後の範囲も含む）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        max_abs = float(cp.maximum(cp.abs(cp.min(valid_data)), 
                                   cp.abs(cp.max(valid_data))))
        
        # 正規化後の値でガンマ補正をシミュレート
        if max_abs > 0:
            normalized = valid_data / max_abs  # -1～1の範囲
            # 符号を保持したガンマ補正
            sign = cp.sign(normalized)
            gamma_corrected = sign * cp.power(cp.abs(normalized), Constants.DEFAULT_GAMMA)
            gamma_min = float(cp.min(gamma_corrected))
            gamma_max = float(cp.max(gamma_corrected))
        else:
            gamma_min, gamma_max = -1.0, 1.0
            
        return (max_abs, gamma_min, gamma_max)
    return (1.0, -1.0, 1.0)

def tpi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """TPI/LRM用の正規化"""
    max_abs = stats[0]
    if max_abs > 0:
        return cp.clip(block / max_abs, -1, 1)
    return cp.zeros_like(block)

# Visual Saliency用
# 統計関数を修正してガンマ補正後の範囲を返すように
def visual_saliency_stat_func_with_gamma(data: cp.ndarray) -> Tuple[float, float, float, float]:
    """Visual Saliency用の統計量計算（ガンマ補正後）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        low_p = float(cp.percentile(valid_data, 5))
        high_p = float(cp.percentile(valid_data, 95))
        
        if (high_p - low_p) < cp.std(valid_data) * 0.3:
            low_p = float(cp.percentile(valid_data, 2))
            high_p = float(cp.percentile(valid_data, 98))
        
        # テスト正規化とガンマ補正
        test_normalized = cp.linspace(0, 1, 100)
        test_gamma = cp.power(test_normalized, Constants.DEFAULT_GAMMA)
        gamma_min = float(cp.min(test_gamma))
        gamma_max = float(cp.max(test_gamma))
        
        return (low_p, high_p, gamma_min, gamma_max)
    return (0.0, 1.0, 0.0, 1.0)

def specular_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Specular用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # Specularは既にガンマ補正（0.7）が適用されている
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def atmospheric_scattering_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """大気散乱用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # すでにガンマ補正済みの値の範囲を取得
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def curvature_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """曲率用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        # すでにガンマ補正済みの値の範囲を取得
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def atmospheric_perspective_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """大気遠近法用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def ambient_occlusion_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """環境光遮蔽用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def openness_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """開度用の統計量計算（ガンマ補正後の範囲）"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

## 共通したHillshade計算の関数
def compute_hillshade_component(block: cp.ndarray, nan_mask: cp.ndarray, 
                               pixel_size: float = 1.0,
                               azimuth: float = Constants.DEFAULT_AZIMUTH,
                               altitude: float = Constants.DEFAULT_ALTITUDE) -> cp.ndarray:
    """Hillshadeコンポーネントの計算（0-1範囲）"""
    # NaNマスクを再利用
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    
    dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    aspect = cp.arctan2(-dy, dx)
    
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad)
    
    # 0-1に正規化
    return (hillshade + 1) / 2

def determine_optimal_radii(terrain_stats: dict) -> Tuple[List[int], List[float]]:
    """地形統計に基づいて最適な半径を決定"""
    pixel_size = terrain_stats.get('pixel_size', 1.0)
    mean_slope = terrain_stats['mean_slope']
    std_dev = terrain_stats['std_dev']
    
    # 地形の複雑さ
    complexity = mean_slope * std_dev
    
    # 基本的な実世界距離（メートル）
    if complexity < 0.1:
        # 平坦な地形：大きめのスケール
        base_distances = [10, 40, 160, 640]
    elif complexity < 0.3:
        # 緩やかな地形：中程度のスケール
        base_distances = [5, 20, 80, 320]
    else:
        # 複雑な地形：細かいスケール
        base_distances = [2.5, 10, 40, 160]
    
    # 最初からsetで管理してソート
    radii_set = set()
    for dist in base_distances:
        radius = int(dist / pixel_size)
        radius = max(2, min(radius, 256))
        radii_set.add(radius)
    radii = sorted(radii_set)
    
    # 最大4つまでに制限
    if len(radii) > 4:
        # 対数的に分布
        indices = np.logspace(0, np.log10(len(radii)-1), 4).astype(int)
        radii = [radii[int(i)] for i in indices]
    
    # 重みの決定（小さいスケールを重視）
    weights = []
    for i, r in enumerate(radii):
        weight = 1.0 / (i + 1)  # 1, 1/2, 1/3, 1/4
        weights.append(weight)
    
    # 正規化
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return radii, weights

def determine_optimal_sigmas(terrain_stats: dict) -> List[float]:
    """地形統計に基づいて最適なsigma値を決定"""
    sigmas_set = set()  # setを使って重複を確実に排除
    
    # 1. 標高レンジと勾配に基づく基本スケール
    elev_range = terrain_stats['elevation_range']
    mean_slope = terrain_stats['mean_slope']
    
    # 地形の複雑さの指標
    terrain_complexity = mean_slope * terrain_stats['std_dev'] / (elev_range + 1e-6)
    
    # 基本スケール（地形の複雑さに応じて調整）
    if terrain_complexity < 0.1:  # 平坦な地形
        base_scales = [50, 100, 150]  # より小さい値に制限
    elif terrain_complexity < 0.3:  # 緩やかな地形
        base_scales = [50, 100, 200]
    else:  # 複雑な地形
        base_scales = [25, 50, 100, 200]
    
    # 2. FFT解析から得られたスケールを追加
    if terrain_stats.get('dominant_scales', []):
        for scale in terrain_stats['dominant_scales']:
            if 10 < scale < 500:  # 現実的な範囲のスケールのみ
                # Gaussianフィルタのsigmaは、検出されたスケールの約1/4
                sigma_candidate = round(scale / 4, 0)  # 整数に丸める
                if 5 <= sigma_candidate <= 150:  # 最大値を150に制限
                    sigmas_set.add(sigma_candidate)
    
    # 3. 曲率に基づく微細スケール
    mean_curv = terrain_stats['mean_curvature']
    if mean_curv > 0.01:  # 曲率が高い場合は細かいスケールも追加
        sigmas_set.add(10)
    
    # 基本スケールを追加
    for scale in base_scales:
        if 5 <= scale <= 150:  # 最大値を150に制限
            sigmas_set.add(scale)
    
    # setをリストに変換してソート
    sigmas = sorted(list(sigmas_set))
    
    # 最大3つまでに制限（メモリ効率のため、5→3に削減）
    if len(sigmas) > 3:
        indices = cp.linspace(0, len(sigmas)-1, 3).astype(int)
        sigmas = [sigmas[i] for i in indices]
    
    return [float(s) for s in sigmas]

def compute_tpi_lrm_no_gamma(block: cp.ndarray, *, radius: int, max_abs: float) -> cp.ndarray:
    """TPIを計算（ガンマ補正なし）"""
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
    if max_abs > 0:
        tpi = tpi / max_abs
    
    # -1～1にクリップ（ガンマ補正はしない）
    result = cp.clip(tpi, -1, 1)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

###############################################################################
# 2.1. RVI (Ridge-Valley Index) アルゴリズム
###############################################################################

def compute_rvi_efficient_block(block: cp.ndarray, *, 
                               radii: List[int] = [4, 16, 64], 
                               weights: Optional[List[float]] = None) -> cp.ndarray:
    """効率的なRVI計算"""
    if not radii:
        raise ValueError("radii list cannot be empty")
    if any(r <= 0 for r in radii):
        raise ValueError("All radii must be positive integers")
    
    nan_mask = cp.isnan(block)
    
    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    elif not isinstance(weights, cp.ndarray):
        weights = cp.array(weights, dtype=cp.float32)
    if len(weights) != len(radii):
        raise ValueError(f"Length of weights ({len(weights)}) must match length of radii ({len(radii)})")
    
    # 結果をインプレースで累積（メモリ効率向上）
    rvi_combined = None
    
    for i, (radius, weight) in enumerate(zip(radii, weights)):
        if radius <= 1:
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=1.0, mode='nearest')
        else:
            kernel_size = 2 * radius + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='reflect')
        
        # 差分を計算
        diff = weight * (block - mean_elev)
        
        if rvi_combined is None:
            rvi_combined = diff
        else:
            rvi_combined += diff
        
        # メモリを即座に解放
        del mean_elev, diff
    
    # NaN処理
    rvi_combined = restore_nan(rvi_combined, nan_mask)
    
    return rvi_combined


def multiscale_rvi(gpu_arr: da.Array, *, 
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
        terrain_stats = params.get('terrain_stats', None)
        if radii is None:
            radii, weights = determine_optimal_radii(terrain_stats)
        max_radius = max(radii)
        rvi = multiscale_rvi(gpu_arr, radii=radii, weights=weights)

        stats = compute_global_stats(
            rvi,
            rvi_stat_func_with_gamma,
            compute_rvi_efficient_block,
            {'radii': radii, 'weights': weights},
            params.get('downsample_factor', None),
            depth=max_radius * 2 + 1,
            algorithm_name='rvi'
        )

        # 正規化とint16変換を一度に
        std_global, gamma_min, gamma_max = stats
        return rvi.map_blocks(
            lambda block: normalize_and_convert_to_int16(
                apply_global_normalization(block, rvi_norm_func, (std_global,)),
                gamma_min, gamma_max, apply_gamma=True, signed_input=True
            ),
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
        )
    
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
# 2.2. Hillshade アルゴリズム
###############################################################################

def compute_hillshade_block(block: cp.ndarray, *, azimuth: float = Constants.DEFAULT_AZIMUTH, 
                           altitude: float = Constants.DEFAULT_ALTITUDE, z_factor: float = 1.0,
                           pixel_size: float = 1.0) -> cp.ndarray:
    """1ブロックに対するHillshade計算（0-1の範囲で返す）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # z_factorを適用
    scaled_block = block * z_factor

    # compute_hillshade_componentを再利用
    hillshade = compute_hillshade_component(scaled_block, nan_mask, pixel_size, azimuth, altitude)
    
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
        sigmas = params.get('sigmas', [1])
        agg = params.get('agg', 'mean')
        
        # Hillshadeは常に0-1の範囲
        # ガンマ補正後の範囲を事前計算
        test_values = cp.linspace(0, 1, 100)
        gamma_corrected = cp.power(test_values, Constants.DEFAULT_GAMMA)
        gamma_min = 0.0  # ガンマ補正後も0が最小
        gamma_max = float(cp.max(gamma_corrected))
        
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
                
                # Hillshade計算とint16変換を一度に
                hs = smoothed.map_overlap(
                    lambda block: normalize_and_convert_to_int16(
                        compute_hillshade_block(
                            block,
                            azimuth=azimuth,
                            altitude=altitude,
                            z_factor=z_factor,
                            pixel_size=pixel_size
                        ),
                        gamma_min,
                        gamma_max,
                        apply_gamma=True  # ガンマ補正を適用
                    ),
                    depth=1,
                    boundary='reflect',
                    dtype=cp.int16,
                    meta=cp.empty((0, 0), dtype=cp.int16)
                )
                results.append(hs)
            
            # 集約（int16のまま処理）
            stacked = da.stack(results, axis=0)
            if agg == "stack":
                return stacked
            elif agg == "mean":
                # int16の平均を計算
                # 一時的にfloat32に変換して平均を取り、結果をint16に戻す
                mean_float = da.mean(stacked.astype(cp.float32), axis=0)
                return mean_float.map_blocks(
                    lambda b: cp.clip(b, -32767, 32767).astype(cp.int16),
                    dtype=cp.int16,
                    meta=cp.empty((0, 0), dtype=cp.int16)
                )
            elif agg == "min":
                return da.min(stacked, axis=0)
            elif agg == "max":
                return da.max(stacked, axis=0)
            else:
                # デフォルトは平均
                mean_float = da.mean(stacked.astype(cp.float32), axis=0)
                return mean_float.map_blocks(
                    lambda b: cp.clip(b, -32767, 32767).astype(cp.int16),
                    dtype=cp.int16,
                    meta=cp.empty((0, 0), dtype=cp.int16)
                )
        else:
            # 単一スケールHillshade
            return gpu_arr.map_overlap(
                lambda block: normalize_and_convert_to_int16(
                    compute_hillshade_block(
                        block,
                        azimuth=azimuth,
                        altitude=altitude,
                        z_factor=z_factor,
                        pixel_size=pixel_size
                    ),
                    gamma_min,
                    gamma_max,
                    apply_gamma=True  # ガンマ補正を適用
                ),
                depth=1,
                boundary='reflect',
                dtype=cp.int16,
                meta=cp.empty((0, 0), dtype=cp.int16)
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
    
    # マルチスケール処理（メモリ効率改善版）
    combined_saliency = cp.zeros_like(block)
    
    for i, scale in enumerate(scales):
        # 局所的なコントラスト
        center_sigma = scale
        surround_sigma = scale * 2
        
        if nan_mask.any():
            center, _ = handle_nan_with_gaussian(block, sigma=center_sigma, mode='nearest')
            surround, _ = handle_nan_with_gaussian(block, sigma=surround_sigma, mode='nearest')
        else:
            center = gaussian_filter(block, sigma=center_sigma, mode='nearest')
            surround = gaussian_filter(block, sigma=surround_sigma, mode='nearest')
        
        # 差分の絶対値
        contrast = cp.abs(center - surround) / (cp.log(scale + cp.e))
        
        # 中間変数を早期解放
        del center
        del surround
        
        # 勾配の強度
        if scale > 1:
            # sigmaを対数的にスケール（線形ではなく）
            sigma_for_gradient = cp.log(scale + 1)
            if nan_mask.any():
                filled_grad = cp.where(nan_mask, 0, gradient_mag_base)
                valid = (~nan_mask).astype(cp.float32)
                
                smoothed_values = gaussian_filter(filled_grad * valid, sigma=sigma_for_gradient, mode='nearest')
                smoothed_weights = gaussian_filter(valid, sigma=sigma_for_gradient, mode='nearest')
                gradient_mag = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)
            else:
                gradient_mag = gaussian_filter(gradient_mag_base, sigma=sigma_for_gradient, mode='nearest')
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
        
        # 各スケールの値の範囲を揃える（より控えめな正規化）
        # コントラストは log(scale + e) で除算済みなので、追加の正規化は軽めに
        if scale > 1:
            scale_normalization = 1.0 + cp.log(scale) * 0.2  # より控えめな係数
            feature = feature / scale_normalization
        
        # 重み付けして累積
        scale_weight = 1.0 / len(scales)
        combined_saliency += feature * scale_weight
        
        # 不要になった変数を削除
        del feature
    
    # 勾配ベースも解放
    del gradient_mag_base
    
    # 正規化処理を条件分岐に変更
    if normalize:
        if norm_min is not None and norm_max is not None:
            # 提供された統計量で正規化
            if norm_max > norm_min:
                # より適切な正規化範囲を使用（パーセンタイルベース）
                range_val = norm_max - norm_min
                # 外れ値の影響を軽減するためのソフトクリッピング
                result = cp.tanh((combined_saliency - norm_min) / (range_val * 0.5)) * 0.5 + 0.5
                result = cp.clip(result, 0, 1)
            else:
                result = cp.full_like(block, 0.5)
        else:
            # 従来のローカル正規化（互換性のため残す）
            valid_result = combined_saliency[~nan_mask] if nan_mask.any() else combined_saliency.ravel()
            if len(valid_result) > 0:
                # パーセンタイルベースの正規化で外れ値の影響を軽減
                min_val = float(cp.percentile(valid_result, 2))
                max_val = float(cp.percentile(valid_result, 98))
                if max_val > min_val:
                    result = (combined_saliency - min_val) / (max_val - min_val)
                    result = cp.clip(result, 0, 1)
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
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [2, 4, 8, 16])
        max_scale = max(scales)

        # 統計量を計算（新しい共通関数を使用）
        stats = compute_global_stats(
            gpu_arr,
            visual_saliency_stat_func_with_gamma,
            compute_visual_saliency_block,
            {
                'scales': scales,
                'pixel_size': params.get('pixel_size', 1.0),
                'normalize': False
            },
            downsample_factor=params.get('downsample_factor', None),
            depth=int(max_scale * 8),
            algorithm_name='visual_saliency'
        )
        
        # 修正: 正しい統計値の取り出し
        raw_min, raw_max, gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(  # 修正: 正しい関数名
                compute_visual_saliency_block(
                    block,
                    scales=scales,
                    pixel_size=params.get('pixel_size', 1.0),
                    normalize=True,
                    norm_min=raw_min,   # 修正: raw_min
                    norm_max=raw_max    # 修正: raw_max
                ),
                gamma_min,  # ガンマ補正後の最小値
                gamma_max,  # ガンマ補正後の最大値
                apply_gamma=False  # compute_visual_saliency_block内で既にガンマ補正済み
            ),
            depth=int(max_scale * 8),
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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
        
        # NPREdgesは常に0.2-1.0の範囲で出力される
        # ガンマ補正後の範囲を事前計算
        test_values = cp.linspace(0.2, 1.0, 100)
        gamma_corrected = cp.power(test_values, Constants.DEFAULT_GAMMA)
        gamma_min = float(cp.min(gamma_corrected))
        gamma_max = float(cp.max(gamma_corrected))
        
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_npr_edges_block(
                    block,
                    edge_sigma=edge_sigma,
                    threshold_low=threshold_low,
                    threshold_high=threshold_high,
                    pixel_size=pixel_size
                ),
                gamma_min,  # ガンマ補正後の最小値
                gamma_max,  # ガンマ補正後の最大値
                apply_gamma=False  # compute_npr_edges_block内で既にガンマ補正済み
            ),
            depth=depth,
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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
    
    hillshade = compute_hillshade_component(block, nan_mask, pixel_size)
    
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
        
        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            atmospheric_perspective_stat_func,
            compute_atmospheric_perspective_block,
            {
                'depth_scale': depth_scale,
                'haze_strength': haze_strength,
                'pixel_size': pixel_size
            },
            downsample_factor=None,
            depth=1,
            algorithm_name='atmospheric_perspective'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_atmospheric_perspective_block(
                    block,
                    depth_scale=depth_scale,
                    haze_strength=haze_strength,
                    pixel_size=pixel_size
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # すでにガンマ補正済み
            ),
            depth=1,
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
        )
    
    def get_default_params(self) -> dict:
        return {
            'depth_scale': 1000.0,
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
    
    # サンプリング方向を事前計算（修正: CuPy配列として作成）
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
        
        # GPU側で変位を計算（修正: すべてCuPyで処理）
        dx_all = cp.round(r * directions[:, 0]).astype(cp.int32)
        dy_all = cp.round(r * directions[:, 1]).astype(cp.int32)
        
        for i in range(num_samples):
            dx = int(dx_all[i])  # CuPyスカラーをPython intに変換
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
            # 平地での処理を追加 高さの差が非常に小さい場合は遮蔽なしとする
            small_diff_mask = cp.abs(height_diff) < (pixel_size * 0.01)  # 1%未満の高さ変化
            # 実際の距離（メートル）を使用
            distance = r * pixel_size
            # ゼロ除算とオーバーフロー対策
            occlusion_angle = cp.arctan(height_diff / (distance + Constants.EPSILON))
            occlusion_angle = cp.where(small_diff_mask, 0, occlusion_angle)
            
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
        
        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            ambient_occlusion_stat_func,
            compute_ambient_occlusion_block,
            {
                'num_samples': num_samples,
                'radius': radius,
                'intensity': intensity,
                'pixel_size': pixel_size
            },
            downsample_factor=None,
            depth=int(radius + 1),
            algorithm_name='ambient_occlusion'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_ambient_occlusion_block(
                    block,
                    num_samples=num_samples,
                    radius=radius,
                    intensity=intensity,
                    pixel_size=pixel_size
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # すでにガンマ補正済み
            ),
            depth=int(radius + 1),
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
        )
    
    def get_default_params(self) -> dict:
        return {
            'num_samples': 16,
            'radius': 10.0,
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
        
        # 統計量を計算（修正版を使用）
        stats = compute_global_stats(
            gpu_arr,
            tpi_lrm_stat_func_with_gamma,  # 修正版を使用
            compute_tpi_block,
            {'radius': radius, 'std_global': None},
            downsample_factor=None,
            depth=radius+1,
            algorithm_name='tpi'
        )
        
        # フルサイズで処理し、int16に変換
        max_abs, gamma_min, gamma_max = stats
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_tpi_lrm_no_gamma(block, radius=radius, max_abs=max_abs),
                gamma_min,  # ガンマ補正後の最小値
                gamma_max,  # ガンマ補正後の最大値
                apply_gamma=True,  # 符号を保持したガンマ補正を適用
                signed_input=True  # 入力が符号付きであることを指定
            ),
            depth=radius+1,
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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
            # より穏やかな正規化（3→6に変更）
            lrm = lrm / (6 * std_global)
        else:
            # std_globalが小さすぎる場合の処理
            lrm = lrm / cp.std(lrm[~nan_mask])
    
    # 結果は-1から+1の範囲
    result = cp.clip(lrm, -1, 1)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class LRMAlgorithm(DaskAlgorithm):
    """局所起伏モデルアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        kernel_size = params.get('kernel_size', 25)
        pixel_size = params.get('pixel_size', 1.0)
        
        # 統計量を計算（TPIと同じ関数を使用）
        stats = compute_global_stats(
            gpu_arr,
            tpi_lrm_stat_func_with_gamma,  # 修正: 正しい関数名
            compute_lrm_block,
            {'kernel_size': kernel_size, 'pixel_size': pixel_size},  # pixel_sizeを追加
            downsample_factor=None,
            depth=int(kernel_size * 2),
            algorithm_name='lrm'
        )
        
        # フルサイズで処理し、int16に変換
        max_abs, gamma_min, gamma_max = stats
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_lrm_block(block, kernel_size=kernel_size, std_global=max_abs),
                gamma_min,  # ガンマ補正後の最小値
                gamma_max,  # ガンマ補正後の最大値
                apply_gamma=True,  # 符号を保持したガンマ補正を適用
                signed_input=True  # 入力が符号付きであることを指定
            ),
            depth=int(kernel_size * 2),
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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
    # ブロックの形状チェック（修正追加）
    if block.ndim != 2:
        # 1次元の場合は適切な2次元形状に変換
        if block.ndim == 1:
            # 正方形に近い形状に変換を試みる
            size = block.size
            h = int(cp.sqrt(size))
            w = size // h
            if h * w != size:
                # 完全に割り切れない場合は、最後の要素を無視
                block = block[:h*w].reshape(h, w)
            else:
                block = block.reshape(h, w)
        else:
            raise ValueError(f"Expected 2D block, got {block.ndim}D")
    
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # 形状が小さすぎる場合の処理（修正追加）
    if h < 3 or w < 3:
        # 小さすぎるブロックは処理できないので、適切なデフォルト値を返す
        if openness_type == 'positive':
            return cp.ones_like(block) * 0.5  # 中間値
        else:
            return cp.ones_like(block) * 0.5
    
    # 以下、既存のコード...
    # 方向ベクトルの事前計算
    angles = cp.linspace(0, 2 * cp.pi, num_directions, endpoint=False)
    
    # より効率的な距離サンプリング
    num_samples = min(5, int(cp.log2(max_distance)) + 1)
    distances = cp.unique(cp.logspace(0, cp.log10(max_distance), num_samples, dtype=cp.float32)).astype(cp.int32)
    distances = distances[distances > 0]
    
    # 初期化
    init_val = -cp.pi/2 if openness_type == 'positive' else cp.pi/2
    max_angles = cp.full((h, w), init_val, dtype=cp.float32)
    
    # バッチサイズの決定（メモリ使用量を考慮）
    available_memory = cp.cuda.runtime.memGetInfo()[0] / (1024**3)  # GB
    if available_memory > 10:
        batch_size = 8
    elif available_memory > 5:
        batch_size = 4
    else:
        batch_size = 2
    
    # 方向をバッチ処理
    for dir_start in range(0, num_directions, batch_size):
        dir_end = min(dir_start + batch_size, num_directions)
        batch_angles = angles[dir_start:dir_end]
        
        cos_vals = cp.cos(batch_angles)[:, cp.newaxis]
        sin_vals = cp.sin(batch_angles)[:, cp.newaxis]
        
        for dist_idx, r in enumerate(distances):
            offset_x = cp.round(r * cos_vals).astype(cp.int32).ravel()
            offset_y = cp.round(r * sin_vals).astype(cp.int32).ravel()
            
            valid_offsets = (offset_x != 0) | (offset_y != 0)
            if not valid_offsets.any():
                continue
            
            offset_x = offset_x[valid_offsets]
            offset_y = offset_y[valid_offsets]
            
            # バッチ内の全方向を処理（修正版）
            for i, (ox, oy) in enumerate(zip(offset_x, offset_y)):
                # 各ピクセルごとに計算（修正版）
                y_indices = cp.arange(h)[:, cp.newaxis]
                x_indices = cp.arange(w)[cp.newaxis, :]
                
                # シフト後のインデックス
                shifted_y = y_indices + oy
                shifted_x = x_indices + ox
                
                # 境界チェック
                valid_mask = (shifted_y >= 0) & (shifted_y < h) & (shifted_x >= 0) & (shifted_x < w)
                
                # 有効な位置の値を取得（修正版）
                shifted_values = cp.full((h, w), cp.nan, dtype=cp.float32)
                
                # valid_maskがTrueの位置を取得
                valid_coords = cp.where(valid_mask)
                if len(valid_coords[0]) > 0:
                    # 対応するシフト後の座標を取得
                    src_y = shifted_y[valid_coords]
                    src_x = shifted_x[valid_coords]
                    # blockから値を取得して設定
                    shifted_values[valid_coords] = block[src_y, src_x]
                
                # 角度計算
                angle = cp.arctan((shifted_values - block) / (r * pixel_size))
                
                # NaNでない有効な位置のみ更新
                valid = ~(cp.isnan(angle) | nan_mask)
                
                if openness_type == 'positive':
                    max_angles = cp.where(valid, cp.maximum(max_angles, angle), max_angles)
                else:
                    max_angles = cp.where(valid, cp.minimum(max_angles, angle), max_angles)
        
        # メモリ解放（バッチごと）
        del cos_vals, sin_vals
        cp.cuda.Stream.null.synchronize()
    
    # 開度の計算と正規化
    if openness_type == 'positive':
        openness = cp.pi/2 - max_angles
    else:
        openness = cp.pi/2 + max_angles
    
    # 0-1に正規化
    openness = cp.clip(openness / (cp.pi/2), 0, 1)
    
    # ガンマ補正
    result = cp.power(openness, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    # メモリ解放
    del max_angles
    
    return result.astype(cp.float32)

class OpennessAlgorithm(DaskAlgorithm):
    """開度アルゴリズム（簡易高速版）"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        openness_type = params.get('openness_type', 'positive')
        num_directions = params.get('num_directions', 16)
        pixel_size = params.get('pixel_size', 1.0)
        
        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            openness_stat_func,
            compute_openness_vectorized,
            {
                'openness_type': openness_type,
                'num_directions': num_directions,
                'max_distance': max_distance,
                'pixel_size': pixel_size
            },
            downsample_factor=None,
            depth=max_distance + 1,
            algorithm_name='openness'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_openness_vectorized(
                    block,
                    openness_type=openness_type,
                    num_directions=num_directions,
                    max_distance=max_distance,
                    pixel_size=pixel_size
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # すでにガンマ補正済み
            ),
            depth=max_distance + 1,
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
        )
    
    def get_default_params(self) -> dict:
        return {
            'openness_type': 'positive',
            'num_directions': 16,
            'max_distance': 50,
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
        
        # 単位に応じた自明な範囲
        if unit == 'degree':
            expected_range = (0.0, 90.0)
        elif unit == 'percent':
            expected_range = (0.0, 1000.0)
        else:  # radians
            expected_range = (0.0, cp.pi/2)
        
        # 正規化関数を定義
        def normalize_slope(block):
            if unit == 'degree':
                normalized = 1.0 - (block / 90.0) # 反転: 急斜面を暗く、平地を明るく
            elif unit == 'percent':
                normalized = 1.0 - cp.tanh(block / 100.0)
            else:
                normalized = 1.0 - (block / (cp.pi / 2))
            return cp.clip(normalized, 0, 1)
        
        # ガンマ補正後の範囲を計算
        test_values = cp.array([0.0, 0.5, 1.0])
        gamma_corrected = cp.power(test_values, Constants.DEFAULT_GAMMA)
        global_min = 0.0
        global_max = float(gamma_corrected[-1])
        
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                normalize_slope(compute_slope_block(block, unit=unit, pixel_size=pixel_size)),
                global_min, global_max, apply_gamma=True, signed_input=False
            ),
            depth=1,
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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

        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            specular_stat_func,
            compute_specular_block,
            {
                'roughness_scale': roughness_scale,
                'shininess': shininess,
                'pixel_size': pixel_size,
                'light_azimuth': light_azimuth,
                'light_altitude': light_altitude
            },
            downsample_factor=None,
            depth=int(roughness_scale),
            algorithm_name='specular'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_specular_block(
                    block,
                    roughness_scale=roughness_scale,
                    shininess=shininess,
                    pixel_size=pixel_size,
                    light_azimuth=light_azimuth,
                    light_altitude=light_altitude
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # compute_specular_block内で既にガンマ補正（0.7）済み
            ),
            depth=int(roughness_scale),
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
        )
    
    def get_default_params(self) -> dict:
        return {
            'roughness_scale': 20.0,  # デフォルト値も調整
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
                                       pixel_size: float = 1.0) -> cp.ndarray:
    """大気散乱によるシェーディング（Rayleigh散乱の簡略版）"""        
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 法線計算
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    slope = cp.sqrt(dx**2 + dy**2)
    
    # 天頂角（法線と垂直方向のなす角）
    zenith_angle = cp.arctan(slope)
    
    # 大気の厚さ（簡略化：天頂角に比例）
    air_mass = 1.0 / (cp.cos(zenith_angle) + Constants.EPSILON)  # ゼロ除算回避
    
    # Rayleigh散乱の近似
    scattering = 1.0 - cp.exp(-scattering_strength * air_mass)
    
    # 青みがかった散乱光を表現（単一チャンネルなので明度のみ）
    ambient = 0.4 + 0.6 * scattering    
    hillshade = compute_hillshade_component(block, nan_mask, pixel_size)
    
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
        pixel_size = params.get('pixel_size', 1.0)
        
        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            atmospheric_scattering_stat_func,
            compute_atmospheric_scattering_block,
            {
                'scattering_strength': scattering_strength,
                'pixel_size': pixel_size
            },
            downsample_factor=None,
            depth=1,
            algorithm_name='atmospheric_scattering'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_atmospheric_scattering_block(
                    block,
                    scattering_strength=scattering_strength,
                    pixel_size=pixel_size
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # compute_atmospheric_scattering_block内で既にガンマ補正済み
            ),
            depth=1,
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
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
            downsample_factor = determine_optimal_downsample_factor(
                gpu_arr.shape,
                algorithm_name='multiscale_terrain'
            )
            
        if weights is None:
            weights = [1.0 / s for s in scales]
        
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)
        
        # Step 1: 縮小版で統計量を計算
        downsampled = gpu_arr[::downsample_factor, ::downsample_factor]
        
        results_small = []
        max_scale_small = max([max(1, s // downsample_factor) for s in scales])
        common_depth_small = min(int(4 * max_scale_small), Constants.MAX_DEPTH)

        def create_weighted_combiner(weights):
            def weighted_combine_for_stats(*blocks):
                result = cp.zeros_like(blocks[0])
                
                for i, block in enumerate(blocks):
                    valid = ~cp.isnan(block)
                    result[valid] += block[valid] * weights[i]
                
                all_nan = cp.ones_like(blocks[0], dtype=bool)
                for block in blocks:
                    all_nan &= cp.isnan(block)
                result[all_nan] = cp.nan
                return result
            return weighted_combine_for_stats
        
        for i, scale in enumerate(scales):
            scale_small = max(1, scale // downsample_factor)
            
            def compute_detail_small(block, *, scale):
                smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            
            detail_small = downsampled.map_overlap(
                compute_detail_small,
                depth=common_depth_small,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                scale=scale_small
            )
            results_small.append(detail_small)
            
        combined_small = da.map_blocks(
            create_weighted_combiner(weights),
            *results_small,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        ).compute()

        # 統計量を計算（ガンマ補正後の範囲も含む）
        valid_data = combined_small[~cp.isnan(combined_small)]
        if len(valid_data) > 0:
            norm_min = float(cp.percentile(valid_data, 5))
            norm_max = float(cp.percentile(valid_data, 95))
            
            # ガンマ補正後の範囲をシミュレート
            if norm_max > norm_min:
                test_normalized = (valid_data - norm_min) / (norm_max - norm_min)
                test_normalized = cp.clip(test_normalized, 0, 1)
                gamma_corrected = cp.power(test_normalized, Constants.DEFAULT_GAMMA)
                gamma_min = float(cp.min(gamma_corrected))
                gamma_max = float(cp.max(gamma_corrected))
            else:
                gamma_min, gamma_max = 0.5, 0.5
        else:
            norm_min, norm_max = 0.0, 1.0
            gamma_min, gamma_max = 0.0, 1.0
        
        if params.get('verbose', False):
            logger.debug(
                "Multiscale Terrain global stats: min=%.3f, max=%.3f | "
                "After γ-correction: min=%.3f, max=%.3f",
                norm_min, norm_max, gamma_min, gamma_max,
            )
        
        # Step 2: フルサイズで処理
        results = []
        for scale in scales:
            def compute_detail_with_smooth(block, *, scale):
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
        
        # 重み付き合成（ガンマ補正なし）
        def weighted_combine_no_gamma(*blocks):
            """グローバル統計量を使用して正規化（ガンマ補正なし）"""
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
            
            # ガンマ補正は削除（normalize_and_convert_to_int16で行う）
            
            # NaN位置を復元
            result[nan_mask] = cp.nan
            
            return result.astype(cp.float32)
        
        combined = da.map_blocks(
            weighted_combine_no_gamma,
            *results,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        
        # int16に変換（ガンマ補正を含む）
        return combined.map_blocks(
            lambda block: normalize_and_convert_to_int16(
                block,
                0.0,  # 既に正規化済みなので0-1
                1.0,
                apply_gamma=True,  # ここでガンマ補正を適用
                signed_input=False
            ),
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
        )
    
    def get_default_params(self) -> dict:
        return {
            'scales': [1, 10, 50, 100],
            'weights': None,
            'downsample_factor': None,
            'verbose': False
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
        # 統計量を計算（修正版の統計関数を使用）
        stats = compute_global_stats(
            gpu_arr,
            freq_stat_func_with_gamma,  # 修正版を使用
            enhance_frequency_block,
            {
                'target_frequency': params.get('target_frequency', 0.1),
                'bandwidth': params.get('bandwidth', 0.05),
                'enhancement': params.get('enhancement', 2.0),
                'normalize': False  # 統計計算時は正規化なし
            },
            downsample_factor=params.get('downsample_factor', None),
            depth=32,
            algorithm_name='frequency_enhancement'
        )
        
        raw_min, raw_max, gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                enhance_frequency_block(
                    block,
                    target_frequency=params.get('target_frequency', 0.1),
                    bandwidth=params.get('bandwidth', 0.05),
                    enhancement=params.get('enhancement', 2.0),
                    normalize=True,
                    norm_min=raw_min,  # 生の最小値
                    norm_max=raw_max   # 生の最大値
                ),
                gamma_min,  # ガンマ補正後の最小値
                gamma_max,  # ガンマ補正後の最大値
                apply_gamma=False  # enhance_frequency_block内で既にガンマ補正済み
            ),
            depth=32,
            boundary='reflect',
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
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
        
        curvature = -numerator / (2 * denominator + Constants.EPSILON)
        
    elif curvature_type == 'gaussian':
        # ガウス曲率
        curvature = (dxx * dyy - dxy**2) / cp.power(1 + dx**2 + dy**2, 2)
        
    elif curvature_type == 'planform':
        # 平面曲率（等高線の曲率）
        curvature = -2 * (dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / \
                   (cp.power(dx**2 + dy**2, 1.5) + Constants.EPSILON)
                   
    else:  # profile
        # 断面曲率（最大傾斜方向の曲率）
        curvature = -2 * (dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) / \
                   ((dx**2 + dy**2) * cp.power(1 + dx**2 + dy**2, 0.5) + Constants.EPSILON)
    
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
        
        # グローバル統計を計算（新規追加）
        stats = compute_global_stats(
            gpu_arr,
            curvature_stat_func,
            compute_curvature_block,
            {
                'curvature_type': curvature_type,
                'pixel_size': pixel_size
            },
            downsample_factor=None,
            depth=2,
            algorithm_name='curvature'
        )
        
        gamma_min, gamma_max = stats
        
        # フルサイズで処理し、int16に変換
        return gpu_arr.map_overlap(
            lambda block: normalize_and_convert_to_int16(
                compute_curvature_block(
                    block,
                    curvature_type=curvature_type,
                    pixel_size=pixel_size
                ),
                gamma_min,
                gamma_max,
                apply_gamma=False  # compute_curvature_block内で既にガンマ補正済み
            ),
            depth=2,
            boundary='reflect',
            dtype=cp.int16,  # 変更
            meta=cp.empty((0, 0), dtype=cp.int16)  # 変更
        )
    
    def get_default_params(self) -> dict:
        return {
            'curvature_type': 'mean',  # 'mean', 'gaussian', 'planform', 'profile'
            'pixel_size': 1.0
        }

###############################################################################
# 2.15. Fractal Anomaly (フラクタル異常検出) アルゴリズム
###############################################################################

def compute_roughness_multiscale(block: cp.ndarray, radii: List[int], window_mult: int = 3) -> cp.ndarray:
    """参考実装に準拠した複数スケールでの局所的な標準偏差（roughness）を計算"""
    nan_mask = cp.isnan(block)
    sigmas = []
    
    for r in radii:
        # 参考実装と同じwindow_size計算
        window_size = r * window_mult
        
        # NaNを0で埋める（uniform_filterのため）
        if nan_mask.any():
            filled = cp.where(nan_mask, 0, block)
            valid = (~nan_mask).astype(cp.float32)
            
            # 局所平均
            sum_values = uniform_filter(filled * valid, size=window_size, mode='constant', cval=0.0)
            sum_weights = uniform_filter(valid, size=window_size, mode='constant', cval=0.0)
            local_mean = cp.where(sum_weights > 0, sum_values / sum_weights, 0)
            
            # 局所平均二乗
            sum_sq_values = uniform_filter((filled ** 2) * valid, size=window_size, mode='constant', cval=0.0)
            local_mean_sq = cp.where(sum_weights > 0, sum_sq_values / sum_weights, 0)
        else:
            # 局所平均
            local_mean = uniform_filter(block, size=window_size, mode='constant', cval=0.0)
            # 局所平均二乗
            local_mean_sq = uniform_filter(block ** 2, size=window_size, mode='constant', cval=0.0)
        
        # 分散 = E[X²] - E[X]²
        variance = local_mean_sq - local_mean ** 2
        
        # 標準偏差
        sigma = cp.sqrt(cp.maximum(variance, 0.0))
        sigmas.append(sigma)
    
    # shape (H, W, n_scales)にスタック
    return cp.stack(sigmas, axis=-1)


def compute_fractal_dimension_block(block: cp.ndarray, *, 
                                    radii: Optional[List[int]] = None,
                                    normalize: bool = True,
                                    mean_global: float = None,
                                    std_global: float = None) -> cp.ndarray:
    """参考実装に準拠したフラクタル次元計算"""
    if radii is None:
        radii = [2, 4, 8, 16, 32]

    nan_mask = cp.isnan(block)
    
    # 1. マルチスケールroughness計算（window_mult=3はデフォルト）
    sigmas = compute_roughness_multiscale(block, radii, window_mult=3)
    
    # 2. log-log回帰（参考実装と同じ方法）
    log_scales = cp.log(cp.asarray(radii, dtype=cp.float32))
    n_scales = len(radii)
    
    # log(sigmas)を計算
    log_sigmas = cp.log(sigmas + Constants.EPSILON)
    
    # 回帰係数の計算（参考実装と同じ）
    mean_log_scale = cp.mean(log_scales)
    mean_log_sigma = cp.mean(log_sigmas, axis=2)
    
    # log_scalesをブロードキャスト
    log_scales_broadcast = log_scales.reshape(1, 1, -1)
    
    # 共分散と分散
    cov = cp.mean(log_sigmas * log_scales_broadcast, axis=2) - mean_log_sigma * mean_log_scale
    var_log_scale = cp.mean(log_scales ** 2) - mean_log_scale ** 2
    
    # Hurst指数
    H = cov / (var_log_scale + Constants.EPSILON)
    
    # フラクタル次元
    D = 3.0 - H
    
    # 3. 正規化処理
    if normalize and mean_global is not None and std_global is not None:
        # Z-score計算
        if std_global > 1e-6:
            Z = (D - mean_global) / std_global
            
            # より滑らかな正規化（tanh関数を使用）
            Z = cp.tanh(Z / 2.0)  # より緩やかな変換
            
            # 階調性を保つためのガンママッピング
            # より緩やかなガンマ値を使用
            gamma_value = 1.0 / (Constants.DEFAULT_GAMMA * 1.5)  # ガンマを弱める
            sign = cp.sign(Z)
            result = sign * cp.power(cp.abs(Z), gamma_value)
        else:
            result = cp.zeros_like(D)
    else:
        result = D
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)


def fractal_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """フラクタル次元Dの平均と標準偏差を計算"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        mean_D = float(cp.mean(valid_data))
        std_D = float(cp.std(valid_data))
        return (mean_D, std_D)
    return (2.5, 0.5)  # デフォルト値

class FractalAnomalyAlgorithm(DaskAlgorithm):
    """フラクタル異常検出アルゴリズム
    
    地形のフラクタル次元を計算し、統計的に異常な領域を検出します。
    - 正の値（明るい）: フラクタル次元が高い = 異常に複雑な地形
    - 負の値（暗い）: フラクタル次元が低い = 異常に平滑な地形
    - 0付近（中間色）: 典型的な地形パターン
    """
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radii = params.get('radii', None)

        # 半径の自動決定（修正: terrain_statsの初期化を追加）
        terrain_stats = params.get('terrain_stats', None)
        if radii is None:
            # terrain_statsがない場合は、ここで計算する
            if terrain_stats is None:
                logger.info("Analyzing terrain characteristics for automatic radii determination...")
                # dask_processor.pyのanalyze_terrain_characteristicsと同様の処理
                from ..core.dask_processor import analyze_terrain_characteristics
                terrain_stats = analyze_terrain_characteristics(gpu_arr, sample_ratio=0.01, include_fft=False)
                # pixel_sizeも設定
                terrain_stats['pixel_size'] = params.get('pixel_size', 1.0)
                
            radii, weights = determine_optimal_radii(terrain_stats)
            logger.info(f"Auto-determined radii for fractal analysis: {radii}")

        # window_mult=3を考慮したdepth
        depth = max(radii) * 3 + 1
        
        # グローバル統計量を計算（平均と標準偏差）
        logger.info("Computing global fractal dimension statistics...")
        stats = compute_global_stats(
            gpu_arr,
            fractal_stat_func,
            compute_fractal_dimension_block,
            {'radii': radii, 'normalize': False},
            downsample_factor=params.get('downsample_factor', None),
            depth=depth,
            algorithm_name='fractal_anomaly'
        )
        
        mean_D, std_D = stats
        logger.info(f"Fractal dimension: μ={mean_D:.3f}, σ={std_D:.3f}")
        
        # フルサイズで処理
        # 注意: compute_fractal_dimension_blockは既に特殊なガンママッピングを含んでいる
        # 結果は-1～1の範囲で、sign * |Z|^(1/gamma) の形式
        result = gpu_arr.map_overlap(
            compute_fractal_dimension_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radii=radii,
            normalize=True,
            mean_global=mean_D,
            std_global=std_D
        )
        
        # int16に変換
        # 既に特殊なガンママッピング済みなので、apply_gamma=False
        return result.map_blocks(
            lambda block: normalize_and_convert_to_int16(
                block,
                -1.0,  # 最小値（理論値）
                1.0,   # 最大値（理論値）
                apply_gamma=False,  # 既に特殊なガンママッピング済み
                signed_input=True   # 符号付き入力
            ),
            dtype=cp.int16,
            meta=cp.empty((0, 0), dtype=cp.int16)
        )
    
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
