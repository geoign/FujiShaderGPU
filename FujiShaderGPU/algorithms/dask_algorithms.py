"""
FujiShaderGPU/algorithms/dask_algorithms.py
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, convolve
from tqdm.auto import tqdm

class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 200
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6
    
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
                               weights: Optional[List[float]] = None,
                               pixel_size: float = 1.0) -> cp.ndarray:
    """効率的なRVI計算（メモリ最適化版）"""
    nan_mask = cp.isnan(block)
    
    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    else:
        weights = cp.array(weights, dtype=cp.float32)
    
    # 結果をインプレースで累積（メモリ効率向上）
    rvi_combined = cp.zeros_like(block, dtype=cp.float32)
    
    # NaN処理の前処理
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
    else:
        filled = block
        valid = None
    
    for radius, weight in zip(radii, weights):
        if radius <= 1:
            # 小さな半径の場合
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=1.0, mode='nearest')
        else:
            # 大きな半径の場合
            kernel_size = 2 * radius + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='nearest')
        
        # インプレース演算でメモリ効率向上
        rvi_combined += weight * (block - mean_elev)
        
        # 明示的なメモリ解放は不要（CuPyが管理）
    
    # NaN処理
    rvi_combined = restore_nan(rvi_combined, nan_mask)
    
    return rvi_combined


def multiscale_rvi_efficient(gpu_arr: da.Array, *, 
                            radii: List[int], 
                            weights: Optional[List[float]] = None,
                            show_progress: bool = True) -> da.Array:
    """効率的なマルチスケールRVI（Dask版）"""
    
    if not radii:
        raise ValueError("At least one radius value is required")
    
    # 最大半径に基づいてdepthを設定（Gaussianよりも大幅に小さい）
    max_radius = max(radii)
    depth = max_radius + 1  # 半径+1で十分
    
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
        # パラメータ取得
        mode = params.get('mode', 'radius')  # 'radius' or 'sigma'
        show_progress = params.get('show_progress', True)
        
        if mode == 'sigma':
            # 従来のsigmaベース（互換性のため残す）
            sigmas = params.get('sigmas', [50])
            agg = params.get('agg', 'mean')
            rvi = multiscale_rvi(gpu_arr, sigmas=sigmas, agg=agg, show_progress=show_progress)
        else:
            # 新しい効率的な半径ベース
            radii = params.get('radii', None)
            weights = params.get('weights', None)
            
            # 自動決定
            if radii is None:
                pixel_size = params.get('pixel_size', 1.0)
                radii = self._determine_optimal_radii(pixel_size)
            
            rvi = multiscale_rvi_efficient(gpu_arr, radii=radii, weights=weights, 
                                         show_progress=show_progress)
        
        # 正規化とガンマ補正
        def normalize_rvi(block):
            # NaNマスクを保存
            nan_mask = cp.isnan(block)
            
            # 有効な値の範囲を取得
            valid_block = block[~nan_mask]
            if len(valid_block) > 0:
                # 標準偏差でスケーリング
                std = cp.std(valid_block)
                if std > 0:
                    normalized = block / (3 * std)  # ±3σの範囲に
                    normalized = cp.clip(normalized, -1, 1)
                else:
                    normalized = cp.zeros_like(block)
            else:
                normalized = cp.zeros_like(block)
            
            # NaN位置を復元
            normalized = restore_nan(normalized, nan_mask)
            
            return normalized.astype(cp.float32)
        
        # 正規化を適用
        normalized_rvi = rvi.map_blocks(
            normalize_rvi,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        
        return normalized_rvi
    
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
            import numpy as np
            indices = np.logspace(0, np.log10(len(radii)-1), 4).astype(int)
            radii = [radii[i] for i in indices]
        
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
        # depth = int(4 * sigma)  # この行を削除
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
                        lambda x: gaussian_filter(x, sigma=sigma, mode='nearest'),
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

def compute_visual_saliency_block(block: cp.ndarray, *, scales: List[float] = [2, 4, 8, 16],
                                pixel_size: float = 1.0) -> cp.ndarray:
    """視覚的顕著性の計算（Itti-Kochモデルの簡略版）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 基本の勾配を最初に計算（すべてのスケールで使用）
    # 修正: NaN処理を追加
    dy_orig, dx_orig, _ = handle_nan_for_gradient(block, scale=1.0, pixel_size=pixel_size)
    gradient_mag_base = cp.sqrt(dx_orig**2 + dy_orig**2)
    gradient_mag_base = cp.where(cp.isnan(gradient_mag_base), 0, gradient_mag_base)
    
    saliency_maps = []
    
    # マルチスケールでの特徴抽出
    for scale in scales:
        # 局所的なコントラスト（Center-Surround差分）
        center_sigma = scale
        surround_sigma = scale * 2
        
        if nan_mask.any():
            # NaNがある場合はcenterとsurroundを両方計算
            center, _ = handle_nan_with_gaussian(block, sigma=center_sigma, mode='nearest')
            surround, _ = handle_nan_with_gaussian(block, sigma=surround_sigma, mode='nearest')
        else:
            # NaNがない場合
            center = gaussian_filter(block, sigma=center_sigma, mode='nearest')
            surround = gaussian_filter(block, sigma=surround_sigma, mode='nearest')
        
        # 差分の絶対値
        contrast = cp.abs(center - surround)
        
        # 勾配の強度（スケールに応じてスムージング）
        if scale > 1:
            # 修正: NaN処理を追加
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
        
        # 特徴の組み合わせ
        feature = contrast * 0.5 + gradient_mag * 0.5
        feature = cp.where(cp.isnan(feature), 0, feature)
        saliency_maps.append(feature)
    
    # スケール間での正規化と統合
    combined_saliency = cp.zeros_like(block)
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
    
    # 平均化
    if valid_count > 0:
        combined_saliency /= valid_count
    else:
        # すべてのマップが無効な場合のフォールバック
        combined_saliency = cp.full_like(block, 0.5)
    
    # ガンマ補正
    result = cp.power(combined_saliency, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class VisualSaliencyAlgorithm(DaskAlgorithm):
    """視覚的顕著性アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [2, 4, 8, 16])
        pixel_size = params.get('pixel_size', 1.0)
        max_scale = max(scales)
        
        return gpu_arr.map_overlap(
            compute_visual_saliency_block,
            depth=int(max_scale * 4),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'scales': [2, 4, 8, 16],
            'pixel_size': 1.0
        }

###############################################################################
# 2.9. NPR Edges (非写実的レンダリング輪郭線) アルゴリズム
###############################################################################

def compute_npr_edges_block(block: cp.ndarray, *, edge_sigma: float = 1.0,
                          threshold_low: float = 0.1, threshold_high: float = 0.3,
                          pixel_size: float = 1.0) -> cp.ndarray:
    """NPRスタイルの輪郭線抽出（Cannyエッジ検出の変形）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # ノイズ除去（NaN考慮）
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        smoothed = gaussian_filter(filled, sigma=edge_sigma, mode='nearest')
    else:
        smoothed = gaussian_filter(block, sigma=edge_sigma, mode='nearest')
    
    # Sobelフィルタによる勾配計算
    dy, dx = cp.gradient(smoothed, pixel_size)
    gradient_mag = cp.sqrt(dx**2 + dy**2)
    gradient_dir = cp.arctan2(dy, dx)
    
    # 勾配の統計情報を使用して適応的な閾値を計算
    valid_grad = gradient_mag[~nan_mask] if nan_mask.any() else gradient_mag
    if len(valid_grad) > 0:
        # パーセンタイルベースの閾値設定
        low_percentile = 70  # 上位30%
        high_percentile = 90  # 上位10%
        
        grad_low = cp.percentile(valid_grad, low_percentile)
        grad_high = cp.percentile(valid_grad, high_percentile)
        
        # ユーザー指定の閾値でスケーリング
        actual_threshold_low = grad_low + (grad_high - grad_low) * threshold_low
        actual_threshold_high = grad_low + (grad_high - grad_low) * threshold_high
    else:
        actual_threshold_low = threshold_low
        actual_threshold_high = threshold_high
    
    # 非最大値抑制（簡易版）
    # 勾配方向を8方向に量子化
    angle = gradient_dir * 180.0 / cp.pi
    angle[angle < 0] += 180
    
    # 8方向でのシフトによる非最大値抑制
    nms = gradient_mag.copy()
    
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
    
    # ダブルスレッショルド（適応的閾値を使用）
    strong = nms > actual_threshold_high
    weak = (nms > actual_threshold_low) & (nms <= actual_threshold_high)
    
    # エッジの強調（NPRスタイル）
    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5
    
    # ヒステリシス処理（簡易版）- 弱いエッジを強いエッジに接続
    for _ in range(2):  # 2回繰り返して接続を強化
        # 修正: maximum.reduceの代わりに順次maximumを使用
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
        
        edges = cp.where(weak & (dilated > 0.5), 1.0, edges)
    
    # エッジ強度を調整（完全な黒線を避ける）
    edges = edges * 0.8  # エッジの最大強度を0.8に
    
    # 輪郭線を反転（黒線で描画）
    result = 1.0 - edges
    
    # コントラスト調整（エッジをより見やすく）
    result = cp.clip(result, 0.2, 1.0)  # 最小値を0.2に設定
    
    # ガンマ補正
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR輪郭線アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        edge_sigma = params.get('edge_sigma', 1.0)
        threshold_low = params.get('threshold_low', 0.2)  # デフォルト値を調整
        threshold_high = params.get('threshold_high', 0.5)  # デフォルト値を調整
        pixel_size = params.get('pixel_size', 1.0)
        
        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=int(edge_sigma * 4 + 2),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            edge_sigma=edge_sigma,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'edge_sigma': 1.0,
            'threshold_low': 0.2,   # 0.1から0.2に変更
            'threshold_high': 0.5,  # 0.3から0.5に変更
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
    """スクリーン空間環境光遮蔽（SSAO）の地形版（デバッグ版）"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # デバッグ: 入力データの確認
    if not nan_mask.all():
        valid_data = block[~nan_mask]
        print(f"[DEBUG] Input block stats - min: {cp.min(valid_data):.2f}, max: {cp.max(valid_data):.2f}, mean: {cp.mean(valid_data):.2f}")
    
    # サンプリング方向を事前計算
    angles = cp.linspace(0, 2 * cp.pi, num_samples, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    # 距離のサンプリング（より細かく）
    r_factors = cp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # 遮蔽カウント用
    occlusion_count = cp.zeros((h, w), dtype=cp.float32)
    total_samples = cp.zeros((h, w), dtype=cp.float32)
    
    # デバッグ用カウンタ
    positive_occlusion_count = 0
    total_comparisons = 0
    
    # バッチ処理で高速化
    for r_idx, r_factor in enumerate(r_factors):
        r = radius * r_factor
        
        # 全方向の変位を一度に計算
        dx_all = cp.round(r * directions[:, 0] / pixel_size).astype(int)
        dy_all = cp.round(r * directions[:, 1] / pixel_size).astype(int)
        
        for i in range(num_samples):
            dx = int(dx_all[i])
            dy = int(dy_all[i])
            
            if dx == 0 and dy == 0:
                continue
            
            # 必要な方向のみパディング（constant modeに変更）
            pad_left = max(0, -dx)
            pad_right = max(0, dx)
            pad_top = max(0, -dy)
            pad_bottom = max(0, dy)
            
            # パディング（constant modeで最小値を使用）
            if nan_mask.any():
                pad_value = cp.nanmin(block)
            else:
                pad_value = cp.min(block)
            
            padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='constant', constant_values=pad_value)
            
            # シフト
            start_y = pad_top + dy
            start_x = pad_left + dx
            shifted = padded[start_y:start_y+h, start_x:start_x+w]
            
            # 高さの差
            height_diff = shifted - block
            
            # 有効なピクセルのマスク
            valid = ~(cp.isnan(shifted) | nan_mask)
            
            # デバッグ: 最初のサンプルで統計を出力
            if r_idx == 0 and i == 0 and cp.any(valid):
                valid_diff = height_diff[valid]
                if len(valid_diff) > 0:
                    print(f"[DEBUG] Height diff stats - min: {cp.min(valid_diff):.4f}, max: {cp.max(valid_diff):.4f}, positive ratio: {cp.sum(valid_diff > 0) / len(valid_diff):.2%}")
            
            # 遮蔽の判定（高い点による遮蔽）
            # height_diffが正の場合、サンプル点の方が高い = 遮蔽あり
            is_occluded = (height_diff > 0) & valid
            
            # 距離による重み（近いほど影響大）
            weight = 1.0 - r_factor * 0.5
            
            # 遮蔽カウント
            occlusion_count += cp.where(is_occluded, weight, 0)
            total_samples += cp.where(valid, weight, 0)
            
            # デバッグカウンタ更新
            total_comparisons += cp.sum(valid)
            positive_occlusion_count += cp.sum(is_occluded)
    
    # デバッグ: 遮蔽統計
    if total_comparisons > 0:
        occlusion_ratio = positive_occlusion_count / total_comparisons
        print(f"[DEBUG] Total occlusion ratio: {occlusion_ratio:.2%}")
        print(f"[DEBUG] Occlusion count - min: {cp.min(occlusion_count):.2f}, max: {cp.max(occlusion_count):.2f}")
        print(f"[DEBUG] Total samples - min: {cp.min(total_samples):.2f}, max: {cp.max(total_samples):.2f}")
    
    # ゼロ除算を防ぐ
    total_samples = cp.maximum(total_samples, 1.0)
    
    # 遮蔽率を計算（0-1の範囲）
    occlusion_ratio = occlusion_count / total_samples
    
    # デバッグ: 遮蔽率の統計
    print(f"[DEBUG] Occlusion ratio - min: {cp.min(occlusion_ratio):.4f}, max: {cp.max(occlusion_ratio):.4f}, mean: {cp.mean(occlusion_ratio):.4f}")
    
    # AOの計算（遮蔽が多いほど暗く）
    ao = 1.0 - occlusion_ratio * intensity
    ao = cp.clip(ao, 0, 1)
    
    # デバッグ: AO値の統計
    print(f"[DEBUG] AO before smoothing - min: {cp.min(ao):.4f}, max: {cp.max(ao):.4f}, mean: {cp.mean(ao):.4f}")
    
    # スムージング（NaN考慮）
    if nan_mask.any():
        filled_ao = cp.where(nan_mask, 1.0, ao)
        ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
    else:
        ao = gaussian_filter(ao, sigma=1.0, mode='nearest')
    
    # ガンマ補正
    result = cp.power(ao, Constants.DEFAULT_GAMMA)
    
    # デバッグ: 最終結果の統計
    print(f"[DEBUG] Final result - min: {cp.min(result):.4f}, max: {cp.max(result):.4f}, mean: {cp.mean(result):.4f}")
    print("-" * 50)
    
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
        
        # AOは計算量が多いため、チャンクごとに処理
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
            'radius': 10.0,
            'intensity': 1.0,
            'pixel_size': 1.0
        }

###############################################################################
# 2.12. TPI (Topographic Position Index) アルゴリズム
###############################################################################

def compute_tpi_block(block: cp.ndarray, *, radius: int = 10,
                     pixel_size: float = 1.0) -> cp.ndarray:
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
    
    # 標準化（-1から1の範囲に）
    valid_tpi = tpi[~nan_mask] if nan_mask.any() else tpi
    if len(valid_tpi) > 0:
        max_abs = cp.maximum(cp.abs(cp.min(valid_tpi)), cp.abs(cp.max(valid_tpi)))
        if max_abs > 0:
            tpi = tpi / max_abs
    
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
        pixel_size = params.get('pixel_size', 1.0)  # pixel_sizeを取得
        return gpu_arr.map_overlap(
            compute_tpi_block,
            depth=radius+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radius=radius,  # 明示的に必要なパラメータのみ渡す
            pixel_size=pixel_size  # 明示的に必要なパラメータのみ渡す
        )
    
    def get_default_params(self) -> dict:
        return {
            'radius': 10,  # 解析半径（ピクセル）
            'pixel_size': 1.0
        }

###############################################################################
# 2.13. LRM (Local Relief Model) アルゴリズム
###############################################################################

def compute_lrm_block(block: cp.ndarray, *, kernel_size: int = 25,
                     pixel_size: float = 1.0) -> cp.ndarray:
    """局所起伏モデルの計算（微地形の強調）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 大規模な地形トレンドを計算（NaN考慮）
    sigma = kernel_size / 3.0
    trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
    
    # 微地形の抽出
    lrm = block - trend
    
    # 正規化（標準偏差でスケーリング）
    valid_lrm = lrm[~nan_mask] if nan_mask.any() else lrm
    if len(valid_lrm) > 0:
        std = cp.std(valid_lrm)
        if std > 0:
            lrm = lrm / (3 * std)  # ±3σの範囲に
    
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
        depth = int(kernel_size * 2)  # 十分なマージン
        return gpu_arr.map_overlap(
            compute_lrm_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            kernel_size=kernel_size,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'kernel_size': 25,  # トレンド除去のカーネルサイズ
            'pixel_size': 1.0
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
            
            # パディング（NaN考慮）
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
    """勾配（傾斜）アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        unit = params.get('unit', 'degree')
        pixel_size = params.get('pixel_size', 1.0)
        
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
    kernel_size = int(roughness_scale)
    
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
        cp.cos(light_az_rad) * cp.cos(light_alt_rad),
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
            'roughness_scale': 20.0,  # 50.0から20.0に変更
            'shininess': 10.0,        # 20.0から10.0に変更
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
        
        if weights is None:
            # デフォルト：スケールに反比例する重み
            weights = [1.0 / s for s in scales]
        
        # 重みを正規化
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        
        # 各スケールでの処理
        results = []
        for scale in scales:
            if scale > 1:
                depth = min(int(4 * scale), Constants.MAX_DEPTH)  # depthを制限
                
                # NaN対応のガウシアンフィルタを使用
                def smooth_with_nan_handling(block):
                    smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=scale, mode='nearest')
                    return smoothed
                
                smoothed = gpu_arr.map_overlap(
                    smooth_with_nan_handling,
                    depth=depth,
                    boundary='reflect',
                    dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32)
                )
            else:
                smoothed = gpu_arr
            
            # 詳細成分の抽出（NaN対応版）
            def extract_detail(original, smoothed):
                """NaNを考慮した詳細成分の抽出"""
                nan_mask = cp.isnan(original)
                
                # NaNでない部分のみ差分を計算
                detail = cp.zeros_like(original)
                valid_mask = ~nan_mask & ~cp.isnan(smoothed)
                detail[valid_mask] = original[valid_mask] - smoothed[valid_mask]
                
                # NaN位置を保持
                detail[nan_mask] = cp.nan
                
                return detail
            
            detail = da.map_blocks(
                extract_detail,
                gpu_arr,
                smoothed,
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32)
            )
            results.append(detail)
        
        # 重み付き合成（NaN対応版）
        def weighted_combine_with_nan(*blocks):
            """NaNを考慮した複数ブロックの重み付き合成"""
            # 最初のブロックからNaNマスクを取得
            nan_mask = cp.isnan(blocks[0])
            
            # 各ピクセルで有効な値の重み付き平均を計算
            result = cp.zeros_like(blocks[0])
            weight_sum = cp.zeros_like(blocks[0])
            
            for i, block in enumerate(blocks):
                # 現在のブロックの有効なピクセル
                valid = ~cp.isnan(block)
                
                # 有効なピクセルのみ加算
                result[valid] += block[valid] * weights[i]
                weight_sum[valid] += weights[i]
            
            # 重みの合計で除算（ゼロ除算を回避）
            result = cp.where(weight_sum > 0, result / weight_sum, 0)
            
            # 正規化（NaNでない値のみを使用）
            valid_result = result[~nan_mask]
            if len(valid_result) > 0:
                min_val = cp.min(valid_result)
                max_val = cp.max(valid_result)
                if max_val > min_val:
                    result = (result - min_val) / (max_val - min_val)
                else:
                    result = cp.full_like(result, 0.5)
            else:
                result = cp.full_like(result, 0.5)
            
            # ガンマ補正
            result = cp.power(result, Constants.DEFAULT_GAMMA)
            
            # 元のNaN位置を復元
            result[nan_mask] = cp.nan
            
            return result.astype(cp.float32)
        
        # map_blocksで合成
        combined = da.map_blocks(
            weighted_combine_with_nan,
            *results,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        
        return combined
    
    def get_default_params(self) -> dict:
        return {
            'scales': [1, 10, 50, 100],
            'weights': None  # Noneの場合は自動計算
        }

###############################################################################
# 2.6. Frequency Enhancement (周波数強調) アルゴリズム
###############################################################################

def enhance_frequency_block(block: cp.ndarray, *, target_frequency: float = 0.1,
                          bandwidth: float = 0.05, enhancement: float = 2.0) -> cp.ndarray:
    """特定周波数成分の強調"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # NaNを平均値で一時的に埋める
    if nan_mask.any():
        block_filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        block_filled = block
    
    # 2D FFT
    fft = cp.fft.fft2(block_filled)
    freq_x = cp.fft.fftfreq(block.shape[0])
    freq_y = cp.fft.fftfreq(block.shape[1])
    freq_grid = cp.sqrt(freq_x[:, None]**2 + freq_y[None, :]**2)
    
    # バンドパスフィルタ
    filter_mask = cp.exp(-((freq_grid - target_frequency)**2) / (2 * bandwidth**2))
    filter_mask = 1 + (enhancement - 1) * filter_mask
    
    # フィルタ適用
    filtered_fft = fft * filter_mask
    enhanced = cp.real(cp.fft.ifft2(filtered_fft))
    
    # 正規化
    valid_enhanced = enhanced[~nan_mask]
    if len(valid_enhanced) > 0:
        min_val = cp.min(valid_enhanced)
        max_val = cp.max(valid_enhanced)
        if max_val > min_val:
            enhanced = (enhanced - min_val) / (max_val - min_val)
        else:
            enhanced = cp.full_like(enhanced, 0.5)
    
    # ガンマ補正
    result = cp.power(enhanced, Constants.DEFAULT_GAMMA)
    
    # NaN処理
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class FrequencyEnhancementAlgorithm(DaskAlgorithm):
    """周波数強調アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        target_frequency = params.get('target_frequency', 0.1)
        bandwidth = params.get('bandwidth', 0.05)
        enhancement = params.get('enhancement', 2.0)
        
        return gpu_arr.map_blocks(
            enhance_frequency_block,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            target_frequency=target_frequency,
            bandwidth=bandwidth,
            enhancement=enhancement
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
