"""
FujiShaderGPU/algorithms/dask_algorithms.py
"""
from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, convolve
from tqdm.auto import tqdm

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
    if nan_mask.any():
        result[nan_mask] = cp.nan
        
    return result

class RVIAlgorithm(DaskAlgorithm):
    """Ridge-Valley Indexアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        sigmas = params.get('sigmas', [50])
        agg = params.get('agg', 'mean')
        show_progress = params.get('show_progress', True)
        
        # sigmasがNoneまたは空の場合はデフォルト値を使用
        if not sigmas:
            sigmas = [50]
        
        # マルチスケールRVIを計算
        rvi = multiscale_rvi(gpu_arr, sigmas=sigmas, agg=agg, show_progress=show_progress)
        
        # 正規化とガンマ補正（-1から+1の範囲に）
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
            if nan_mask.any():
                normalized[nan_mask] = cp.nan
            
            return normalized.astype(cp.float32)
        
        # 正規化を適用
        normalized_rvi = rvi.map_blocks(
            normalize_rvi,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        
        return normalized_rvi
    
    def get_default_params(self) -> dict:
        return {
            'sigmas': None,  # Noneの場合は自動決定
            'agg': 'mean',
            'auto_sigma': True
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
        depth = min(int(4 * sigma), 200)  # 最大depth=200に制限
        
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

def compute_hillshade_block(block: cp.ndarray, *, azimuth: float = 315, 
                           altitude: float = 45, z_factor: float = 1.0,
                           pixel_size: float = 1.0) -> cp.ndarray:
    """1ブロックに対するHillshade計算"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 方位角と高度角をラジアンに変換
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    
    # 勾配計算（中央差分）- NaNを含む場合は隣接値で補間
    if nan_mask.any():
        # NaNを隣接値で一時的に埋める
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        dy, dx = cp.gradient(filled * z_factor, pixel_size, edge_order=2)
    else:
        dy, dx = cp.gradient(block * z_factor, pixel_size, edge_order=2)
    
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
    if nan_mask.any():
        hillshade[nan_mask] = cp.nan
    
    return hillshade

class HillshadeAlgorithm(DaskAlgorithm):
    """Hillshadeアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        azimuth = params.get('azimuth', 315)
        altitude = params.get('altitude', 45)
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
            'azimuth': 315,
            'altitude': 45,
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
    
    saliency_maps = []
    
    # マルチスケールでの特徴抽出
    for scale in scales:
        # ガウシアンピラミッドの作成
        if scale > 1:
            if nan_mask.any():
                # NaN領域を考慮したガウシアンフィルタ
                filled = cp.where(nan_mask, 0, block)
                valid = (~nan_mask).astype(cp.float32)
                smoothed_values = gaussian_filter(filled * valid, sigma=scale, mode='nearest')
                smoothed_weights = gaussian_filter(valid, sigma=scale, mode='nearest')
                smoothed = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)
            else:
                smoothed = gaussian_filter(block, sigma=scale, mode='nearest')
            center = smoothed  # この行を追加
        else:
            smoothed = block.copy()
        
        # 局所的なコントラスト（Center-Surround差分）
        center_sigma = scale
        surround_sigma = scale * 2
        
        if nan_mask.any():
            # NaNがある場合はcenterとsurroundを両方計算
            center = gaussian_filter(filled * valid, sigma=center_sigma, mode='nearest')
            center_w = gaussian_filter(valid, sigma=center_sigma, mode='nearest')
            center = cp.where(center_w > 0, center / center_w, 0)
            
            surround = gaussian_filter(filled * valid, sigma=surround_sigma, mode='nearest')
            surround_w = gaussian_filter(valid, sigma=surround_sigma, mode='nearest')
            surround = cp.where(surround_w > 0, surround / surround_w, 0)
        else:
            # NaNがない場合の最適化
            if scale > 1:
                # smoothedはすでに計算済み（sigma=scale）なので再利用
                center = smoothed  # 再計算を避ける
                # surroundは追加のぼかし
                surround = gaussian_filter(center, sigma=scale, mode='nearest')
            else:
                # scale == 1 の場合
                center = smoothed  # または block.copy()
                surround = gaussian_filter(block, sigma=surround_sigma, mode='nearest')
        
        # 差分の絶対値
        contrast = cp.abs(center - surround)
        
        # 勾配の強度
        if scale == scales[0]:  # 最初のスケールのみ
            dy_orig, dx_orig = cp.gradient(block, pixel_size)
            gradient_mag_base = cp.sqrt(dx_orig**2 + dy_orig**2)
        # その後のスケールでは、基本勾配をスムージング
        gradient_mag = gaussian_filter(gradient_mag_base, sigma=scale/2, mode='nearest')

        # 特徴の組み合わせ
        feature = contrast * 0.5 + gradient_mag * 0.5
        saliency_maps.append(feature)
    
    # スケール間での正規化と統合
    combined_saliency = cp.zeros_like(block)
    for smap in saliency_maps:
        # 各マップを正規化
        valid_smap = smap[~nan_mask] if nan_mask.any() else smap
        if len(valid_smap) > 0 and cp.max(valid_smap) > cp.min(valid_smap):
            normalized = (smap - cp.min(valid_smap)) / (cp.max(valid_smap) - cp.min(valid_smap))
            combined_saliency += normalized
    
    combined_saliency /= len(scales)
    
    # ガンマ補正
    result = cp.power(combined_saliency, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class VisualSaliencyAlgorithm(DaskAlgorithm):
    """視覚的顕著性アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_scale = max(params.get('scales', [2, 4, 8, 16]))
        return gpu_arr.map_overlap(
            compute_visual_saliency_block,
            depth=int(max_scale * 4),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    
    # 正規化
    valid_nms = nms[~nan_mask] if nan_mask.any() else nms
    if len(valid_nms) > 0 and cp.max(valid_nms) > 0:
        nms = nms / cp.max(valid_nms)
    
    # ダブルスレッショルド
    strong = nms > threshold_high
    weak = (nms > threshold_low) & (nms <= threshold_high)
    
    # エッジの強調（NPRスタイル）
    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5
    
    # 輪郭線を反転（黒線で描画）
    result = 1.0 - edges
    
    # ガンマ補正
    result = cp.power(result, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR輪郭線アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=int(params.get('edge_sigma', 1.0) * 4 + 2),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        dy, dx = cp.gradient(filled, pixel_size)
    else:
        dy, dx = cp.gradient(block, pixel_size)
    
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    aspect = cp.arctan2(-dy, dx)
    
    azimuth_rad = cp.radians(315)
    altitude_rad = cp.radians(45)
    
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
    result = cp.power(result, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class AtmosphericPerspectiveAlgorithm(DaskAlgorithm):
    """大気遠近法アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_overlap(
            compute_atmospheric_perspective_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    
    # バッチ処理で高速化
    for r_factor in r_factors:
        r = radius * r_factor
        
        # 全方向の変位を一度に計算
        dx_all = (r * directions[:, 0] / pixel_size).astype(int)
        dy_all = (r * directions[:, 1] / pixel_size).astype(int)
        
        # パディング（最大変位分）
        max_disp = int(r / pixel_size) + 1
        padded = cp.pad(block, max_disp, mode='edge')
        
        for i, (dx, dy) in enumerate(zip(dx_all, dy_all)):
            # シフトされた配列を取得
            shifted = padded[max_disp+dy:max_disp+dy+h, 
                           max_disp+dx:max_disp+dx+w]
            
            # 高さの差と遮蔽角度
            height_diff = shifted - block
            occlusion_angle = cp.maximum(0, cp.arctan(height_diff / (r + 1e-6)))
            
            # 距離による減衰
            distance_factor = 1.0 - (r_factor * 0.5)
            
            # 遮蔽の累積
            valid = ~(cp.isnan(occlusion_angle) | nan_mask)
            occlusion_total += cp.where(valid, 
                                      occlusion_angle * distance_factor,
                                      0)
    
    # 正規化
    ao = 1.0 - (occlusion_total / (num_samples * 4)) * intensity
    ao = cp.clip(ao, 0, 1)
    
    # スムージング（NaN考慮）
    if nan_mask.any():
        filled_ao = cp.where(nan_mask, 1.0, ao)  # NaN領域は明るく（遮蔽なし）
        ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
    else:
        ao = gaussian_filter(ao, sigma=1.0, mode='nearest')
    
    # ガンマ補正
    result = cp.power(ao, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class AmbientOcclusionAlgorithm(DaskAlgorithm):
    """環境光遮蔽アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        # AOは計算量が多いため、チャンクごとに処理
        return gpu_arr.map_overlap(
            compute_ambient_occlusion_block,
            depth=int(params.get('radius', 10.0) + 1),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class TPIAlgorithm(DaskAlgorithm):
    """地形位置指数アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radius = params.get('radius', 10)
        return gpu_arr.map_overlap(
            compute_tpi_block,
            depth=radius+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        
        trend_values = gaussian_filter(filled * valid, sigma=sigma, mode='nearest')
        trend_weights = gaussian_filter(valid, sigma=sigma, mode='nearest')
        trend = cp.where(trend_weights > 0, trend_values / trend_weights, 0)
    else:
        trend = gaussian_filter(block, sigma=sigma, mode='nearest')
    
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
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class LRMAlgorithm(DaskAlgorithm):
    """局所起伏モデルアルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        kernel_size = params.get('kernel_size', 25)
        depth = int(kernel_size * 2)  # 十分なマージン
        return gpu_arr.map_overlap(
            compute_lrm_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    """開度の計算（ベクトル化された高速版）"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # 方向ベクトル
    angles = cp.linspace(0, 2 * cp.pi, num_directions, endpoint=False)
    cos_angles = cp.cos(angles)
    sin_angles = cp.sin(angles)
    
    # 結果の初期化
    if openness_type == 'positive':
        max_angles = cp.full((h, w), -cp.pi/2, dtype=cp.float32)
    else:
        max_angles = cp.full((h, w), cp.pi/2, dtype=cp.float32)
    
    # アダプティブな距離サンプリング
    for r_factor in cp.linspace(0.1, 1.0, 10):
        r = int(max_distance * r_factor)
        if r == 0:
            continue
        
        for i, (cos_a, sin_a) in enumerate(zip(cos_angles, sin_angles)):
            # オフセット計算
            offset_x = int(round(r * cos_a))
            offset_y = int(round(r * sin_a))
            
            if offset_x == 0 and offset_y == 0:
                continue
            
            # パディングしてシフト
            pad_x = (max(0, offset_x), max(0, -offset_x))
            pad_y = (max(0, offset_y), max(0, -offset_y))
            
            # NaN領域は非常に低い/高い値で埋める（遮蔽を計算しない）
            if nan_mask.any():
                fill_value = -1e6 if openness_type == 'positive' else 1e6
                padded = cp.pad(block, (pad_y, pad_x), mode='constant', constant_values=fill_value)
            else:
                padded = cp.pad(block, (pad_y, pad_x), mode='constant', constant_values=cp.nan)
            
            # シフトして切り出し
            if offset_y >= 0 and offset_x >= 0:
                shifted = padded[offset_y:offset_y+h, offset_x:offset_x+w]
            elif offset_y >= 0 and offset_x < 0:
                # 負のオフセットの場合、パディングされた配列の適切な部分を取得
                shifted = padded[offset_y:offset_y+h, pad_x[0]:pad_x[0]+w]
            elif offset_y < 0 and offset_x >= 0:
                shifted = padded[pad_y[0]:pad_y[0]+h, offset_x:offset_x+w]
            else:
                shifted = padded[pad_y[0]:pad_y[0]+h, pad_x[0]:pad_x[0]+w]
            
            # 高さの差と角度
            height_diff = shifted - block
            angle = cp.arctan(height_diff / (r * pixel_size))
            
            # 最大/最小角度の更新
            if openness_type == 'positive':
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.maximum(max_angles, angle), max_angles)
            else:
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.minimum(max_angles, angle), max_angles)
    
    # 開度の計算
    if openness_type == 'positive':
        openness = cp.pi/2 - max_angles
    else:
        openness = cp.pi/2 + max_angles
    
    # 正規化（0-1に）
    openness = cp.clip(openness / (cp.pi/2), 0, 1)
    
    # ガンマ補正
    result = cp.power(openness, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class OpennessAlgorithm(DaskAlgorithm):
    """開度アルゴリズム（簡易高速版）"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        return gpu_arr.map_overlap(
            compute_openness_vectorized,  # ベクトル化版を使用
            depth=max_distance+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    else:
        dy, dx = cp.gradient(block, pixel_size, edge_order=2)
    
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
    if nan_mask.any():
        slope[nan_mask] = cp.nan
    
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
                          light_azimuth: float = 315, light_altitude: float = 45) -> cp.ndarray:
    """金属光沢効果の計算（Cook-Torranceモデルの簡略版）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 法線ベクトルの計算
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    else:
        dy, dx = cp.gradient(block, pixel_size, edge_order=2)
    
    normal = cp.stack([-dx, -dy, cp.ones_like(dx)], axis=-1)
    normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
    
    # ラフネスの計算（局所的な標高の分散）
    kernel_size = int(roughness_scale)
    pad_width = kernel_size // 2
    padded = cp.pad(block, pad_width, mode='edge')
    
    roughness = cp.zeros_like(block)
    # ラフネスの高速計算（局所標準偏差）

    # 平均と平均二乗を計算
    mean_filter = uniform_filter(block, size=kernel_size, mode='constant')
    mean_sq_filter = uniform_filter(block**2, size=kernel_size, mode='constant')

    # 標準偏差 = sqrt(E[X^2] - E[X]^2)
    roughness = cp.sqrt(cp.maximum(mean_sq_filter - mean_filter**2, 0))

    # NaN処理
    if nan_mask.any():
        roughness[nan_mask] = cp.nan
    
    # ラフネスを正規化
    roughness_valid = roughness[~cp.isnan(roughness)]
    if len(roughness_valid) > 0:
        roughness = cp.clip(roughness / cp.max(roughness_valid), 0.01, 1.0)
    
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
    
    # スペキュラー計算
    n_dot_h = cp.clip(cp.dot(normal, half_vec), 0, 1)
    specular = cp.power(n_dot_h, shininess / roughness)
    
    # ガンマ補正（視覚的に適切な明るさに）
    result = cp.power(specular, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class SpecularAlgorithm(DaskAlgorithm):
    """金属光沢効果アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_overlap(
            compute_specular_block,
            depth=int(params.get('roughness_scale', 50)),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
        )
    
    def get_default_params(self) -> dict:
        return {
            'roughness_scale': 50.0,
            'shininess': 20.0,
            'light_azimuth': 315,
            'light_altitude': 45,
            'pixel_size': 1.0
        }

###############################################################################
# 2.4. Atmospheric Scattering (大気散乱光) アルゴリズム
###############################################################################

def compute_atmospheric_scattering_block(block: cp.ndarray, *, 
                                       scattering_strength: float = 0.5,
                                       intensity: float | None = None,
                                       pixel_size: float = 1.0) -> cp.ndarray:
    # intensity は scattering_strength のエイリアス（後方互換）
    if intensity is not None:
        scattering_strength = intensity
    """大気散乱によるシェーディング（Rayleigh散乱の簡略版）"""
    # NaNマスクを保存
    nan_mask = cp.isnan(block)
    
    # 法線計算
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    else:
        dy, dx = cp.gradient(block, pixel_size, edge_order=2)
    
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
    azimuth_rad = cp.radians(315)
    altitude_rad = cp.radians(45)
    aspect = cp.arctan2(-dy, dx)
    
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad)
    
    # 散乱光と直接光の合成
    result = ambient * 0.3 + hillshade * 0.7
    result = cp.clip(result, 0, 1)
    
    # ガンマ補正
    result = cp.power(result, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class AtmosphericScatteringAlgorithm(DaskAlgorithm):
    """大気散乱光アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_overlap(
            compute_atmospheric_scattering_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
        
        # 各スケールでの処理
        results = []
        for scale in scales:
            if scale > 1:
                depth = int(4 * scale)
                smoothed = gpu_arr.map_overlap(
                    lambda x: gaussian_filter(x, sigma=scale, mode='nearest'),
                    depth=depth,
                    boundary='reflect',
                    dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32)
                )
            else:
                smoothed = gpu_arr
            
            # 詳細成分の抽出
            detail = gpu_arr - smoothed
            results.append(detail)
        
        # 重み付き合成
        def weighted_combine(blocks_and_weights):
            """複数のブロックを重み付き合成"""
            blocks = blocks_and_weights[:-1]
            weights_arr = blocks_and_weights[-1]
            
            result = cp.zeros_like(blocks[0])
            for block, weight in zip(blocks, weights_arr):
                result += block * weight
            
            # NaNマスクを保存
            nan_mask = cp.isnan(blocks[0])
            
            # 正規化（-1から+1の範囲に）
            valid_result = result[~nan_mask]
            if len(valid_result) > 0:
                min_val = cp.min(valid_result)
                max_val = cp.max(valid_result)
                if max_val > min_val:
                    result = (result - min_val) / (max_val - min_val)
                else:
                    result = cp.full_like(result, 0.5)
            
            # ガンマ補正
            result = cp.power(result, 1/2.2)
            
            # NaN処理
            if nan_mask.any():
                result[nan_mask] = cp.nan
            
            return result.astype(cp.float32)
        
        # Dask配列として重みを作成
        weights_da = da.from_array(cp.array(weights), chunks=(len(weights),))
        
        # map_blocksで合成
        combined = da.map_blocks(
            weighted_combine,
            *results,
            weights_da,
            dtype=cp.float32,
            drop_axis=0,  # weightsの軸を削除
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
    result = cp.power(enhanced, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class FrequencyEnhancementAlgorithm(DaskAlgorithm):
    """周波数強調アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_blocks(
            enhance_frequency_block,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
    result = cp.power(result, 1/2.2)
    
    # NaN処理
    if nan_mask.any():
        result[nan_mask] = cp.nan
    
    return result.astype(cp.float32)

class CurvatureAlgorithm(DaskAlgorithm):
    """曲率アルゴリズム"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        return gpu_arr.map_overlap(
            compute_curvature_block,
            depth=2,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            **params
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
