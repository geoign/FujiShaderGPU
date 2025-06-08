"""
FujiShaderGPU/algorithms/visual_saliency.py
視覚的顕著性に基づく地形強調をGPUで高速計算
"""
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from .base import BaseAlgorithm


class VisualSaliencyAlgorithm(BaseAlgorithm):
    """GPU上で視覚的顕著性を計算して地形を強調"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "center_surround_scales": [(1, 4), (2, 8), (4, 16)],  # センターサラウンド差分のスケール
            "edge_weight": 0.5,          # エッジ強度の重み
            "saliency_weight": 0.5,      # 顕著性の重み
            "enhancement_factor": 2.0,    # 強調係数
            "color_output": True,        # カラー出力するか
            "smoothing": 1.0,            # 最終的なスムージング
        }
    
    def process(self, dem_gpu, **params):
        """GPU上で視覚的顕著性を計算"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        # DEMの正規化
        dem_normalized = self._normalize_dem(dem_gpu)
        
        # 1. エッジ強度（勾配の大きさ）
        edge_saliency = self._compute_edge_saliency(
            dem_normalized, 
            p["edge_weight"]
        )
        
        # 2. センターサラウンド差分による顕著性
        center_surround_saliency = self._compute_center_surround_saliency(
            dem_normalized,
            p["center_surround_scales"],
            p["saliency_weight"]
        )
        
        # 3. 顕著性マップの統合
        total_saliency = edge_saliency + center_surround_saliency
        total_saliency = self._normalize_saliency(total_saliency)
        
        # 4. 顕著性による強調
        enhanced = self._apply_saliency_enhancement(
            dem_normalized,
            total_saliency,
            p["enhancement_factor"]
        )
        
        # 5. スムージング（オプション）
        if p["smoothing"] > 0:
            enhanced = cpx_ndimage.gaussian_filter(
                enhanced, 
                sigma=p["smoothing"],
                mode='reflect'
            )
        
        # 6. 出力形式の処理
        if p["color_output"]:
            result = self._create_color_output(enhanced, total_saliency)
        else:
            result = cp.stack([enhanced] * 3, axis=-1)
        
        return result
    
    def _normalize_dem(self, dem_gpu):
        """DEMを正規化"""
        dem_min = cp.percentile(dem_gpu, 0.1)
        dem_max = cp.percentile(dem_gpu, 99.9)
        dem_normalized = (dem_gpu - dem_min) / (dem_max - dem_min + 1e-10)
        return cp.clip(dem_normalized, 0, 1)
    
    def _compute_edge_saliency(self, dem, weight):
        """エッジベースの顕著性（勾配強度）"""
        if weight <= 0:
            return cp.zeros_like(dem)
        
        # Sobelフィルタで勾配計算
        grad_x = cpx_ndimage.sobel(dem, axis=1, mode='reflect')
        grad_y = cpx_ndimage.sobel(dem, axis=0, mode='reflect')
        
        # 勾配の大きさ
        grad_magnitude = cp.sqrt(grad_x**2 + grad_y**2)
        
        # 正規化
        grad_max = cp.max(grad_magnitude)
        if grad_max > 0:
            grad_magnitude = grad_magnitude / grad_max
        
        return grad_magnitude * weight
    
    def _compute_center_surround_saliency(self, dem, scales, weight):
        """センターサラウンド差分による顕著性"""
        if weight <= 0:
            return cp.zeros_like(dem)
        
        saliency = cp.zeros_like(dem)
        
        for center_scale, surround_scale in scales:
            # センターとサラウンドの計算
            center = cpx_ndimage.gaussian_filter(
                dem, 
                sigma=center_scale,
                mode='reflect'
            )
            surround = cpx_ndimage.gaussian_filter(
                dem, 
                sigma=surround_scale,
                mode='reflect'
            )
            
            # 差分の絶対値
            diff = cp.abs(center - surround)
            saliency += diff
        
        # スケール数で平均化
        saliency = saliency / len(scales)
        
        return saliency * weight
    
    def _normalize_saliency(self, saliency):
        """顕著性マップの正規化"""
        # パーセンタイルベースの正規化
        low, high = cp.percentile(saliency, [5, 95])
        
        if high - low > 1e-6:
            saliency = (saliency - low) / (high - low)
        
        return cp.clip(saliency, 0, 1)
    
    def _apply_saliency_enhancement(self, dem, saliency, factor):
        """顕著性に基づく強調"""
        # 顕著性が高い部分ほど強調
        enhancement = 1.0 + saliency * (factor - 1.0)
        enhanced = dem * enhancement
        
        # クリッピング
        return cp.clip(enhanced, 0, 1)
    
    def _create_color_output(self, enhanced, saliency):
        """カラー出力の作成（顕著性を色相で表現）"""
        # HSV的な色づけ
        # H: 顕著性（赤→黄→緑→青）
        # S: 固定値
        # V: 強調されたDEM値
        
        result = cp.zeros((*enhanced.shape, 3), dtype=cp.float32)
        
        # 簡易的なカラーマッピング
        # 低顕著性: 青系、高顕著性: 赤系
        hue = 0.7 - saliency * 0.7  # 青(0.7) → 赤(0.0)
        
        # HSVからRGBへの簡易変換
        # ここでは単純化のため、特定の色相のみ使用
        result[:, :, 0] = enhanced * (1.0 - hue)      # 赤成分
        result[:, :, 1] = enhanced * (1.0 - cp.abs(hue - 0.5) * 2)  # 緑成分
        result[:, :, 2] = enhanced * hue              # 青成分
        
        return cp.clip(result, 0, 1)


# 追加: 多重解像度顕著性
class MultiResolutionSaliency(BaseAlgorithm):
    """多重解像度での顕著性計算"""
    
    def get_default_params(self):
        return {
            "pyramid_levels": 4,         # ピラミッドレベル数
            "feature_types": ["intensity", "orientation"],  # 特徴タイプ
            "normalization": "local",    # 正規化方法
        }
    
    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        
        # ガウシアンピラミッドの構築
        pyramid = self._build_gaussian_pyramid(dem_gpu, p["pyramid_levels"])
        
        # 各特徴マップの計算
        feature_maps = []
        
        if "intensity" in p["feature_types"]:
            intensity_maps = self._compute_intensity_features(pyramid)
            feature_maps.extend(intensity_maps)
        
        if "orientation" in p["feature_types"]:
            orientation_maps = self._compute_orientation_features(pyramid)
            feature_maps.extend(orientation_maps)
        
        # 顕著性マップの統合
        saliency_map = self._combine_feature_maps(feature_maps, dem_gpu.shape)
        
        # 結果の作成
        enhanced = dem_gpu * (1.0 + saliency_map)
        enhanced = (enhanced - cp.min(enhanced)) / (cp.max(enhanced) - cp.min(enhanced) + 1e-6)
        
        return cp.stack([enhanced] * 3, axis=-1)
    
    def _build_gaussian_pyramid(self, image, levels):
        """ガウシアンピラミッドの構築"""
        pyramid = [image]
        current = image
        
        for _ in range(1, levels):
            # ダウンサンプリング前にガウシアンフィルタ
            blurred = cpx_ndimage.gaussian_filter(current, sigma=1.0, mode='reflect')
            # 2x2ダウンサンプリング
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid
    
    def _compute_intensity_features(self, pyramid):
        """強度特徴の計算"""
        features = []
        
        # センターサラウンド差分
        for c in [2, 3]:
            for s in [c+3, c+4]:
                if s < len(pyramid):
                    # アップサンプリングして差分
                    center = pyramid[c]
                    surround = self._upsample_to_size(pyramid[s], center.shape)
                    diff = cp.abs(center - surround)
                    features.append(diff)
        
        return features
    
    def _compute_orientation_features(self, pyramid):
        """方向特徴の計算"""
        features = []
        orientations = [0, 45, 90, 135]
        
        for level in pyramid[1:]:  # 最細レベルをスキップ
            for angle in orientations:
                # 簡易的な方向性フィルタ
                if angle == 0:
                    kernel = cp.array([[-1, 0, 1]], dtype=cp.float32)
                elif angle == 45:
                    kernel = cp.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=cp.float32)
                elif angle == 90:
                    kernel = cp.array([[-1], [0], [1]], dtype=cp.float32)
                else:  # 135
                    kernel = cp.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=cp.float32)
                
                filtered = cpx_ndimage.convolve(level, kernel, mode='reflect')
                features.append(cp.abs(filtered))
        
        return features
    
    def _upsample_to_size(self, image, target_shape):
        """画像を目標サイズにアップサンプリング"""
        if image.shape == target_shape:
            return image
        
        # バイリニア補間でアップサンプリング
        zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
        return cpx_ndimage.zoom(image, zoom_factors, order=1, mode='reflect')
    
    def _combine_feature_maps(self, features, target_shape):
        """特徴マップの統合"""
        combined = cp.zeros(target_shape, dtype=cp.float32)
        
        for feature in features:
            # 目標サイズにリサイズ
            resized = self._upsample_to_size(feature, target_shape)
            # 正規化
            if cp.max(resized) > 0:
                resized = resized / cp.max(resized)
            combined += resized
        
        # 最終正規化
        if cp.max(combined) > 0:
            combined = combined / cp.max(combined)
        
        return combined
