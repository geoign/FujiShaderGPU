"""
FujiShaderGPU/algorithms/composite_terrain.py
複数の地形可視化手法を組み合わせた複合レンダリング
"""
import cupy as cp
from .base import BaseAlgorithm
from .hillshade import HillshadeAlgorithm
from .atmospheric_scattering import AtmosphericScatteringAlgorithm
from .rvi_gaussian import _compute_multiscale_rvi_ultra_fast


class CompositeTerrainAlgorithm(BaseAlgorithm):
    """複数の地形可視化手法を組み合わせる"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            # 各手法の有効/無効と重み
            "layers": {
                "hillshade": {"enabled": True, "weight": 1.2},
                "atmospheric": {"enabled": True, "weight": 0.8},
                "rvi": {"enabled": False, "weight": 0.5},
            },
            
            # Hillshadeのパラメータ
            "hillshade_params": {
                "azimuth": 315.0,
                "altitude": 45.0,
                "color_mode": "warm",
            },
            
            # Atmospheric Scatteringのパラメータ
            "atmospheric_params": {
                "tpi_radii": [4, 16, 64],
                "base_ambient": 0.35,
            },
            
            # RVIのパラメータ
            "rvi_params": {
                "target_distances": [10, 50, 250],
                "weights": [0.5, 0.3, 0.2],
            },
            
            # 合成パラメータ
            "blend_mode": "weighted",  # "weighted", "multiply", "overlay"
            "tone_mapping": True,      # トーンマッピングを行うか
            "gamma": 2.0,              # 最終的なガンマ補正
        }
    
    def process(self, dem_gpu, **params):
        """複数の手法を組み合わせて地形を可視化"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        # 各レイヤーの計算
        layers = []
        weights = []
        
        # 1. Hillshade
        if p["layers"]["hillshade"]["enabled"]:
            hillshade_algo = HillshadeAlgorithm()
            hillshade_params = p["hillshade_params"].copy()
            hillshade_params["pixel_size"] = params.get("pixel_size", 1.0)
            
            hillshade_result = hillshade_algo.process(dem_gpu, **hillshade_params)
            layers.append(hillshade_result)
            weights.append(p["layers"]["hillshade"]["weight"])
        
        # 2. Atmospheric Scattering
        if p["layers"]["atmospheric"]["enabled"]:
            atmospheric_algo = AtmosphericScatteringAlgorithm()
            atmospheric_result = atmospheric_algo.process(dem_gpu, **p["atmospheric_params"])
            layers.append(atmospheric_result)
            weights.append(p["layers"]["atmospheric"]["weight"])
        
        # 3. RVI（マルチスケール相対起伏）
        if p["layers"]["rvi"]["enabled"]:
            rvi_result = self._compute_rvi_layer(dem_gpu, p["rvi_params"], params.get("pixel_size", 1.0))
            layers.append(rvi_result)
            weights.append(p["layers"]["rvi"]["weight"])
        
        # レイヤーの合成
        if not layers:
            # 有効なレイヤーがない場合はグレースケールのDEMを返す
            return self._normalize_dem(dem_gpu)
        
        # 合成処理
        combined = self._blend_layers(layers, weights, p["blend_mode"])
        
        # トーンマッピング
        if p["tone_mapping"]:
            combined = self._apply_tone_mapping(combined, p["gamma"])
        
        return combined
    
    def _compute_rvi_layer(self, dem_gpu, rvi_params, pixel_size):
        """RVIレイヤーの計算"""
        # RVI計算
        rvi = _compute_multiscale_rvi_ultra_fast(
            dem_gpu,
            rvi_params["target_distances"],
            rvi_params["weights"],
            pixel_size
        )
        
        # 正規化
        rvi_min, rvi_max = cp.percentile(rvi, [2, 98])
        rvi_normalized = (rvi - rvi_min) / (rvi_max - rvi_min + 1e-8)
        rvi_normalized = cp.clip(rvi_normalized, 0, 1)
        
        # グレースケールからRGBへ（紫系の色調）
        rvi_color = cp.array([0.7, 0.5, 0.9])
        result = cp.zeros((*rvi.shape, 3), dtype=cp.float32)
        
        for i in range(3):
            result[:, :, i] = rvi_normalized * rvi_color[i]
        
        return result
    
    def _normalize_dem(self, dem_gpu):
        """DEMを正規化してRGB画像として返す"""
        dem_min = cp.percentile(dem_gpu, 0.1)
        dem_max = cp.percentile(dem_gpu, 99.9)
        dem_normalized = (dem_gpu - dem_min) / (dem_max - dem_min + 1e-10)
        dem_normalized = cp.clip(dem_normalized, 0, 1)
        
        # グレースケールRGB
        return cp.stack([dem_normalized] * 3, axis=-1)
    
    def _blend_layers(self, layers, weights, blend_mode):
        """複数のレイヤーをブレンド"""
        # 重みの正規化
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / cp.sum(weights)
        
        if blend_mode == "weighted":
            # 重み付き平均
            combined = cp.zeros_like(layers[0])
            for layer, weight in zip(layers, weights):
                combined += layer * weight
                
        elif blend_mode == "multiply":
            # 乗算ブレンド
            combined = cp.ones_like(layers[0])
            for layer, weight in zip(layers, weights):
                # 重み付き乗算（1に近づける）
                blend = 1.0 - weight + layer * weight
                combined *= blend
                
        elif blend_mode == "overlay":
            # オーバーレイブレンド
            combined = layers[0] * weights[0]
            for i in range(1, len(layers)):
                layer = layers[i]
                weight = weights[i]
                
                # オーバーレイ計算
                mask = combined < 0.5
                overlay = cp.where(
                    mask,
                    2 * combined * layer,
                    1 - 2 * (1 - combined) * (1 - layer)
                )
                
                # 重み付きブレンド
                combined = combined * (1 - weight) + overlay * weight
        
        else:
            # デフォルトは重み付き平均
            combined = cp.zeros_like(layers[0])
            for layer, weight in zip(layers, weights):
                combined += layer * weight
        
        return cp.clip(combined, 0, 1)
    
    def _apply_tone_mapping(self, image, gamma):
        """色バランスを保持したトーンマッピング"""
        # 明度の計算（ITU-R BT.709）
        if len(image.shape) == 3:
            luminance = (
                0.2126 * image[:, :, 0] + 
                0.7152 * image[:, :, 1] + 
                0.0722 * image[:, :, 2]
            )
        else:
            luminance = image
        
        # 明度の正規化
        lum_low, lum_high = cp.percentile(luminance, [2, 98])
        lum_range = lum_high - lum_low + 1e-6
        normalized_lum = cp.clip((luminance - lum_low) / lum_range, 0, 1)
        
        # 色相と彩度を保持しながら明度を調整
        lum_factor = cp.where(
            luminance > 1e-6, 
            normalized_lum / (luminance + 1e-6), 
            1.0
        )
        
        # 色バランスを保持して調整
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] *= lum_factor
        else:
            image *= lum_factor
        
        # ガンマ補正
        image = cp.power(cp.clip(image, 0, 1), 1.0 / gamma)
        
        return cp.clip(image, 0, 1)
