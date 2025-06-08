"""
FujiShaderGPU/algorithms/tile_algorithms.py
Windows/macOS向けタイルベース処理用のアルゴリズム集
全てのアルゴリズムはTileAlgorithmを継承したクラスとして実装
"""
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
import cupyx.scipy.fft as cpx_fft
from abc import ABC, abstractmethod
from typing import Dict, Any

class TileAlgorithm(ABC):
    """地形解析アルゴリズムの基底クラス"""
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """デフォルトパラメータを返す"""
        pass
    
    @abstractmethod
    def process(self, dem_gpu: cp.ndarray, **params) -> cp.ndarray:
        """
        GPU上でアルゴリズムを実行
        
        Parameters
        ----------
        dem_gpu : cp.ndarray
            GPU上のDEMデータ
        **params : dict
            アルゴリズム固有のパラメータ
            
        Returns
        -------
        cp.ndarray
            処理結果（GPU上）
        """
        pass

class RVIGaussianAlgorithm(TileAlgorithm):
    """GPU上でRidge-Valley Index (RVI)を計算（マルチスケール対応）"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "target_distances": [10, 50, 250],    # 各スケールの実距離（メートル）
            "weights": [0.5, 0.3, 0.2],          # 各スケールの重み
            "pixel_size": 1.0,                   # 1ピクセルあたりの実距離
            "multiscale_mode": True,             # マルチスケールモード
            "sigma": 10.0,                       # シングルスケール時のsigma
        }
    
    def process(self, dem_gpu, **params):
        """GPU上でRVIを計算"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        if p["multiscale_mode"]:
            # マルチスケールRVI
            return self._compute_multiscale_rvi(
                dem_gpu,
                p["target_distances"],
                p["weights"],
                p["pixel_size"]
            )
        else:
            # シングルスケールRVI
            return self._compute_single_scale_rvi(
                dem_gpu,
                p["sigma"]
            )
    
    def _compute_multiscale_rvi(self, dem_gpu, target_distances, weights, pixel_size):
        """マルチスケールRVI計算（バッチ処理対応）"""
        # 実距離をピクセル単位のsigmaに変換
        sigma_values = [max(0.5, dist / pixel_size) for dist in target_distances]
        
        # 結果初期化
        rvi_combined = cp.zeros_like(dem_gpu, dtype=cp.float32)
        
        # バッチ処理でメモリ効率向上
        for sigma, weight in zip(sigma_values, weights):
            # 最適化されたGaussianフィルタ
            dem_blur = cpx_ndimage.gaussian_filter(
                dem_gpu, 
                sigma=sigma, 
                mode="nearest",  # 境界処理
                truncate=4.0     # 計算範囲の制限
            )
            
            # RVI = 元のDEM - スムージングされたDEM
            rvi_scale = dem_gpu - dem_blur
            
            # インプレース演算でメモリ節約
            rvi_combined += weight * rvi_scale
            
            # 不要なメモリを解放
            del dem_blur, rvi_scale
        
        return rvi_combined
    
    def _compute_single_scale_rvi(self, dem_gpu, sigma):
        """シングルスケールRVI計算"""
        # ガウシアンフィルタでスムージング
        dem_blur = cpx_ndimage.gaussian_filter(
            dem_gpu,
            sigma=sigma,
            mode="nearest",
            truncate=4.0
        )
        
        # RVI計算
        rvi = dem_gpu - dem_blur
        
        return rvi


class AtmosphericScatteringAlgorithm(TileAlgorithm):
    """GPU上で大気散乱光効果を計算（マルチスケールTPI版）"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "tpi_radii": [4, 16, 64],        # TPIの計算半径（ピクセル）
            "tpi_weights": [0.3, 0.4, 0.3],  # 各スケールの重み
            "base_ambient": 0.35,            # 基本環境光の強度
            "scattering_variation": 0.4,     # 散乱光の変化幅
            "elevation_bonus": 0.08,         # 相対標高によるボーナス
            "smoothing_sigma": 0.5,          # スムージングのシグマ値
            "sharpening_amount": 0.3,        # シャープニングの強度
            "color_intensity": 0.7,          # 色の強度
            "num_directions": 8,             # 天空可視性計算の方向数
        }
    
    def process(self, dem_gpu, **params):
        """GPU上で大気散乱光を計算"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        # 1. マルチスケールTPIの計算
        multiscale_tpi = self._compute_multiscale_tpi(
            dem_gpu, 
            p["tpi_radii"], 
            p["tpi_weights"]
        )
        
        # 2. 天空可視性の計算
        sky_visibility = self._compute_sky_visibility(
            multiscale_tpi,
            p["num_directions"]
        )
        
        # 3. 散乱光強度の計算
        scattering_intensity = self._compute_scattering_intensity(
            sky_visibility,
            multiscale_tpi,
            p["base_ambient"],
            p["scattering_variation"],
            p["elevation_bonus"],
            p["smoothing_sigma"],
            p["sharpening_amount"]
        )
        
        # 4. カラーマッピング
        result = self._apply_atmospheric_color(
            scattering_intensity,
            p["color_intensity"]
        )
        
        return result
    
    def _compute_multiscale_tpi(self, dem_gpu, radii, weights):
        """マルチスケールTPI（Topographic Position Index）の計算"""
        # DEMの正規化
        dem_min = cp.percentile(dem_gpu, 0.1)
        dem_max = cp.percentile(dem_gpu, 99.9)
        dem_normalized = (dem_gpu - dem_min) / (dem_max - dem_min + 1e-10)
        dem_normalized = cp.clip(dem_normalized, 0, 1)
        
        # 各スケールでのTPI計算
        tpi_layers = []
        for radius in radii:
            # ガウシアンブラーで周囲の平均を計算
            sigma = radius / 3.0
            surrounding_mean = cpx_ndimage.gaussian_filter(
                dem_normalized, 
                sigma=sigma, 
                mode='reflect'
            )
            
            # 相対標高 = 各点の標高 - 周囲の平均標高
            relative_elevation = dem_normalized - surrounding_mean
            tpi_layers.append(relative_elevation)
        
        # 重み付き合成
        multiscale_tpi = cp.zeros_like(dem_normalized, dtype=cp.float32)
        for tpi_layer, weight in zip(tpi_layers, weights):
            multiscale_tpi += tpi_layer * weight
        
        # 正規化（より広い範囲で）
        tpi_min, tpi_max = cp.percentile(multiscale_tpi, [2, 98])
        if tpi_max - tpi_min > 1e-6:
            multiscale_tpi = 2.5 * (multiscale_tpi - tpi_min) / (tpi_max - tpi_min) - 1.25
        else:
            multiscale_tpi = cp.zeros_like(multiscale_tpi)
        
        return cp.clip(multiscale_tpi, -1.25, 1.25)
    
    def _compute_sky_visibility(self, multiscale_tpi, num_directions):
        """8方向の天空可視性を計算"""
        angles = cp.linspace(0, 360, num_directions, endpoint=False)
        altitude = 25  # 固定の高度角
        
        sky_visibility = cp.zeros_like(multiscale_tpi, dtype=cp.float32)
        
        for angle in angles:
            # 各方向のヒルシェード計算
            shade = self._compute_directional_shade(
                multiscale_tpi, 
                float(angle), 
                altitude
            )
            sky_visibility += shade
        
        # 平均化
        sky_visibility = sky_visibility / num_directions
        
        # 正規化
        low_val, high_val = cp.percentile(sky_visibility, [12, 88])
        sky_visibility = (sky_visibility - low_val) / (high_val - low_val + 1e-8)
        
        return cp.clip(sky_visibility, 0.0, 1.0)
    
    def _compute_directional_shade(self, dem, azimuth, altitude):
        """特定方向の陰影を計算（簡易版）"""
        # 太陽の方向ベクトル
        azimuth_rad = cp.radians(azimuth)
        altitude_rad = cp.radians(altitude)
        
        sun_x = cp.sin(azimuth_rad) * cp.cos(altitude_rad)
        sun_y = cp.cos(azimuth_rad) * cp.cos(altitude_rad)
        sun_z = cp.sin(altitude_rad)
        
        # Sobelフィルタで勾配計算
        grad_x = cpx_ndimage.sobel(dem, axis=1, mode='reflect')
        grad_y = cpx_ndimage.sobel(dem, axis=0, mode='reflect')
        
        # 簡易的な法線計算
        normal_z = 1.0 / cp.sqrt(grad_x**2 + grad_y**2 + 1.0)
        normal_x = -grad_x * normal_z
        normal_y = -grad_y * normal_z
        
        # ドット積
        shade = cp.maximum(0.0, sun_x * normal_x + sun_y * normal_y + sun_z * normal_z)
        
        return shade
    
    def _compute_scattering_intensity(self, sky_visibility, multiscale_tpi, 
                                    base_ambient, scattering_variation, 
                                    elevation_bonus, smoothing_sigma, 
                                    sharpening_amount):
        """散乱光強度の計算"""
        # 天空可視性による散乱光の変化
        scattering_var = sky_visibility * scattering_variation
        
        # 基本強度
        scattering_intensity = base_ambient + scattering_var
        
        # 相対標高によるボーナス
        elevation_bonus_map = cp.maximum(multiscale_tpi, 0) * elevation_bonus
        scattering_intensity += elevation_bonus_map
        
        # 軽いスムージング
        if smoothing_sigma > 0:
            scattering_intensity = cpx_ndimage.gaussian_filter(
                scattering_intensity, 
                sigma=smoothing_sigma,
                mode='reflect'
            )
        
        # シャープニング
        if sharpening_amount > 0:
            blurred = cpx_ndimage.gaussian_filter(
                scattering_intensity, 
                sigma=1.0,
                mode='reflect'
            )
            sharpened = scattering_intensity + sharpening_amount * (scattering_intensity - blurred)
            scattering_intensity = sharpened
        
        return cp.clip(scattering_intensity, 0.0, 0.8)
    
    def _apply_atmospheric_color(self, intensity, color_intensity):
        """大気散乱の寒色系カラーマッピング"""
        # 寒色系の色調（青みがかった色）
        cool_color = cp.array([0.3, 0.6, 1.0])
        
        # RGB配列の作成
        result = cp.zeros((*intensity.shape, 3), dtype=cp.float32)
        
        for i in range(3):
            channel_value = intensity * cool_color[i] * color_intensity
            result[:, :, i] = cp.clip(channel_value, 0.0, 1.0)
        
        return result


class CompositeTerrainAlgorithm(TileAlgorithm):
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
        # RVI計算（RVIGaussianAlgorithmを使用）
        rvi_algo = RVIGaussianAlgorithm()
        params = {
            "target_distances": rvi_params["target_distances"],
            "weights": rvi_params["weights"],
            "pixel_size": pixel_size,
            "multiscale_mode": True
        }
        
        rvi = rvi_algo.process(dem_gpu, **params)
        
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


class CurvatureAlgorithm(TileAlgorithm):
    """GPU上で地形の曲率を計算"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "scales": [2, 8, 32],           # 計算スケール（シグマ値）
            "scale_weights": [0.5, 0.3, 0.2],  # 各スケールの重み
            "curvature_type": "both",       # "mean", "gaussian", "both"
            "color_mapping": True,          # カラーマッピングを行うか
            "enhancement_factor": 1.0,      # 曲率の強調係数
        }
    
    def process(self, dem_gpu, **params):
        """GPU上で曲率を計算"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        # DEMの正規化
        dem_normalized = self._normalize_dem(dem_gpu)
        
        # マルチスケール曲率の計算
        curvatures = []
        for scale in p["scales"]:
            curvature = self._compute_curvature_at_scale(
                dem_normalized, 
                scale, 
                p["curvature_type"]
            )
            curvatures.append(curvature)
        
        # スケール統合
        combined = self._combine_scales(curvatures, p["scale_weights"])
        
        # 強調
        if p["enhancement_factor"] != 1.0:
            combined *= p["enhancement_factor"]
        
        # カラーマッピング
        if p["color_mapping"]:
            result = self._apply_curvature_coloring(combined)
        else:
            # グレースケール
            combined = cp.clip((combined + 1) / 2, 0, 1)  # [-1, 1] -> [0, 1]
            result = cp.stack([combined] * 3, axis=-1)
        
        return result
    
    def _normalize_dem(self, dem_gpu):
        """DEMを正規化"""
        dem_min = cp.percentile(dem_gpu, 0.1)
        dem_max = cp.percentile(dem_gpu, 99.9)
        dem_normalized = (dem_gpu - dem_min) / (dem_max - dem_min + 1e-10)
        return dem_normalized
    
    def _compute_curvature_at_scale(self, dem, scale, curvature_type):
        """特定スケールでの曲率計算"""
        # ガウシアンフィルタでスムージング
        smoothed = cpx_ndimage.gaussian_filter(dem, sigma=scale, mode='reflect')
        
        # 1次微分（Sobelフィルタ）
        Ix = cpx_ndimage.sobel(smoothed, axis=1, mode='reflect')
        Iy = cpx_ndimage.sobel(smoothed, axis=0, mode='reflect')
        
        # 2次微分（1次微分の微分）
        Ixx = cpx_ndimage.sobel(Ix, axis=1, mode='reflect')
        Iyy = cpx_ndimage.sobel(Iy, axis=0, mode='reflect')
        Ixy = cpx_ndimage.sobel(Ix, axis=0, mode='reflect')
        
        # 勾配の大きさ
        grad_mag = cp.sqrt(Ix**2 + Iy**2) + 1e-6
        
        if curvature_type == "mean" or curvature_type == "both":
            # 平均曲率
            mean_curvature = (
                Ixx * (1 + Iy**2) - 2 * Ix * Iy * Ixy + Iyy * (1 + Ix**2)
            ) / (2 * grad_mag**3)
            
            # 正規化（tanh使用）
            mean_curvature = cp.tanh(mean_curvature * scale)
        
        if curvature_type == "gaussian" or curvature_type == "both":
            # ガウス曲率
            gaussian_curvature = (Ixx * Iyy - Ixy**2) / (grad_mag**4)
            
            # 正規化
            gaussian_curvature = cp.tanh(gaussian_curvature * scale * scale)
        
        # 結果を返す
        if curvature_type == "mean":
            return mean_curvature
        elif curvature_type == "gaussian":
            return gaussian_curvature
        else:  # both
            return mean_curvature * 0.7 + gaussian_curvature * 0.3
    
    def _combine_scales(self, curvatures, weights):
        """マルチスケールの統合"""
        combined = cp.zeros_like(curvatures[0])
        
        # 重みの正規化
        weights = cp.array(weights) / cp.sum(weights)
        
        for curvature, weight in zip(curvatures, weights):
            combined += curvature * weight
        
        return combined
    
    def _apply_curvature_coloring(self, curvature):
        """曲率に基づくカラーマッピング"""
        # RGB配列の初期化
        result = cp.zeros((*curvature.shape, 3), dtype=cp.float32)
        
        # 凸部（尾根）を赤系、凹部（谷）を青系に
        positive = cp.maximum(curvature, 0)
        negative = cp.maximum(-curvature, 0)
        
        # 色の割り当て
        result[:, :, 0] = positive      # 赤チャンネル（凸部）
        result[:, :, 2] = negative      # 青チャンネル（凹部）
        result[:, :, 1] = 0.5 - cp.abs(curvature) * 0.5  # 緑チャンネル
        
        return cp.clip(result, 0, 1)


class FrequencyEnhancementAlgorithm(TileAlgorithm):
    """GPU上で周波数領域での地形強調を実行"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "low_cutoff": 10,           # 低周波カットオフ（ピクセル）
            "high_cutoff": 100,         # 高周波カットオフ（ピクセル）
            "enhancement_factor": 2.0,   # 強調係数
            "transition_width": 5,       # 遷移幅（スムーズなフィルタのため）
            "apply_hillshade": True,    # 結果にヒルシェードを適用
            "preserve_dc": True,        # DC成分（平均値）を保持
        }
    
    def process(self, dem_gpu, **params):
        """GPU上で周波数強調を実行"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        # DEMの正規化
        dem_normalized = self._normalize_dem(dem_gpu)
        
        # FFT実行
        fft_dem = cpx_fft.fft2(dem_normalized)
        fft_shifted = cpx_fft.fftshift(fft_dem)
        
        # 周波数フィルタの作成
        freq_filter = self._create_frequency_filter(
            dem_normalized.shape,
            p["low_cutoff"],
            p["high_cutoff"],
            p["enhancement_factor"],
            p["transition_width"]
        )
        
        # DC成分の保持
        if p["preserve_dc"]:
            dc_component = fft_shifted[fft_shifted.shape[0]//2, fft_shifted.shape[1]//2]
        
        # フィルタ適用
        fft_filtered = fft_shifted * freq_filter
        
        # DC成分の復元
        if p["preserve_dc"]:
            fft_filtered[fft_filtered.shape[0]//2, fft_filtered.shape[1]//2] = dc_component
        
        # 逆FFT
        fft_ishifted = cpx_fft.ifftshift(fft_filtered)
        enhanced_dem = cp.real(cpx_fft.ifft2(fft_ishifted))
        
        # 再正規化
        enhanced_dem = self._renormalize(enhanced_dem)
        
        # ヒルシェード適用
        if p["apply_hillshade"]:
            # 強調されたDEMをヒルシェード表示
            hillshade_algo = HillshadeAlgorithm()
            # 適切なz_factorのためにDEMを仮想的にスケール
            scaled_dem = enhanced_dem * 1000  # 仮想的な標高スケール
            result = hillshade_algo.process(
                scaled_dem, 
                azimuth=315, 
                altitude=45,
                color_mode="grayscale"
            )
            
            # グレースケールの場合はRGBに変換
            if len(result.shape) == 2:
                result = cp.stack([result] * 3, axis=-1)
        else:
            # そのままRGBとして返す
            result = cp.stack([enhanced_dem] * 3, axis=-1)
        
        return result
    
    def _normalize_dem(self, dem_gpu):
        """DEMを正規化"""
        dem_min = cp.percentile(dem_gpu, 0.1)
        dem_max = cp.percentile(dem_gpu, 99.9)
        dem_normalized = (dem_gpu - dem_min) / (dem_max - dem_min + 1e-10)
        return dem_normalized
    
    def _create_frequency_filter(self, shape, low_cutoff, high_cutoff, 
                                enhancement_factor, transition_width):
        """バンドパスフィルタの作成（スムーズな遷移付き）"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # メッシュグリッドで各点の周波数（距離）を計算
        y = cp.arange(rows) - crow
        x = cp.arange(cols) - ccol
        Y, X = cp.meshgrid(y, x, indexing='ij')
        distance = cp.sqrt(X**2 + Y**2)
        
        # スムーズなバンドパスフィルタ
        filter_mask = cp.ones_like(distance, dtype=cp.float32)
        
        # 低周波側の遷移
        low_transition = cp.clip(
            (distance - (low_cutoff - transition_width)) / (2 * transition_width),
            0, 1
        )
        
        # 高周波側の遷移
        high_transition = cp.clip(
            ((high_cutoff + transition_width) - distance) / (2 * transition_width),
            0, 1
        )
        
        # バンド内は強調、外は減衰
        in_band = (distance >= low_cutoff) & (distance <= high_cutoff)
        transition_band_low = (distance >= (low_cutoff - transition_width)) & (distance < low_cutoff)
        transition_band_high = (distance > high_cutoff) & (distance <= (high_cutoff + transition_width))
        
        # フィルタ値の設定
        filter_mask[in_band] = enhancement_factor
        filter_mask[transition_band_low] = 1.0 + (enhancement_factor - 1.0) * low_transition[transition_band_low]
        filter_mask[transition_band_high] = 1.0 + (enhancement_factor - 1.0) * high_transition[transition_band_high]
        filter_mask[distance < (low_cutoff - transition_width)] = 1.0
        filter_mask[distance > (high_cutoff + transition_width)] = 0.5
        
        return filter_mask
    
    def _renormalize(self, data):
        """データの再正規化"""
        data_min = cp.min(data)
        data_max = cp.max(data)
        if data_max - data_min > 1e-6:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = data
        return cp.clip(normalized, 0, 1)


class HillshadeAlgorithm(TileAlgorithm):
    """GPU上でHillshade（陰影起伏図）を計算"""
    
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        return {
            "azimuth": 315.0,       # 太陽の方位角（度）
            "altitude": 45.0,       # 太陽の高度角（度）
            "z_factor": None,       # 垂直誇張率（Noneの場合自動計算）
            "contrast_enhance": True,  # コントラスト強化を行うか
            "gamma": 1.8,           # ガンマ補正値
            "color_mode": "warm",   # カラーモード: "warm", "cool", "grayscale"
        }
    
    def process(self, dem_gpu, **params):
        """GPU上でHillshadeを計算"""
        # パラメータ取得
        p = self.get_default_params()
        p.update(params)
        
        azimuth = p["azimuth"]
        altitude = p["altitude"]
        z_factor = p["z_factor"]
        contrast_enhance = p["contrast_enhance"]
        gamma = p["gamma"]
        color_mode = p["color_mode"]
        
        # z_factorの自動計算
        if z_factor is None:
            # pixel_sizeはparams経由で渡される想定
            pixel_size = params.get("pixel_size", 1.0)
            dem_range = cp.percentile(dem_gpu, 95) - cp.percentile(dem_gpu, 5)
            z_factor = pixel_size * 5.0 / cp.maximum(dem_range, 1.0)
        
        # Hillshade計算
        hillshade = self._compute_hillshade_gpu(dem_gpu, azimuth, altitude, z_factor)
        
        # コントラスト強化
        if contrast_enhance:
            hillshade = self._enhance_contrast(hillshade, gamma)
        
        # カラーマッピング
        result = self._apply_color_mapping(hillshade, color_mode)
        
        return result
    
    def _compute_hillshade_gpu(self, dem_gpu, azimuth, altitude, z_factor):
        """GPU上でHillshadeを計算（8方向の隣接セルを使用）"""
        # 太陽の方向ベクトル
        azimuth_rad = cp.radians(azimuth)
        altitude_rad = cp.radians(altitude)
        
        sun_x = cp.sin(azimuth_rad) * cp.cos(altitude_rad)
        sun_y = cp.cos(azimuth_rad) * cp.cos(altitude_rad)
        sun_z = cp.sin(altitude_rad)
        
        # パディング（境界処理のため）
        dem_padded = cp.pad(dem_gpu, pad_width=1, mode='edge')
        
        # 8方向の勾配計算（Sobelフィルタ風）
        kernel_x = cp.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=cp.float32) / 8.0
        
        kernel_y = cp.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=cp.float32) / 8.0
        
        # 勾配計算
        dzdx = cpx_ndimage.convolve(dem_padded, kernel_x)[1:-1, 1:-1] * z_factor
        dzdy = cpx_ndimage.convolve(dem_padded, kernel_y)[1:-1, 1:-1] * z_factor
        
        # 法線ベクトルの計算
        normal_z = 1.0 / cp.sqrt(dzdx**2 + dzdy**2 + 1.0)
        normal_x = -dzdx * normal_z
        normal_y = -dzdy * normal_z
        
        # ドット積による陰影計算
        hillshade = cp.maximum(0.0, sun_x * normal_x + sun_y * normal_y + sun_z * normal_z)
        
        return hillshade
    
    def _enhance_contrast(self, hillshade, gamma):
        """コントラスト強化処理"""
        # ヒストグラム調整
        low_val, high_val = cp.percentile(hillshade, [2, 98])
        hillshade = (hillshade - low_val) / (high_val - low_val + 1e-8)
        hillshade = cp.clip(hillshade, 0.0, 1.0)
        
        # ガンマ補正
        hillshade = cp.power(hillshade, gamma)
        
        # S字カーブでさらにコントラスト強化
        hillshade = 0.5 * (cp.tanh(4 * (hillshade - 0.5)) + 1)
        
        return hillshade
    
    def _apply_color_mapping(self, hillshade, color_mode):
        """カラーマッピングを適用"""
        if color_mode == "grayscale":
            # グレースケールでもRGB形式で返す
            return cp.stack([hillshade] * 3, axis=-1)
        
        # RGB配列を作成
        result = cp.zeros((*hillshade.shape, 3), dtype=cp.float32)
        
        if color_mode == "warm":
            # 暖色系（オレンジ〜茶色）
            lit_color = cp.array([1.0, 0.75, 0.4])      # 明るい暖色
            shadow_color = cp.array([0.1, 0.08, 0.15])  # 深い影色
        elif color_mode == "cool":
            # 寒色系（青〜紫）
            lit_color = cp.array([0.7, 0.85, 1.0])      # 明るい寒色
            shadow_color = cp.array([0.05, 0.1, 0.2])   # 深い青影
        else:
            # デフォルトは暖色
            lit_color = cp.array([1.0, 0.75, 0.4])
            shadow_color = cp.array([0.1, 0.08, 0.15])
        
        # 照明強度に応じた色の補間
        for i in range(3):
            result[:, :, i] = (
                hillshade * lit_color[i] + 
                (1.0 - hillshade) * shadow_color[i]
            )
        
        return result


class VisualSaliencyAlgorithm(TileAlgorithm):
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