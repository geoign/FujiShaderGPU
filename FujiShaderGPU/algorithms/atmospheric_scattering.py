"""
FujiShaderGPU/algorithms/atmospheric_scattering.py
寒色の大気散乱光による谷間・盆地の照らし出し効果をGPUで高速計算
"""
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from .base import BaseAlgorithm
from .utils import calculate_padding


class AtmosphericScatteringAlgorithm(BaseAlgorithm):
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


# 高速化用のCUDAカーネル（オプション）
_tpi_kernel = cp.RawKernel(r'''
extern "C" __global__
void tpi_kernel(
    const float* dem,
    const float* surrounding_mean,
    float* tpi,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        tpi[idx] = dem[idx] - surrounding_mean[idx];
    }
}
''', 'tpi_kernel')
