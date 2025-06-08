"""
FujiShaderGPU/algorithms/frequency_enhancement.py
フーリエ変換による特定周波数の強調をGPUで高速計算
"""
import cupy as cp
import cupyx.scipy.fft as cpx_fft
from .base import BaseAlgorithm
from .hillshade import HillshadeAlgorithm


class FrequencyEnhancementAlgorithm(BaseAlgorithm):
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


# 方向性フィルタリング用の追加クラス
class DirectionalFrequencyFilter(BaseAlgorithm):
    """特定方向の周波数成分を強調"""
    
    def get_default_params(self):
        return {
            "direction": 45,         # 強調する方向（度）
            "direction_width": 30,   # 方向の幅（度）
            "frequency_range": (10, 100),  # 周波数範囲
            "enhancement": 3.0,      # 強調度
        }
    
    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        
        # FFT
        fft_dem = cpx_fft.fft2(dem_gpu)
        fft_shifted = cpx_fft.fftshift(fft_dem)
        
        # 方向性フィルタ作成
        rows, cols = dem_gpu.shape
        crow, ccol = rows // 2, cols // 2
        
        y = cp.arange(rows) - crow
        x = cp.arange(cols) - ccol
        Y, X = cp.meshgrid(y, x, indexing='ij')
        
        # 角度と距離
        angle = cp.arctan2(Y, X) * 180 / cp.pi
        distance = cp.sqrt(X**2 + Y**2)
        
        # 方向性マスク
        target_angle = p["direction"]
        angle_width = p["direction_width"]
        
        # 角度差の計算（-180 to 180の範囲を考慮）
        angle_diff = cp.abs(angle - target_angle)
        angle_diff = cp.minimum(angle_diff, 360 - angle_diff)
        
        # 方向性フィルタ
        angle_mask = cp.exp(-(angle_diff / angle_width)**2)
        
        # 周波数範囲マスク
        freq_mask = (distance >= p["frequency_range"][0]) & (distance <= p["frequency_range"][1])
        
        # 最終フィルタ
        final_filter = 1.0 + (p["enhancement"] - 1.0) * angle_mask * freq_mask.astype(cp.float32)
        
        # 適用
        fft_filtered = fft_shifted * final_filter
        
        # 逆FFT
        enhanced = cp.real(cpx_fft.ifft2(cpx_fft.ifftshift(fft_filtered)))
        
        # 正規化してRGB化
        enhanced = (enhanced - cp.min(enhanced)) / (cp.max(enhanced) - cp.min(enhanced) + 1e-6)
        return cp.stack([enhanced] * 3, axis=-1)
