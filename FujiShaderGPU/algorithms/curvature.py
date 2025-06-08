"""
FujiShaderGPU/algorithms/curvature.py
多重スケール曲率解析をGPUで高速計算
"""
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from .base import BaseAlgorithm


class CurvatureAlgorithm(BaseAlgorithm):
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


# プロファイル曲率と平面曲率を計算するカーネル
_profile_plan_curvature_kernel = cp.RawKernel(r'''
extern "C" __global__
void profile_plan_curvature(
    const float* Ix, const float* Iy,
    const float* Ixx, const float* Iyy, const float* Ixy,
    float* profile_curv, float* plan_curv,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        float fx = Ix[idx];
        float fy = Iy[idx];
        float fxx = Ixx[idx];
        float fyy = Iyy[idx];
        float fxy = Ixy[idx];
        
        float grad_mag2 = fx*fx + fy*fy;
        float grad_mag = sqrtf(grad_mag2) + 1e-6f;
        
        // プロファイル曲率（流下方向の曲率）
        if (grad_mag2 > 1e-6f) {
            profile_curv[idx] = -(fx*fx*fxx + 2*fx*fy*fxy + fy*fy*fyy) / (grad_mag2 * grad_mag);
        } else {
            profile_curv[idx] = 0.0f;
        }
        
        // 平面曲率（等高線方向の曲率）
        if (grad_mag2 > 1e-6f) {
            plan_curv[idx] = (fy*fy*fxx - 2*fx*fy*fxy + fx*fx*fyy) / (grad_mag * grad_mag2);
        } else {
            plan_curv[idx] = 0.0f;
        }
    }
}
''', 'profile_plan_curvature')
