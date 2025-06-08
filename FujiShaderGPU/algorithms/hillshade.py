"""
FujiShaderGPU/algorithms/hillshade.py
暖色の直射日光によるHillshade的な陰影をGPUで高速計算
"""
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from .base import BaseAlgorithm
from .utils import calculate_padding


class HillshadeAlgorithm(BaseAlgorithm):
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
            return hillshade
        
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


# カスタムCUDAカーネル（さらなる高速化が必要な場合）
_hillshade_kernel = cp.RawKernel(r'''
extern "C" __global__
void hillshade_kernel(
    const float* dem,
    float* hillshade,
    int rows, int cols,
    float sun_x, float sun_y, float sun_z,
    float z_factor
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 境界チェック（内側のピクセルのみ処理）
    if (i >= 1 && i < rows - 1 && j >= 1 && j < cols - 1) {
        int idx = i * cols + j;
        
        // 8方向の勾配計算
        float dzdx = (
            (dem[(i-1)*cols + (j+1)] + 2*dem[i*cols + (j+1)] + dem[(i+1)*cols + (j+1)] -
             dem[(i-1)*cols + (j-1)] - 2*dem[i*cols + (j-1)] - dem[(i+1)*cols + (j-1)])
            / 8.0f
        ) * z_factor;
        
        float dzdy = (
            (dem[(i+1)*cols + (j-1)] + 2*dem[(i+1)*cols + j] + dem[(i+1)*cols + (j+1)] -
             dem[(i-1)*cols + (j-1)] - 2*dem[(i-1)*cols + j] - dem[(i-1)*cols + (j+1)])
            / 8.0f
        ) * z_factor;
        
        // 法線ベクトル
        float norm = 1.0f / sqrtf(dzdx*dzdx + dzdy*dzdy + 1.0f);
        float normal_x = -dzdx * norm;
        float normal_y = -dzdy * norm;
        float normal_z = norm;
        
        // ドット積
        float dot = sun_x * normal_x + sun_y * normal_y + sun_z * normal_z;
        hillshade[idx] = fmaxf(0.0f, dot);
    }
}
''', 'hillshade_kernel')
