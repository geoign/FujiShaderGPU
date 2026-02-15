"""
FujiShaderGPU/algorithms/_impl_npr_edges.py

NPR Edges (非写実的レンダリング輪郭線) アルゴリズム実装。
dask_shared.py からの分離モジュール (Phase 2)。
"""
from __future__ import annotations
import cupy as cp
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, convolve, binary_dilation

from ._base import Constants, DaskAlgorithm, classify_resolution
from ._nan_utils import restore_nan


def compute_npr_edges_block(block: cp.ndarray, *, edge_sigma: float = 1.0,
                          threshold_low: float = 0.1, threshold_high: float = 0.3,
                          pixel_size: float = 1.0) -> cp.ndarray:
    """NPRスタイルの輪郭線抽出（簡略版v2）"""
    nan_mask = cp.isnan(block)
    resolution_class = classify_resolution(pixel_size)

    # 解像度に応じたスムージング
    if resolution_class in ['ultra_high', 'very_high']:
        adaptive_sigma = 0.5
    elif resolution_class in ['high', 'medium']:
        adaptive_sigma = 1.0
    elif resolution_class == 'low':
        adaptive_sigma = 0.5
    else:
        adaptive_sigma = 0.3

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

    # Sobelフィルタを使用した勾配計算
    sobel_x = cp.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=cp.float32) / 8.0
    sobel_y = cp.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=cp.float32) / 8.0

    dx = convolve(smoothed, sobel_x, mode='nearest')
    dy = convolve(smoothed, sobel_y, mode='nearest')

    gradient_mag = cp.sqrt(dx**2 + dy**2)

    # 解像度適応型の勾配強調
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        local_max = maximum_filter(smoothed, size=3, mode='nearest')
        local_min = minimum_filter(smoothed, size=3, mode='nearest')
        local_range = local_max - local_min
        gradient_mag = cp.maximum(gradient_mag, local_range * 0.3)

    gradient_dir = cp.arctan2(dy, dx)

    # 適応的な閾値設定
    valid_grad = gradient_mag[~nan_mask] if nan_mask.any() else gradient_mag.ravel()
    if len(valid_grad) > 0:
        grad_std = cp.std(valid_grad)
        grad_mean = cp.mean(valid_grad)

        if resolution_class in ['low', 'very_low', 'ultra_low']:
            base_threshold = grad_mean
            threshold_range = grad_std * 1.5
        else:
            base_threshold = cp.percentile(valid_grad, 50)
            threshold_range = cp.percentile(valid_grad, 90) - base_threshold

        actual_threshold_low = base_threshold + threshold_range * threshold_low * 0.5
        actual_threshold_high = base_threshold + threshold_range * threshold_high

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

    # 8方向での非最大値抑制
    shifted_pos = cp.roll(gradient_mag, 1, axis=1)
    shifted_neg = cp.roll(gradient_mag, -1, axis=1)
    mask = ((angle < 22.5) | (angle >= 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    shifted_pos = cp.roll(cp.roll(gradient_mag, 1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, -1, axis=0), 1, axis=1)
    mask = ((angle >= 22.5) & (angle < 67.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    shifted_pos = cp.roll(gradient_mag, 1, axis=0)
    shifted_neg = cp.roll(gradient_mag, -1, axis=0)
    mask = ((angle >= 67.5) & (angle < 112.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    shifted_pos = cp.roll(cp.roll(gradient_mag, -1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, 1, axis=0), 1, axis=1)
    mask = ((angle >= 112.5) & (angle < 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)

    # ダブルスレッショルド
    strong = nms > actual_threshold_high
    weak = (nms > actual_threshold_low) & (nms <= actual_threshold_high)

    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5

    # ヒステリシス処理
    for _ in range(3):
        dilated = cp.maximum(
            cp.maximum(cp.roll(edges, 1, axis=0), cp.roll(edges, -1, axis=0)),
            cp.maximum(cp.roll(edges, 1, axis=1), cp.roll(edges, -1, axis=1))
        )
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, 1, axis=0), 1, axis=1))
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, -1, axis=0), -1, axis=1))
        edges = cp.where(weak & (dilated > 0.5), 1.0, edges)

    # 後処理：解像度に応じたエッジの太さ調整
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        structure = cp.ones((3, 3))
        edges_binary = edges > 0.5
        edges_dilated = binary_dilation(edges_binary, structure=structure).astype(cp.float32)
        edges = cp.where(edges_dilated, cp.maximum(edges, 0.8), edges)

    edges = edges * 0.8
    result = 1.0 - edges
    result = cp.clip(result, 0.2, 1.0)
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR輪郭線アルゴリズム（簡略版）"""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        edge_sigma = params.get('edge_sigma', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        threshold_low = params.get('threshold_low', 0.1)
        threshold_high = params.get('threshold_high', 0.3)

        depth = 3
        if edge_sigma != 1.0:
            depth = max(depth, int(edge_sigma * 4 + 2))

        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            edge_sigma=edge_sigma,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            pixel_size=pixel_size,
        )

    def get_default_params(self) -> dict:
        return {
            'edge_sigma': 1.0,
            'threshold_low': 0.1,
            'threshold_high': 0.3,
            'pixel_size': 1.0
        }


__all__ = [
    "compute_npr_edges_block",
    "NPREdgesAlgorithm",
]
