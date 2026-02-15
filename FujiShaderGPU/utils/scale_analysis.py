"""
FujiShaderGPU/utils/scale_analysis.py
地形スケール自動分析ユーティリティ
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Tuple, List
import logging

# scipy は CPU フォールバック用（オプション）
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scipy が利用できません。一部の CPU フォールバックに切り替えます。"
    )


def _analyze_scale_variances_scipy_fast(
    dem_2d: np.ndarray,
    candidate_distances: List[float],
    pixel_size: float,
) -> List[float]:
    """
    Scipy高速CPU分析（2D空間構造を維持した状態で処理）
    """
    if not SCIPY_AVAILABLE:
        return [1.0] * len(candidate_distances)

    # NaN領域を平均値で一時的に埋めてフィルタ処理
    nan_mask = np.isnan(dem_2d)
    if nan_mask.any():
        fill_val = float(np.nanmean(dem_2d))
        dem_work = np.where(nan_mask, fill_val, dem_2d)
    else:
        dem_work = dem_2d

    variances = []
    for distance in candidate_distances:
        sigma = max(0.5, distance / pixel_size)
        blurred = gaussian_filter(dem_work, sigma=sigma, mode="nearest", truncate=4.0)
        # 元データとの差分を取り、NaN位置は結果もNaNにする
        rvi = dem_2d - blurred
        variance = float(np.nanvar(rvi))
        variances.append(variance)

    return variances


def analyze_terrain_scales(
    input_cog_path: str,
    pixel_size: float,
    sample_size: int = 8192,
) -> Tuple[List[float], List[float]]:
    """
    地形スケール分析（超高速化版）
    """
    logger = logging.getLogger(__name__)
    logger.info("=== 地形スケール超高速分析開始 ===")

    try:
        with rasterio.open(input_cog_path) as src:
            width = src.width
            height = src.height

            # より大きなサンプルで精度向上
            sample_x = max(0, (width - sample_size) // 2)
            sample_y = max(0, (height - sample_size) // 2)
            sample_w = min(sample_size, width - sample_x)
            sample_h = min(sample_size, height - sample_y)

            logger.debug(
                "サンプル範囲: x=%d, y=%d, w=%d, h=%d",
                sample_x, sample_y, sample_w, sample_h,
            )

            window = Window(sample_x, sample_y, sample_w, sample_h)
            dem_sample = src.read(1, window=window, out_dtype=np.float32)

            # NoData処理: 空間構造を維持するためNaNで置換（1Dへの縮退を防止）
            nodata = src.nodata
            if nodata is not None:
                valid_mask = dem_sample != nodata
                if np.sum(valid_mask) < dem_sample.size * 0.5:
                    return _get_default_scales()
                dem_sample = np.where(valid_mask, dem_sample, np.nan).astype(
                    np.float32
                )

            # 拡張候補スケール
            candidate_distances = [
                pixel_size * 2, pixel_size * 4, pixel_size * 8,
                pixel_size * 16, pixel_size * 32, pixel_size * 64,
                pixel_size * 128, pixel_size * 256, pixel_size * 512
            ]

            # GPU高速分析（2Dデータをそのまま渡す）
            scale_variances = _analyze_scale_variances_ultra_fast(
                dem_sample, candidate_distances, pixel_size
            )
            optimal_scales, optimal_weights = _select_optimal_scales_enhanced(
                candidate_distances, scale_variances
            )

            logger.info(
                "[OK] スケール分析完了: %dスケール選択", len(optimal_scales)
            )
            return optimal_scales, optimal_weights

    except Exception as e:
        logger.info("地形分析エラー: %s", e)
        return _get_default_scales()


def _analyze_scale_variances_ultra_fast(
    dem_2d: np.ndarray,
    candidate_distances: List[float],
    pixel_size: float,
) -> List[float]:
    """
    GPU バッチ処理による超高速スケール分析
    入力は2D配列（空間構造を維持）を前提とする。
    """
    logger = logging.getLogger(__name__)

    try:
        # CuPy 遅延インポート（GPU未搭載環境でもモジュール読み込みを許容）
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_ndimage

        dem_gpu = cp.asarray(dem_2d, dtype=cp.float32)

        # NaN領域を平均値で一時的に埋めてフィルタ処理
        nan_mask = cp.isnan(dem_gpu)
        if nan_mask.any():
            fill_val = cp.nanmean(dem_gpu)
            dem_filled = cp.where(nan_mask, fill_val, dem_gpu)
        else:
            dem_filled = dem_gpu

        variances = []
        sigma_values = [max(0.5, dist / pixel_size) for dist in candidate_distances]

        for sigma in sigma_values:
            blurred = cpx_ndimage.gaussian_filter(
                dem_filled, sigma=sigma, mode="nearest", truncate=4.0
            )
            rvi = dem_gpu - blurred
            # NaN箇所を除外した分散を計算
            variance = float(cp.nanvar(rvi))
            variances.append(variance)
            del blurred, rvi

        del dem_gpu, dem_filled
        return variances

    except Exception as e:
        logger.info("GPU分析失敗、scipy高速CPUにフォールバック: %s", e)
        return _analyze_scale_variances_scipy_fast(dem_2d, candidate_distances, pixel_size)


def _select_optimal_scales_enhanced(
    candidate_distances: List[float],
    variances: List[float],
) -> Tuple[List[float], List[float]]:
    """
    改良された最適スケール選択
    """
    variances = np.array(variances)
    if np.max(variances) > 0:
        variances = variances / np.max(variances)

    # より洗練された選択アルゴリズム
    n_scales = min(5, len(candidate_distances))  # 最大5スケール

    if len(variances) >= n_scales:
        # 分散の高い上位スケールを選択
        top_indices = np.argsort(variances)[-n_scales:]
        top_indices = np.sort(top_indices)
    else:
        top_indices = np.arange(len(variances))

    optimal_distances = [candidate_distances[i] for i in top_indices]
    optimal_variances = [variances[i] for i in top_indices]

    # 指数的重み付け（小スケール重視）
    if np.sum(optimal_variances) > 0:
        weights_raw = np.array(optimal_variances)
        # 距離に反比例する重み付けを追加
        distance_weights = 1.0 / np.array(optimal_distances)
        combined_weights = weights_raw * distance_weights
        optimal_weights = (combined_weights / np.sum(combined_weights)).tolist()
    else:
        optimal_weights = [1.0 / len(optimal_distances)] * len(optimal_distances)

    return optimal_distances, optimal_weights


def _get_default_scales() -> Tuple[List[float], List[float]]:
    """
    改良されたデフォルトスケール
    """
    default_distances = [2.5, 10.0, 40.0, 160.0, 320.0]
    default_weights = [0.4, 0.25, 0.2, 0.1, 0.05]
    return default_distances, default_weights
