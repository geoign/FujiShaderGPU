"""
FujiShaderGPU/core/tile_processor.py
タイルベース地形解析処理のコア実装（Windows/macOS向け）
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.gpu_memory import gpu_memory_pool
from ..config.system_config import get_gpu_config
from ..io.raster_info import detect_pixel_size_from_cog
from ..utils.types import TileResult
from ..utils.scale_analysis import analyze_terrain_scales, _get_default_scales
from ..utils.nodata_handler import _handle_nodata_ultra_fast
from ..io.cog_builder import _build_vrt_and_cog_ultra_fast
from ..io.cog_validator import _validate_cog_for_qgis
from ..algorithms.tile_algorithms import RVIGaussianAlgorithm
import os
import math
import glob
import shutil
import rasterio
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from rasterio.windows import Window
from rasterio.transform import Affine
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# デフォルトで利用可能なアルゴリズム（Windows/macOS）
DEFAULT_ALGORITHMS = {
    "rvi_gaussian": "RVIGaussian",  # 特殊ケース：関数として実装
    "hillshade": "HillshadeAlgorithm",
    "atmospheric_scattering": "AtmosphericScatteringAlgorithm",
    "composite_terrain": "CompositeTerrainAlgorithm",
    "curvature": "CurvatureAlgorithm",
    "frequency_enhancement": "FrequencyEnhancementAlgorithm",
    "visual_saliency": "VisualSaliencyAlgorithm"
}

def _load_algorithm(name: str):
    """アルゴリズムを動的にロード（クラスベース）"""
    # RVI Gaussianは特殊ケース（関数として実装）
    if name == "rvi_gaussian":
        # 関数をクラス風にラップ
        class RVIWrapper:
            def process(self, dem_gpu, **params):
                if params.get('multiscale_mode', True):
                    rvi_algo = RVIGaussianAlgorithm()
                    return rvi_algo._compute_multiscale_rvi(
                        dem_gpu,
                        params.get('target_distances', [10, 50, 250]),
                        params.get('weights', [0.5, 0.3, 0.2]),
                        params.get('pixel_size', 1.0)
                    )
                else:
                    # シングルスケール
                    sigma = params.get('sigma', 10.0)
                    dem_blur = cpx_ndimage.gaussian_filter(
                        dem_gpu, sigma=sigma, mode="nearest", truncate=4.0
                    )
                    return dem_gpu - dem_blur
        return RVIWrapper()
    
    # その他のアルゴリズムはtile_algorithmsから読み込み
    if name in DEFAULT_ALGORITHMS:
        try:
            # tile_algorithmsモジュールから全てのアルゴリズムをインポート
            from ..algorithms.tile_algorithms import (
                HillshadeAlgorithm,
                AtmosphericScatteringAlgorithm,
                CompositeTerrainAlgorithm,
                CurvatureAlgorithm,
                FrequencyEnhancementAlgorithm,
                VisualSaliencyAlgorithm
            )
            
            # クラス名でアルゴリズムを取得
            algorithm_class_name = DEFAULT_ALGORITHMS[name]
            algorithm_class = locals().get(algorithm_class_name)
            
            if algorithm_class:
                return algorithm_class()
            
        except ImportError as e:
            logger.warning(f"Failed to load algorithm {name}: {e}")
    
    raise ValueError(f"Algorithm {name} not found or not available on this platform")

def process_single_tile(
    input_cog_path: str,
    tile_info: Tuple[int, int, int, int, int, int, int, int, int, int],
    tmp_tile_dir: str,
    algorithm: str,
    sigma: float,
    nodata: Optional[float],
    src_transform: Affine,
    src_crs,
    profile: dict,
    nodata_threshold: float = 1.0,
    vram_monitor: bool = False,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5,
    target_distances: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    **algo_params
) -> TileResult:
    """
    単一タイル処理（アルゴリズム選択対応版）
    """
    ty, tx, core_x, core_y, core_w, core_h, win_x_off, win_y_off, win_w, win_h = tile_info
    
    try:
        with gpu_memory_pool():
            # メモリマップド読み込み（最適化）
            with rasterio.open(input_cog_path, 'r') as src:
                window = Window(win_x_off, win_y_off, win_w, win_h)
                
                # 高速読み込み（dtype指定で変換コスト削減）
                dem_tile = src.read(1, window=window, out_dtype=np.float32)
                
                # NoData処理とスキップ判定（最適化）
                mask_nodata = None
                if nodata is not None:
                    mask_nodata = (dem_tile == nodata)
                    nodata_ratio = np.count_nonzero(mask_nodata) / mask_nodata.size
                    
                    if nodata_ratio >= nodata_threshold:
                        return TileResult(
                            ty, tx, False,
                            skipped_reason=f"NoDataが{nodata_ratio:.1%}を占める（閾値:{nodata_threshold:.1%}）"
                        )
                    
                    if nodata_ratio > 0.8:
                        logger.warning(f"タイル({ty}, {tx}) NoData率が高いです: {nodata_ratio:.1%}")
                    
                    # 超高速NoData処理
                    dem_tile_processed = _handle_nodata_ultra_fast(dem_tile, mask_nodata)
                else:
                    dem_tile_processed = dem_tile

                # GPU転送（最適化）
                dem_gpu = cp.asarray(dem_tile_processed, dtype=cp.float32)

                # アルゴリズム選択と実行
                algo_instance = _load_algorithm(algorithm)
                
                if algorithm == "rvi_gaussian":
                    # RVI計算（特殊処理）
                    params = {
                        'multiscale_mode': multiscale_mode,
                        'target_distances': target_distances,
                        'weights': weights,
                        'pixel_size': pixel_size,
                        'sigma': sigma,
                    }
                    result_gpu = algo_instance.process(dem_gpu, **params)
                else:
                    # その他のアルゴリズム（クラスベース）
                    # アルゴリズムに渡すパラメータを準備
                    params = {
                        'sigma': sigma,
                        'pixel_size': pixel_size,
                        **algo_params
                    }
                    result_gpu = algo_instance.process(dem_gpu, **params)

                # NoData復元（必要時のみ）
                if mask_nodata is not None:
                    result_gpu[cp.asarray(mask_nodata)] = cp.float32(nodata or 0)

                # CPU転送（最適化）
                result_tile = cp.asnumpy(result_gpu)
                del dem_gpu, result_gpu

                # コア領域抽出
                core_x_in_win = core_x - win_x_off
                core_y_in_win = core_y - win_y_off
                result_core = result_tile[
                    core_y_in_win : core_y_in_win + core_h,
                    core_x_in_win : core_x_in_win + core_w,
                ]

                # 最適化されたタイルプロファイル
                core_transform = rasterio.windows.transform(
                    Window(core_x, core_y, core_w, core_h), src_transform
                )

                tile_profile = profile.copy()
                tile_profile.update({
                    "driver": "GTiff",
                    "height": core_h,
                    "width": core_w,
                    "count": 1,
                    "dtype": np.float32,
                    "crs": src_crs,
                    "transform": core_transform,
                    "compress": "ZSTD",
                    "zlevel": 1,
                    "tiled": True,
                    "blockxsize": min(512, core_w),
                    "blockysize": min(512, core_h),
                    "BIGTIFF": "YES",
                    "nodata": nodata,
                    "NUM_THREADS": "ALL_CPUS"
                })

                tile_filename = os.path.join(
                    tmp_tile_dir, f"tile_{ty:03d}_{tx:03d}.tif"
                )

                # 高速書き込み
                with rasterio.open(tile_filename, "w", **tile_profile) as dst:
                    dst.write(result_core, 1)

                return TileResult(ty, tx, True, tile_filename)
                
    except Exception as e:
        logger.error(f"Tile ({ty}, {tx}) processing failed: {e}")
        return TileResult(ty, tx, False, error_message=str(e))


def process_dem_tiles(
    input_cog_path: str,
    output_cog_path: str,
    tmp_tile_dir: str = "tiles_tmp",
    algorithm: str = "rvi_gaussian",  # アルゴリズム選択を追加
    tile_size: Optional[int] = None,
    padding: Optional[int] = None,
    sigma: float = 10.0,
    max_workers: Optional[int] = None,
    nodata_threshold: float = 1.0,
    gpu_type: str = "auto",
    multiscale_mode: bool = True,
    pixel_size: Optional[float] = None,
    auto_scale_analysis: bool = True,
    cog_only: bool = False,
    **algo_params  # アルゴリズム固有のパラメータ
):
    """
    タイルベースDEM処理メイン関数（アルゴリズム選択対応版）
    """
    # COG生成のみの場合
    if cog_only:
        resume_cog_generation(
            tmp_tile_dir, 
            output_cog_path, 
            gpu_type, 
            sigma, 
            multiscale_mode, 
            pixel_size or 0.5
        )
        return
    
    logger.info(f"=== DEM→{algorithm.upper()}処理開始 ===")
    
    # ピクセルサイズ検出
    if pixel_size is None:
        pixel_size = detect_pixel_size_from_cog(input_cog_path)
    
    # スケール分析（RVIの場合のみ）
    if algorithm == "rvi_gaussian" and multiscale_mode and auto_scale_analysis:
        target_distances, weights = analyze_terrain_scales(input_cog_path, pixel_size)
    elif algorithm == "rvi_gaussian" and multiscale_mode:
        target_distances, weights = _get_default_scales()
    else:
        target_distances, weights = None, None

    # GPU設定取得
    gpu_config = get_gpu_config(gpu_type, sigma, multiscale_mode, pixel_size, target_distances)
    
    # パラメータ最適化
    if tile_size is None:
        tile_size = gpu_config["tile_size"]
    if padding is None:
        padding = gpu_config["padding"]
    if max_workers is None:
        max_workers = gpu_config["max_workers"]
    
    logger.info(f"処理設定: {gpu_config['description']}")
    logger.info(f"タイルサイズ: {tile_size}x{tile_size}, ワーカー数: {max_workers}")

    # 一時ディレクトリ準備
    if os.path.exists(tmp_tile_dir):
        shutil.rmtree(tmp_tile_dir)
    os.makedirs(tmp_tile_dir, exist_ok=True)

    try:
        with rasterio.open(input_cog_path, 'r') as src:
            width = src.width
            height = src.height
            profile = src.profile.copy()
            nodata = src.nodata
            src_transform = src.transform
            src_crs = src.crs

            n_tiles_x = math.ceil(width / tile_size)
            n_tiles_y = math.ceil(height / tile_size)
            total_tiles = n_tiles_x * n_tiles_y

            logger.info(f"処理タイル数: {n_tiles_x} x {n_tiles_y} = {total_tiles}")

            # タイル情報事前計算
            tile_infos = []
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    core_x = tx * tile_size
                    core_y = ty * tile_size
                    core_w = min(tile_size, width - core_x)
                    core_h = min(tile_size, height - core_y)

                    win_x_off = max(core_x - padding, 0)
                    win_y_off = max(core_y - padding, 0)
                    win_x_end = min(core_x + core_w + padding, width)
                    win_y_end = min(core_y + core_h + padding, height)
                    win_w = win_x_end - win_x_off
                    win_h = win_y_end - win_y_off

                    tile_info = (ty, tx, core_x, core_y, core_w, core_h, 
                               win_x_off, win_y_off, win_w, win_h)
                    tile_infos.append(tile_info)

            # 並列処理
            processed_tiles = []
            skipped_tiles = []
            error_tiles = []
            completed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tile = {
                    executor.submit(
                        process_single_tile,
                        input_cog_path, tile_info, tmp_tile_dir, algorithm, sigma,
                        nodata, src_transform, src_crs, profile,
                        nodata_threshold, gpu_config.get("vram_monitor", False),
                        multiscale_mode, pixel_size, target_distances, weights,
                        **algo_params
                    ): tile_info for tile_info in tile_infos
                }

                for future in as_completed(future_to_tile):
                    result = future.result()
                    completed_count += 1
                    progress = completed_count / total_tiles * 100
                    
                    if result.success:
                        processed_tiles.append(result)
                        if completed_count % 10 == 0:
                            logger.info(f"✓ 処理完了: {completed_count}/{total_tiles} ({progress:.1f}%)")
                    elif result.skipped_reason:
                        skipped_tiles.append(result)
                    else:
                        error_tiles.append(result)

            logger.info(f"処理結果: 成功{len(processed_tiles)}, スキップ{len(skipped_tiles)}, エラー{len(error_tiles)}")

            if error_tiles:
                error_details = "\n".join([f"タイル({t.tile_y}, {t.tile_x}): {t.error_message}" 
                                         for t in error_tiles[:3]])
                raise RuntimeError(f"タイル処理エラー:\n{error_details}")

            if not processed_tiles:
                raise ValueError("処理されたタイルがありません")

        # COG生成
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        
        # COG品質検証
        _validate_cog_for_qgis(output_cog_path)

    except Exception as e:
        if os.path.exists(tmp_tile_dir):
            logger.error(f"エラーが発生しました。タイルディレクトリを保持します: {tmp_tile_dir}")
            logger.info("COG生成のみ実行するには: --cog-only オプションを使用してください")
        raise
    
    logger.info("=== 処理完了 ===")
    logger.info("🎯 生成されたCOGは最適化済みです")


def resume_cog_generation(
    tmp_tile_dir: str,
    output_cog_path: str,
    gpu_type: str = "auto",
    sigma: float = 10.0,
    multiscale_mode: bool = True,
    pixel_size: float = 0.5
):
    """
    既存のタイルからCOG生成のみを実行する関数
    """
    logger.info("=== タイルからCOG生成再開 ===")
    
    # タイル存在確認
    if not os.path.exists(tmp_tile_dir):
        raise ValueError(f"タイルディレクトリが存在しません: {tmp_tile_dir}")
    
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))
    if not tile_files:
        raise ValueError(f"タイルファイルが見つかりません: {tmp_tile_dir}")
    
    logger.info(f"発見されたタイル数: {len(tile_files)}")
    
    # 最初のタイルから基本情報を取得
    sample_tile = tile_files[0]
    try:
        with rasterio.open(sample_tile) as src:
            logger.info(f"タイル例: {os.path.basename(sample_tile)}")
            logger.info(f"  サイズ: {src.width} x {src.height}")
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  データ型: {src.dtypes[0]}")
    except Exception as e:
        logger.warning(f"タイル情報取得警告: {e}")
    
    # GPU設定取得（元の処理設定を考慮）
    try:
        target_distances, weights = _get_default_scales()
        gpu_config = get_gpu_config(gpu_type, sigma, multiscale_mode, pixel_size, target_distances)
    except Exception as e:
        logger.warning(f"GPU設定警告: {e}, デフォルト設定を使用")
        gpu_config = get_gpu_config(gpu_type)
    
    # COG生成実行
    try:
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        _validate_cog_for_qgis(output_cog_path)
        logger.info("✅ COG生成完了")
        
        # 成功時のクリーンアップ提案
        logger.info("\n💡 COG生成が完了しました。")
        logger.info("一時タイルディレクトリを削除しますか？")
        logger.info(f"削除コマンド: rm -rf {tmp_tile_dir}")
        
    except Exception as e:
        logger.error(f"COG生成エラー: {e}")
        logger.error(f"タイルディレクトリは保持されています: {tmp_tile_dir}")
        raise