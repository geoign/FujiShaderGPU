"""
FujiShaderGPU/core/tile_processor.py
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.gpu_memory import gpu_memory_pool
from ..config.system_config import get_gpu_config
from ..io.raster_info import detect_pixel_size_from_cog
from ..utils.types import TileResult
from ..utils.scale_analysis import analyze_terrain_scales
from ..utils.nodata_handler import _handle_nodata_ultra_fast
from ..algorithms.rvi_gaussian import _compute_multiscale_rvi_ultra_fast
from ..utils.scale_analysis import _get_default_scales
from ..io.cog_builder import _build_vrt_and_cog_ultra_fast
from ..io.cog_validator import _validate_cog_for_qgis
import os, math, glob, shutil, rasterio
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from rasterio.windows import Window
from rasterio.transform import Affine
from typing import Optional, Tuple, List
from importlib import import_module

def _load_algorithm(name: str):
    mod = import_module(f"..algorithms.{name}", package=__package__)
    # クラスの場合と関数の場合の両対応
    if hasattr(mod, "process"):          # クラス実装
        return mod
    elif hasattr(mod, f"_{name}"):       # 関数実装
        return getattr(mod, f"_{name}")
    raise ValueError(f"Algorithm {name} not found")

def process_single_tile(
    input_cog_path: str,
    tile_info: Tuple[int, int, int, int, int, int, int, int, int, int],
    tmp_tile_dir: str,
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
    weights: Optional[List[float]] = None
) -> TileResult:
    """
    単一タイル処理（超高速化版）
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
                    nodata_ratio = np.count_nonzero(mask_nodata) / mask_nodata.size  # np.sum → np.count_nonzero
                    
                    if nodata_ratio >= nodata_threshold:
                        return TileResult(
                            ty, tx, False,
                            skipped_reason=f"NoDataが{nodata_ratio:.1%}を占める（閾値:{nodata_threshold:.1%}）"
                        )
                    
                    if nodata_ratio > 0.8:
                        print(f"注意: タイル({ty}, {tx}) NoData率が高いです: {nodata_ratio:.1%}")
                    
                    # 超高速NoData処理
                    dem_tile_processed = _handle_nodata_ultra_fast(dem_tile, mask_nodata)
                else:
                    dem_tile_processed = dem_tile

                # GPU転送（最適化）
                dem_gpu = cp.asarray(dem_tile_processed, dtype=cp.float32)

                # マルチスケールRVI計算（超高速化）
                if multiscale_mode:
                    if target_distances is None or weights is None:
                        target_distances, weights = _get_default_scales()
                    rvi_gpu = _compute_multiscale_rvi_ultra_fast(dem_gpu, target_distances, weights, pixel_size)
                else:
                    # シングルスケール（最適化）
                    dem_blur_gpu = cpx_ndimage.gaussian_filter(
                        dem_gpu, sigma=sigma, mode="nearest", truncate=4.0
                    )
                    rvi_gpu = dem_gpu - dem_blur_gpu
                    del dem_blur_gpu

                # NoData復元（必要時のみ）
                if mask_nodata is not None:
                    rvi_gpu[cp.asarray(mask_nodata)] = cp.float32(nodata or 0)

                # CPU転送（最適化）
                rvi_tile = cp.asnumpy(rvi_gpu)
                del dem_gpu, rvi_gpu

                # コア領域抽出
                core_x_in_win = core_x - win_x_off
                core_y_in_win = core_y - win_y_off
                rvi_core = rvi_tile[
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
                    "zlevel": 1,           # 高速圧縮
                    "tiled": True,
                    "blockxsize": min(512, core_w),   # QGIS最適ブロックサイズ（1024→512）
                    "blockysize": min(512, core_h),
                    "BIGTIFF": "YES",
                    "nodata": nodata,
                    "NUM_THREADS": "ALL_CPUS"  # 全CPU活用
                })

                tile_filename = os.path.join(
                    tmp_tile_dir, f"tile_{ty:03d}_{tx:03d}.tif"
                )

                # 高速書き込み
                with rasterio.open(tile_filename, "w", **tile_profile) as dst:
                    dst.write(rvi_core, 1)

                return TileResult(ty, tx, True, tile_filename)
                
    except Exception as e:
        return TileResult(ty, tx, False, error_message=str(e))


def process_dem_tiles(
    input_cog_path: str,
    output_cog_path: str,
    tmp_tile_dir: str = "tiles_tmp",
    tile_size: Optional[int] = None,
    padding: Optional[int] = None,
    sigma: float = 10.0,
    max_workers: Optional[int] = None,
    nodata_threshold: float = 1.0,
    gpu_type: str = "auto",
    multiscale_mode: bool = True,
    pixel_size: Optional[float] = None,
    auto_scale_analysis: bool = True,
    cog_only: bool = False  # ★ COG生成のみのフラグを追加
):
    """
    超高速DEM処理メイン関数（COG生成のみオプション追加）
    """
    # ★ COG生成のみの場合
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
    
    print("=== DEM→RVI超高速処理開始 ===")
    
    # ピクセルサイズ検出
    if pixel_size is None:
        pixel_size = detect_pixel_size_from_cog(input_cog_path)
    
    # スケール分析
    if multiscale_mode and auto_scale_analysis:
        target_distances, weights = analyze_terrain_scales(input_cog_path, pixel_size)
    elif multiscale_mode:
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
    
    print(f"超高速設定: {gpu_config['description']}")
    print(f"タイルサイズ: {tile_size}x{tile_size}, ワーカー数: {max_workers}")

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

            print(f"処理タイル数: {n_tiles_x} x {n_tiles_y} = {total_tiles}")

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

            # 超高速並列処理
            processed_tiles = []
            skipped_tiles = []
            error_tiles = []
            completed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tile = {
                    executor.submit(
                        process_single_tile,
                        input_cog_path, tile_info, tmp_tile_dir, sigma,
                        nodata, src_transform, src_crs, profile,
                        nodata_threshold, gpu_config.get("vram_monitor", False),
                        multiscale_mode, pixel_size, target_distances, weights
                    ): tile_info for tile_info in tile_infos
                }

                for future in as_completed(future_to_tile):
                    result = future.result()
                    completed_count += 1
                    progress = completed_count / total_tiles * 100
                    
                    if result.success:
                        processed_tiles.append(result)
                        if completed_count % 10 == 0:  # 10タイルごとに表示
                            print(f"✓ 処理完了: {completed_count}/{total_tiles} ({progress:.1f}%)")
                    elif result.skipped_reason:
                        skipped_tiles.append(result)
                    else:
                        error_tiles.append(result)

            print(f"\n処理結果: 成功{len(processed_tiles)}, スキップ{len(skipped_tiles)}, エラー{len(error_tiles)}")

            if error_tiles:
                error_details = "\n".join([f"タイル({t.tile_y}, {t.tile_x}): {t.error_message}" 
                                         for t in error_tiles[:3]])
                raise RuntimeError(f"タイル処理エラー:\n{error_details}")

            if not processed_tiles:
                raise ValueError("処理されたタイルがありません")

        # 超高速COG生成
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        
        # COG品質検証
        _validate_cog_for_qgis(output_cog_path)

    except Exception as e:
        if os.path.exists(tmp_tile_dir):
            print(f"❌ エラーが発生しました。タイルディレクトリを保持します: {tmp_tile_dir}")
            print(f"COG生成のみ実行するには: --cog-only オプションを使用してください")
        raise
    
    print("=== 超高速処理完了 ===")
    print("🎯 QGISでの高速表示のため、生成されたCOGは最適化済みです")


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
    print("=== タイルからCOG生成再開 ===")
    
    # タイル存在確認
    if not os.path.exists(tmp_tile_dir):
        raise ValueError(f"タイルディレクトリが存在しません: {tmp_tile_dir}")
    
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))
    if not tile_files:
        raise ValueError(f"タイルファイルが見つかりません: {tmp_tile_dir}")
    
    print(f"発見されたタイル数: {len(tile_files)}")
    
    # 最初のタイルから基本情報を取得
    sample_tile = tile_files[0]
    try:
        with rasterio.open(sample_tile) as src:
            print(f"タイル例: {os.path.basename(sample_tile)}")
            print(f"  サイズ: {src.width} x {src.height}")
            print(f"  CRS: {src.crs}")
            print(f"  データ型: {src.dtypes[0]}")
    except Exception as e:
        print(f"タイル情報取得警告: {e}")
    
    # GPU設定取得（元の処理設定を考慮）
    try:
        target_distances, weights = _get_default_scales()
        gpu_config = get_gpu_config(gpu_type, sigma, multiscale_mode, pixel_size, target_distances)
    except Exception as e:
        print(f"GPU設定警告: {e}, デフォルト設定を使用")
        gpu_config = get_gpu_config(gpu_type)
    
    # COG生成実行
    try:
        _build_vrt_and_cog_ultra_fast(tmp_tile_dir, output_cog_path, gpu_config)
        _validate_cog_for_qgis(output_cog_path)
        print("✅ COG生成完了")
        
        # 成功時のクリーンアップ提案
        print(f"\n💡 COG生成が完了しました。")
        print(f"一時タイルディレクトリを削除しますか？")
        print(f"削除コマンド: rm -rf {tmp_tile_dir}")
        
    except Exception as e:
        print(f"❌ COG生成エラー: {e}")
        print(f"タイルディレクトリは保持されています: {tmp_tile_dir}")
        raise
