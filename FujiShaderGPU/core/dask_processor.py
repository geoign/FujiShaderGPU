# -*- coding: utf-8 -*-
"""
FujiShaderGPU/core/dask_processor.py
Dask-CUDA地形解析処理のコア実装
"""

###############################################################################
# 依存ライブラリ
###############################################################################
from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import tempfile
import time

import psutil
import rasterio
from pathlib import Path
from typing import List, Tuple, Optional
from osgeo import gdal
import cupy as cp
import dask.array as da
from dask import config as dask_config
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import get_client
import xarray as xr
from tqdm.auto import tqdm


# アルゴリズムのインポート
try:
    from ..algorithms.dask_registry import ALGORITHMS
except ImportError:
    # algorithms registry が存在しない場合の仮の定義
    ALGORITHMS = {}
    logging.warning("dask registry module not found. No algorithms available.")

from ..algorithms.common.auto_params import (
    determine_optimal_radii as determine_optimal_radii_shared,
)
from ..io.raster_info import metric_pixel_scales_from_metadata
from ..config.auto_tune import compute_dask_chunk
from .dask_cluster import (
    get_optimal_chunk_size as _cluster_optimal_chunk_size,
    make_cluster as _cluster_make_cluster,
)
from .dask_io import (
    is_zarr_path as _io_is_zarr_path,
    load_input_dataarray as _io_load_input_dataarray,
    write_zarr_output as _io_write_zarr_output,
)

# ロギング設定
logger = logging.getLogger(__name__)


def _detect_metric_scales_from_dataarray(dem: xr.DataArray) -> Tuple[float, float, float, bool, Optional[float]]:
    """Detect signed x/y metric pixel scales from an xarray+rioxarray DataArray."""
    try:
        transform = dem.rio.transform()
        bounds = dem.rio.bounds()
        crs = dem.rio.crs
        sx, sy, mean_m, is_geo, lat = metric_pixel_scales_from_metadata(
            transform=transform, crs=crs, bounds=bounds
        )
        return float(sx), float(sy), float(mean_m), bool(is_geo), lat
    except Exception:
        try:
            x_res = abs(float(dem.rio.resolution()[0]))
            y_res = abs(float(dem.rio.resolution()[1]))
            mean_m = 0.5 * (x_res + y_res)
            return float(x_res), float(y_res), float(mean_m), False, None
        except Exception:
            return 1.0, 1.0, 1.0, False, None

###############################################################################
# 1. Dask-CUDA クラスタ（改善版）
###############################################################################

def get_optimal_chunk_size(gpu_memory_gb: float = 40) -> int:
    """Return recommended chunk size from GPU memory."""
    return _cluster_optimal_chunk_size(gpu_memory_gb)

def make_cluster(memory_fraction: float = 0.6):
    """Create Dask-CUDA cluster via core/dask_cluster.py."""
    return _cluster_make_cluster(memory_fraction)

###############################################################################
# 3. 地形解析による自動パラメータ決定
###############################################################################

def analyze_terrain_characteristics(dem_arr: da.Array, sample_ratio: float = 0.01, 
                                   include_fft: bool = False) -> dict:
    """地形の特性を統合的に解析"""
    # サンプリング処理（共通部分）
    h, w = dem_arr.shape
    sample_size = int(min(h, w) * sample_ratio)
    sample_size = max(512, min(4096, sample_size))
    
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - sample_size // 2)
    y2 = min(h, cy + sample_size // 2)
    x1 = max(0, cx - sample_size // 2)
    x2 = min(w, cx + sample_size // 2)
    
    sample = dem_arr[y1:y2, x1:x2].compute()
    
    # 基本統計（共通部分）
    valid_mask = ~cp.isnan(sample)
    if not valid_mask.any():
        raise ValueError("No valid elevation data found")
    
    elevations = sample[valid_mask]
    stats = {
        'elevation_range': float(cp.ptp(elevations)),
        'std_dev': float(cp.std(elevations)),
        'sample_size': sample.shape
    }
    
    # 勾配計算（共通部分）
    dy, dx = cp.gradient(sample)
    slope = cp.sqrt(dy**2 + dx**2)
    valid_slope = slope[valid_mask]
    stats['mean_slope'] = float(cp.mean(valid_slope))
    stats['max_slope'] = float(cp.percentile(valid_slope, 95))
    
    # FFT解析（オプション）
    if include_fft:
        # 2次微分と曲率
        dyy, dyx = cp.gradient(dy)
        dxy, dxx = cp.gradient(dx)
        curvature = cp.abs(dxx + dyy)
        valid_curv = curvature[valid_mask]
        stats['mean_curvature'] = float(cp.mean(valid_curv))
        
        # FFTによる周波数解析
        if valid_mask.sum() > 1000:  # 十分なデータがある場合
            # 2D FFTで主要な周波数成分を検出
            fft = cp.fft.fft2(sample - cp.mean(elevations))
            power = cp.abs(fft)**2
            
            # 放射状平均パワースペクトル
            freq = cp.fft.fftfreq(sample.shape[0])
            freq_grid = cp.sqrt(freq[:, None]**2 + freq[None, :]**2)
            
            # 周波数ビンごとの平均パワー
            n_bins = 50
            freq_bins = cp.linspace(0, 0.5, n_bins)
            power_spectrum = []
            
            for i in range(n_bins - 1):
                mask = (freq_grid >= freq_bins[i]) & (freq_grid < freq_bins[i+1])
                if mask.any():
                    power_spectrum.append(float(cp.mean(power[mask])))
                else:
                    power_spectrum.append(0)
            
            # 主要な周波数成分を検出
            power_spectrum = cp.array(power_spectrum)
            peak_indices = cp.where(power_spectrum > cp.percentile(power_spectrum, 90))[0]
            
            if len(peak_indices) > 0:
                dominant_freqs = freq_bins[peak_indices]
                # 周波数を空間スケールに変換（ピクセル単位）
                dominant_scales = [float(1.0 / (f + 1e-10)) for f in dominant_freqs if f > 0.01]
            else:
                dominant_scales = []
            stats['dominant_scales'] = dominant_scales
        else:
            dominant_scales = []
            stats['dominant_scales'] = []
            stats['dominant_freqs'] = []
    else:
        stats['dominant_scales'] = []
        stats['dominant_freqs'] = []
        stats['mean_curvature'] = 0.0
    # auto-parameter 推定の共通指標
    stats["complexity_score"] = float(stats["mean_slope"] * stats["std_dev"])
    return stats


def determine_optimal_radii(terrain_stats: dict) -> Tuple[List[int], List[float]]:
    """共通モジュール経由で最適半径を決定。"""
    return determine_optimal_radii_shared(terrain_stats)

###############################################################################
# 4. 直接 COG 出力 (GDAL >= 3.8) - 改善版
###############################################################################

def get_cog_options(dtype: str) -> dict:
    """データ型に応じた最適なCOGオプションを返す"""
    base_options = {
        "COMPRESS": "ZSTD",
        "LEVEL": "6",
        "BLOCKSIZE": "512",
        "OVERVIEWS": "AUTO",
        "OVERVIEW_RESAMPLING": "NEAREST",
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }
    
    # データ型に応じてPREDICTORを設定
    if dtype in ['float32', 'float64']:
        base_options["PREDICTOR"] = "3"  # 浮動小数点用
    elif dtype in ['int16', 'int32', 'uint16', 'uint32']:
        base_options["PREDICTOR"] = "2"  # 整数用
    # uint8やその他の型ではPREDICTORを使用しない
    
    return base_options

def check_gdal_version() -> tuple:
    """GDAL バージョンをチェック"""
    version = gdal.VersionInfo("VERSION_NUM")
    major = int(version[0])
    minor = int(version[1:3])
    return major, minor

def _write_cog_da_original(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """DataArray を直接 COG として保存（進捗表示付き）"""
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    
    dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    if not hasattr(data, 'rio') or data.rio.crs is None:
        logger.warning("No CRS found in data. Output may not have proper georeferencing.")
    
    if use_cog_driver:
        try:
            logger.info(f"Using COG driver (GDAL {major}.{minor}) with dtype={dtype_str}")
            with rasterio.Env(GDAL_CACHEMAX=512):
                if show_progress:
                    # より詳細な進捗表示
                    logger.info("Computing result chunks...")
                    # tqdmを使用した進捗表示                    
                    class TqdmCallback(Callback):
                        def __init__(self):
                            self.tqdm = None
                            
                        def _start(self, dsk):
                            self.tqdm = tqdm(total=len(dsk), desc='Computing', unit='tasks')
                            
                        def _posttask(self, key, result, dsk, _state, _worker_id):
                            self.tqdm.update(1)
                            
                        def _finish(self, dsk, _state, _failed):
                            self.tqdm.close()
                    
                    with TqdmCallback():
                        computed_data = data.compute()
                else:
                    computed_data = data.compute()
                
                # 計算済みデータをxarrayに戻す
                computed_da = xr.DataArray(
                    computed_data,
                    dims=data.dims,
                    coords=data.coords,
                    attrs=data.attrs,
                    name=data.name
                )
                
                # CRS情報を引き継ぐ
                if hasattr(data, 'rio') and data.rio.crs is not None:
                    computed_da.rio.write_crs(data.rio.crs, inplace=True)
                
                # COG書き込み
                logger.info("Writing to COG...")
                computed_da.rio.to_raster(
                    dst,
                    driver="COG",
                    **cog_options,
                )
            
            size_mb = os.path.getsize(dst) / 2**20
            logger.info(f"[OK] COG written: {dst} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.warning(f"COG driver failed: {e}, falling back to gdal_translate")
            _fallback_cog_write(data, dst, cog_options)
    else:
        logger.warning(f"GDAL {major}.{minor} < 3.8, using fallback method")
        _fallback_cog_write(data, dst, cog_options)

def _write_cog_da_chunked_impl(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """大規模データ用のチャンク単位書き込み実装"""
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    # メモリ制限を考慮してチャンクサイズを動的に調整
    available_memory = cp.get_default_memory_pool().free_bytes() / (1024**3)
    if available_memory < 10:  # 利用可能メモリが10GB未満
        # より積極的なメモリ解放
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        # Daskワーカーのメモリも解放
        try:
            client = get_client()
            client.run(lambda: gc.collect())
        except Exception:
            pass

    # DRAMの空き容量を確認してプリフェッチを有効化
    mem_info = psutil.virtual_memory()
    available_ram_gb = mem_info.available / (1024**3)
    
    if available_ram_gb > 20:  # 20GB以上空いている場合
        logger.info(f"Enabling chunk prefetching (available DRAM: {available_ram_gb:.1f}GB)")
        
        # Daskのスケジューラーヒントを設定
        prefetch_config = {
            "optimization.fuse.active": False,  # チャンクの融合を無効化
            "distributed.worker.memory.pause": 0.90,  # メモリ使用率90%まで許可
            "distributed.worker.memory.spill": 0.95,  # 95%でスピル
        }
    else:
        logger.info(f"Prefetching disabled (available DRAM: {available_ram_gb:.1f}GB < 20GB)")
        prefetch_config = {}
    
    # プリフェッチ設定を適用してチャンク処理を実行
    with dask_config.set(prefetch_config):

        # 一時ディレクトリ作成
        with tempfile.TemporaryDirectory() as tmpdir:
            # チャンクごとに処理
            chunk_files = []
            
            # Dask配列のチャンクを取得
            if hasattr(data.data, 'to_delayed'):
                # data.dataがDask配列の場合
                delayed_chunks = data.data.to_delayed()
                
                # チャンク情報の検証
                if not hasattr(data, 'chunks') or data.chunks is None:
                    logger.warning("No chunk information found, falling back to regular processing")
                    _write_cog_da_original(data, dst, show_progress)
                    return
                
                # 進捗表示の準備
                total_chunks = delayed_chunks.shape[0] * delayed_chunks.shape[1]
                
                # チャンクの形状を保持しながら処理
                chunk_idx = 0
                with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                    for i in range(delayed_chunks.shape[0]):
                        for j in range(delayed_chunks.shape[1]):
                            chunk_file = Path(tmpdir) / f"chunk_{chunk_idx}.tif"
                            try:
                                # チャンクを計算
                                chunk_data = delayed_chunks[i, j].compute()
                                
                                # チャンクの実際のサイズを取得
                                chunk_height, chunk_width = chunk_data.shape
                                
                                # チャンクの開始位置を計算
                                y_start = sum(data.chunks[0][:i])
                                x_start = sum(data.chunks[1][:j])
                                
                                # チャンクの終了位置を計算
                                y_end = y_start + chunk_height
                                x_end = x_start + chunk_width
                                
                                # チャンクをDataArrayとして作成
                                chunk_da = xr.DataArray(
                                    chunk_data,
                                    dims=data.dims,
                                    coords={
                                        data.dims[0]: data.coords[data.dims[0]].isel({data.dims[0]: slice(y_start, y_end)}),
                                        data.dims[1]: data.coords[data.dims[1]].isel({data.dims[1]: slice(x_start, x_end)})
                                    },
                                    attrs=data.attrs
                                )

                                # 座標参照系を設定（存在する場合のみ）
                                if hasattr(data, 'rio') and data.rio.crs is not None:
                                    chunk_da.rio.write_crs(data.rio.crs, inplace=True)
                                
                                # チャンクをGeoTIFFとして保存
                                chunk_da.rio.to_raster(
                                    chunk_file,
                                    driver="GTiff",
                                    compress="ZSTD",
                                    tiled=True,
                                    blockxsize=512,
                                    blockysize=512
                                )
                                chunk_files.append(chunk_file)
                                
                                # メモリ解放
                                del chunk_data, chunk_da
                                chunk_idx += 1

                                # 10チャンクごとに軽量クリーンアップ
                                if chunk_idx % 10 == 0:
                                    cp.get_default_memory_pool().free_all_blocks()
                                
                                # 進捗更新
                                pbar.update(1)
                                pbar.set_postfix({"saved": f"{len(chunk_files)}", "size_MB": f"{os.path.getsize(chunk_file)/(1024**2):.1f}"})
                                
                            except Exception as e:
                                logger.error(f"Failed to process chunk {i},{j}: {e}")
                                raise
            else:
                # Dask配列でない場合は通常の処理にフォールバック
                logger.info("Data is not chunked with Dask, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return
                
            # VRTで統合してCOGに変換
            if not chunk_files:
                raise ValueError("No chunks were successfully processed")
                
            logger.info(f"Creating VRT from {len(chunk_files)} chunks...")
            vrt_file = Path(tmpdir) / "merged.vrt"
            gdal.BuildVRT(str(vrt_file), [str(f) for f in chunk_files])
            
            # COGに変換
            logger.info("Converting to COG format...")
            
            # GDALの進捗コールバック
            pbar = tqdm(total=100, desc="COG conversion", unit="%")
            def gdal_progress_callback(complete, _message, _cb_data):
                pbar.n = int(complete * 100)
                pbar.refresh()
                if complete >= 1.0:
                    pbar.close()
                return 1
            
            gdal.Translate(
                str(dst),
                str(vrt_file),
                format="COG" if use_cog_driver else "GTiff",
                creationOptions=list(f"{k}={v}" for k, v in cog_options.items()),
                callback=gdal_progress_callback
            )
            
            logger.info(f"Successfully created COG from {len(chunk_files)} chunks")

def write_cog_da_chunked(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """COG書き出し（メモリ容量に応じて自動選択）"""
    total_gb = data.nbytes / (1024**3)
    
    # システムメモリに基づいて閾値を自動決定
    mem_info = psutil.virtual_memory()
    total_ram_gb = mem_info.total / (1024**3)
    available_ram_gb = mem_info.available / (1024**3)
    
    # Google Colab環境の検出
    is_colab = 'google.colab' in sys.modules
    
    # 安全係数の設定
    if is_colab:
        # Colabは保守的に（利用可能メモリの40%）
        safety_factor = 0.4
        # ただし、最低でも20GB、最高でも60GBに制限
        min_threshold = 20
        max_threshold = 60
    else:
        # ローカル環境はもう少し積極的に（60%）
        safety_factor = 0.6
        min_threshold = 30
        max_threshold = 100
    
    # 閾値の計算
    # 現在利用可能なメモリベースで計算（より現実的）
    chunk_threshold = available_ram_gb * safety_factor
    
    # 範囲内に収める
    chunk_threshold = max(min_threshold, min(chunk_threshold, max_threshold))
    
    # ログ出力
    logger.info(f"System RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")
    logger.info(f"Memory threshold: {chunk_threshold:.1f}GB (safety factor: {safety_factor*100:.0f}%)")
    
    # GPU情報も参考に表示
    try:
        meminfo = cp.cuda.runtime.memGetInfo()
        vram_free_gb = meminfo[0] / (1024**3)
        vram_total_gb = meminfo[1] / (1024**3)
        logger.info(f"GPU VRAM: {vram_total_gb:.1f}GB total, {vram_free_gb:.1f}GB free")
    except Exception:
        pass
    
    # データサイズと閾値の比較
    if total_gb > chunk_threshold:
        logger.info(f"Large dataset ({total_gb:.1f}GB) > threshold ({chunk_threshold:.1f}GB)")
        logger.info("Using chunked writing to avoid memory issues")
        _write_cog_da_chunked_impl(data, dst, show_progress)
    else:
        logger.info(f"Dataset ({total_gb:.1f}GB) <= threshold ({chunk_threshold:.1f}GB)")
        logger.info("Using direct writing for better performance")
        _write_cog_da_original(data, dst, show_progress)

def _fallback_cog_write(data: xr.DataArray, dst: Path, cog_options: dict):
    """フォールバック：一時ファイル経由でCOG作成"""
    tmp = dst.with_suffix(".tmp.tif")
    try:
        # COG固有のオプションを除外
        tiff_options = {k: v for k, v in cog_options.items() 
                       if k not in ['OVERVIEWS', 'OVERVIEW_RESAMPLING']}
        
        # 進捗表示付きで計算してから書き込み        
        logger.info("Computing result...")
        with ProgressBar():
            computed_data = data.compute()
        
        # 計算済みデータをxarrayに戻す
        computed_da = xr.DataArray(
            computed_data,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs,
            name=data.name
        )
        
        # CRS情報を引き継ぐ
        if hasattr(data, 'rio') and data.rio.crs is not None:
            computed_da.rio.write_crs(data.rio.crs, inplace=True)
        
        logger.info("Writing temporary TIFF...")
        computed_da.rio.to_raster(
            tmp,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            **tiff_options
        )
        build_cog_with_overviews(tmp, dst, cog_options)
    finally:
        tmp.unlink(missing_ok=True)

###############################################################################
# 5. gdal_translate/gdaladdo フォールバック関数
###############################################################################

def build_cog_with_overviews(src: Path, dst: Path, cog_options: dict):
    """旧版 GDAL 用: 一時 TIFF → COG 変換 + オーバービュー"""
    # CPU数を取得して並列処理
    num_cpus = os.cpu_count() or 1
    
    cmd_translate = [
        "gdal_translate", "-of", "COG",
        "-co", f"COMPRESS={cog_options.get('COMPRESS', 'ZSTD')}",
        "-co", f"LEVEL={cog_options.get('LEVEL', '1')}",
        "-co", f"BLOCKSIZE={cog_options.get('BLOCKSIZE', '512')}",
        "-co", "BIGTIFF=YES",
        "-co", f"NUM_THREADS={num_cpus}",
    ]
    
    # PREDICTORがある場合のみ追加
    if 'PREDICTOR' in cog_options:
        cmd_translate.extend(["-co", f"PREDICTOR={cog_options['PREDICTOR']}"])
    
    cmd_translate.extend([str(src), str(dst)])
    
    cmd_addo = [
        "gdaladdo", "-r", "nearest",
        "--config", "COMPRESS_OVERVIEW", cog_options.get('COMPRESS', 'ZSTD'),
        "--config", "GDAL_NUM_THREADS", str(num_cpus),
        str(dst), "2", "4", "8", "16", "32", "64",
    ]
    
    try:
        subprocess.run(cmd_translate, check=True, capture_output=True)
        subprocess.run(cmd_addo, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"GDAL command failed: {e.stderr.decode()}")
        raise

###############################################################################
# 6. メインパイプライン
###############################################################################

def validate_inputs(src_cog: str):
    """入力パラメータの検証"""
    if not Path(src_cog).exists():
        raise FileNotFoundError(f"Input file not found: {src_cog}")


def _is_zarr_path(path: str) -> bool:
    return _io_is_zarr_path(path)


def _load_input_dataarray(src_path: str, chunk: int) -> xr.DataArray:
    return _io_load_input_dataarray(src_path, chunk)


def _write_zarr_output(data: xr.DataArray, dst: Path, show_progress: bool = True):
    logger.info("Writing output as Zarr: %s", dst)
    _io_write_zarr_output(data, dst, show_progress=show_progress)

def run_pipeline(
    src_cog: str,
    dst_cog: str,
    algorithm: str = "rvi",
    radii: Optional[List[int]] = None,
    agg: str = "mean",
    chunk: Optional[int] = None,
    show_progress: bool = True,
    auto_radii: bool = True,
    **algo_params
):
    """改善されたメインパイプライン"""
    # アルゴリズムの確認
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    
    algo = ALGORITHMS[algorithm]
    
    # 入力検証
    validate_inputs(src_cog)
    
    # メモリ状況の確認
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.total / 2**30:.1f} GB total, "
                f"{mem.available / 2**30:.1f} GB available")
    
    # メモリフラクションの取得（algo_paramsから、なければデフォルト）
    memory_fraction = algo_params.pop('memory_fraction', None)
    cluster, client = make_cluster(memory_fraction)
    
    try:
        # チャンクサイズの自動決定
        if chunk is None:
            if _is_zarr_path(src_cog):
                dem_probe = _load_input_dataarray(src_cog, 1024)
                height = dem_probe.sizes[dem_probe.dims[-2]]
                width = dem_probe.sizes[dem_probe.dims[-1]]
                total_pixels = height * width
                total_gb = (total_pixels * 4) / (1024**3)
            else:
                with rasterio.open(src_cog) as src:
                    total_pixels = src.width * src.height
                    total_gb = (total_pixels * 4) / (1024**3)
                
            # VRAM-aware dynamic chunk sizing
            try:
                _meminfo = cp.cuda.runtime.memGetInfo()
                _vram_gb = _meminfo[1] / (1024**3)
            except Exception:
                _vram_gb = 16.0
            chunk = compute_dask_chunk(
                _vram_gb, data_gb=total_gb, algorithm=algorithm,
            )
            
            logger.info(f"Dataset size: {total_gb:.1f} GB, using chunk size: {chunk}x{chunk}")
        
        # 6-1) DEM 遅延ロード (COG または Zarr)
        dem: xr.DataArray = _load_input_dataarray(src_cog, chunk)
        
        logger.info(f"DEM shape: {dem.shape}, dtype: {dem.dtype}, "
                   f"chunks: {dem.chunks}")
        
        # 6-2) CuPy 配列へ変換（改善：メタデータ指定）
        gpu_arr: da.Array = dem.data.map_blocks(
            cp.asarray, 
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        
        # アルゴリズム固有のデフォルトパラメータを取得
        default_params = algo.get_default_params()

        # パラメータの準備
        params = {
            **default_params,
            **algo_params,
            'show_progress': show_progress,
            'agg': agg
        }

        # Inject anisotropic pixel scales (simple geographic DEM support).
        px_m_x, px_m_y, pixel_size_m, is_geo, lat_center = _detect_metric_scales_from_dataarray(dem)
        params['pixel_size'] = float(pixel_size_m)
        params.setdefault('pixel_scale_x', float(px_m_x))
        params.setdefault('pixel_scale_y', float(px_m_y))
        # Tile path parity: set is_geographic_dem and elevation_scale.
        # Note: in the Dask path pixel_scale_x/y are raw meter values so
        #   elevation_scale is NOT applied to the DEM array.  The field is
        #   provided only for algorithm-level flag consistency with the tile
        #   pipeline (where DEM is pre-scaled by elevation_scale).
        params.setdefault('is_geographic_dem', bool(is_geo))
        params.setdefault('elevation_scale',
                          float(1.0 / max(pixel_size_m, 1e-6)) if is_geo else 1.0)
        if is_geo:
            ratio = abs(px_m_y) / max(abs(px_m_x), 1e-9)
            logger.info(
                "Geographic DEM approximation enabled: "
                f"lat={lat_center:.3f}, dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m, dy/dx={ratio:.4f}"
            )
        else:
            logger.info(f"Projected pixel scales: dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m")
        
        # 6-2.5) 自動決定（RVIアルゴリズムの場合）
        if algorithm == "rvi":
            pixel_size = float(params.get('pixel_size', 1.0))
            
            # 新しい効率的なモードをデフォルトに
            if radii is None and auto_radii:
                logger.info("Analyzing terrain for automatic radii determination...")
                
                # 地形解析
                terrain_stats = analyze_terrain_characteristics(gpu_arr, sample_ratio=0.01, include_fft=False)
                terrain_stats['pixel_size'] = pixel_size
                
                # 最適な半径を決定
                radii, weights = determine_optimal_radii(terrain_stats)
                
                logger.info("Terrain analysis results:")
                logger.info(f"  - Elevation range: {terrain_stats['elevation_range']:.1f} m")
                logger.info(f"  - Mean slope: {terrain_stats['mean_slope']:.3f}")
                logger.info(f"  - Auto-determined radii: {radii} pixels")
                logger.info(f"  - Weights: {[f'{w:.2f}' for w in weights]}")
                
                # パラメータに設定
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = weights
                
            elif radii is not None:
                # 手動指定の半径モード
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = algo_params.get('weights', None)
            
            else:
                raise ValueError("Either provide radii or enable auto_radii")

        # 多くのアルゴリズムでピクセルサイズが必要（RVI以外の場合）
        elif algorithm != "rvi" and ('pixel_size' not in params or params['pixel_size'] == 1.0):
            # Already injected above; keep this branch as a no-op for compatibility.
            params['pixel_size'] = float(params.get('pixel_size', 1.0))

        # 追加：fractal_anomaly用のradii処理
        if algorithm == "fractal_anomaly" and radii is not None:
            params['radii'] = radii
            logger.info(f"Setting radii for fractal_anomaly: {radii}")
        
        # 6-3) アルゴリズム実行（run_pipeline内）
        logger.info(f"Computing {algorithm} with parameters: {params}")

        # アルゴリズムを適用（遅延評価）
        result_gpu: da.Array = algo.process(gpu_arr, **params)

        # 6-4) GPU→CPU 戻し（改善：明示的なdtype）
        result_cpu = result_gpu.map_blocks(
            cp.asnumpy, 
            dtype="float32",
            meta=cp.empty((0, 0), dtype=cp.float32).get()
        )

        # 進捗表示付きで計算を実行
        # ────────── GPU→CPU 変換後の計算トリガ ──────────
        # 20 GB を超える場合は persist をスキップし、
        # write_cog_da_chunked() によるストリーム計算に任せる。
        total_gb = result_cpu.nbytes / (1024**3)
        if total_gb <= 20:
            if show_progress:
                logger.info("Persisting small dataset for faster workflow")
                result_cpu = client.persist(result_cpu, optimize_graph=True)
                progress(result_cpu, interval='1s')
            else:
                result_cpu = result_cpu.persist()
        else:
            logger.info(f"Large dataset ({total_gb:.1f} GB) - skip persist; "
                        "chunked writer will stream-compute each tile")
    
        # 6-5) xarray ラップ（改善：座標構築の簡略化）
        if agg == "stack" and 'sigmas' in params and params['sigmas'] is not None:
            dims = ("scale", *dem.dims)
            coords = {"scale": params['sigmas'], **dem.coords}
        else:
            dims = dem.dims
            coords = dem.coords
        
        # アルゴリズムによって適切なデータ範囲を設定
        if algorithm in ['hillshade']:
            # Hillshade is now also float32 (0..1), aligned with tile backend.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1",
                "data_type": "float32"
            }
        elif algorithm in ['slope']:
            # Slopeは単位による（度、パーセント、ラジアン）
            unit = params.get('unit', 'degree')
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "unit": unit,
                "data_type": "float32"
            }
        elif algorithm in ['lrm', 'rvi']:
            # LRM、RVIは-1から+1
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "-1 to +1",
                "data_type": "float32"
            }
        else:
            # その他は0から1
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1",
                "data_type": "float32"
            }
            
        result_da = xr.DataArray(
            result_cpu,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=algorithm.upper(),
        )
        
        # CRS情報を元のDEMから引き継ぐ（COG入力時）
        if hasattr(dem, 'rio') and dem.rio.crs is not None:
            result_da.rio.write_crs(dem.rio.crs, inplace=True)
        
        # 6-6) 出力 (COG または Zarr)
        dst_path = Path(dst_cog)
        if _is_zarr_path(str(dst_path)):
            _write_zarr_output(result_da, dst_path, show_progress=show_progress)
        else:
            write_cog_da_chunked(result_da, dst_path, show_progress=show_progress)
        logger.info("Pipeline completed successfully!")
        gc.collect()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # CuPyメモリプールの明示的なクリア
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        # より確実なクリーンアップ
        try:
            # ワーカーのメモリを強制的にクリア
            def clear_worker_memory():
                import gc
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                gc.collect()
                return True   
            client.run(clear_worker_memory)
            # clientを先に閉じて、完全に終了するまで待つ
            client.close(timeout=10)
            client.shutdown()  # 全てのワーカーとスケジューラーを確実に終了
        except Exception as e:
            logger.debug(f"Client shutdown warning (can be ignored): {e}")
        
        try:
            # clusterの終了（既に終了している可能性があるので例外を無視）
            cluster.close(timeout=10)
        except Exception as e:
            logger.debug(f"Cluster close warning (can be ignored): {e}")
        
        # Daskワーカープロセスの確実な終了を待つ
        time.sleep(3)
        
        # 最終的なGC
        gc.collect()

