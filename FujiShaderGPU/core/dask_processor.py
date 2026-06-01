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
import shutil
import sys
import tempfile
import time

import numpy as np
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

from ..algorithms.common.auto_params import determine_optimal_radii
from ..io.raster_info import metric_pixel_scales_from_metadata
from ..config.auto_tune import compute_dask_chunk
from .dask_cluster import make_cluster
from .dask_io import (
    is_zarr_path,
    load_input_dataarray,
    write_zarr_output,
)

# ロギング設定
logger = logging.getLogger(__name__)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.1f}GB"


def _configure_gdal_read_performance() -> None:
    """Enable multi-threaded COG decoding and block caching for input reads.

    Call this *before* the Dask-CUDA cluster is created: ``LocalCUDACluster``
    spawns its worker (nanny) processes from this process, so they inherit the
    environment configured here, which is where the chunked input reads actually
    run.  ``setdefault`` is used so explicit user environment overrides win.
    """
    try:
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = 8.0
    # GDAL interprets a bare integer < 100000 as megabytes.
    cache_mb = int(max(1024, min(16384, avail_gb * 1024 * 0.1)))

    read_opts = {
        "GDAL_NUM_THREADS": "ALL_CPUS",          # multi-threaded (de)compression
        "GDAL_CACHEMAX": str(cache_mb),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "YES",
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
    }
    for key, value in read_opts.items():
        os.environ.setdefault(key, value)
        try:
            gdal.SetConfigOption(key, os.environ.get(key, value))
        except Exception:
            pass
    logger.info(
        "GDAL read performance configured: NUM_THREADS=%s, CACHEMAX=%sMB",
        os.environ.get("GDAL_NUM_THREADS"),
        os.environ.get("GDAL_CACHEMAX"),
    )


def _log_overview_availability(src_cog: str) -> None:
    """Log whether the input COG carries an overview pyramid.

    Decimated reads (global stats, terrain sampling) and downstream viewers are
    far faster when overviews exist; warn when they do not so the user can add
    them with ``gdaladdo`` ahead of time.
    """
    try:
        if is_zarr_path(src_cog):
            return
        with rasterio.open(src_cog) as src:
            ov = src.overviews(1) if src.count >= 1 else []
        if ov:
            logger.info("Input overviews available: %s (used for decimated sampling)", ov)
        else:
            logger.warning(
                "Input has no overviews; decimated stat/sampling reads will be slow. "
                "FujiShaderGPU expects an overview-bearing COG -- pre-process the input first:\n"
                "    python -m FujiShaderGPU.prepare %s prepared_cog.tif\n"
                "then run the pipeline on 'prepared_cog.tif'.",
                src_cog,
            )
    except Exception as exc:
        logger.debug("Overview availability check skipped: %s", exc)


def _select_chunk_temp_parent(data_nbytes: int) -> Path:
    """Choose and diagnose the temporary directory for chunk GeoTIFFs."""
    selected_from = None
    for env_name in ("FUJISHADER_TMP_DIR", "CPL_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(env_name)
        if value:
            selected_from = env_name
            parent = Path(value)
            break
    else:
        parent = Path(tempfile.gettempdir())

    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    estimated = max(data_nbytes, int(data_nbytes * 0.75))
    origin = f" from ${selected_from}" if selected_from else " from tempfile default"
    logger.info(
        "Chunk temporary directory%s: %s (free=%s)",
        origin,
        parent,
        _format_gib(usage.free),
    )
    if usage.free < estimated:
        logger.warning(
            "Chunk temporary directory may be too small for COG staging "
            "(free=%s, output array=%s). Set FUJISHADER_TMP_DIR, TMPDIR, "
            "TMP, TEMP, and CPL_TMPDIR to a large persistent volume.",
            _format_gib(usage.free),
            _format_gib(data_nbytes),
        )
    if not selected_from and str(parent).startswith("/tmp") and Path("/content").exists():
        try:
            content_free = shutil.disk_usage("/content").free
        except OSError:
            content_free = 0
        if content_free > usage.free:
            logger.warning(
                "Runpod/Colab-compatible layout detected: /tmp has %s free, "
                "while /content has %s free. For large COG output set "
                "FUJISHADER_TMP_DIR=/content/fujishader_tmp and CPL_TMPDIR "
                "to the same path.",
                _format_gib(usage.free),
                _format_gib(content_free),
            )
    return parent


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
# 1. Dask-CUDA クラスタ
#    make_cluster() は dask_cluster.py から直接インポート済み
###############################################################################

###############################################################################
# 3. 地形解析による自動パラメータ決定
###############################################################################

def analyze_terrain_characteristics(dem_arr: da.Array, sample_ratio: float = 0.01) -> dict:
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
    

    # auto-parameter 推定の共通指標
    stats["complexity_score"] = float(stats["mean_slope"] * stats["std_dev"])
    return stats




###############################################################################
# 4. 直接 COG 出力 (GDAL >= 3.8) - 改善版
###############################################################################

def get_cog_options(dtype: str) -> dict:
    """データ型に応じた最適なCOGオプションを返す"""
    base_options = {
        "COMPRESS": "ZSTD",
        "LEVEL": "1",
        "BLOCKSIZE": "512",
        "OVERVIEWS": "IGNORE_EXISTING",
        "OVERVIEW_COMPRESS": "ZSTD",
        "OVERVIEW_RESAMPLING": "AVERAGE",
        "OVERVIEW_COUNT": "8",
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
    """GDAL バージョンをチェック（整数演算で安全にパース）"""
    ver_num = int(gdal.VersionInfo("VERSION_NUM"))
    major = ver_num // 1_000_000
    minor = (ver_num % 1_000_000) // 10_000
    return major, minor


def _get_overview_count(tiff_path: Path) -> int:
    """Return the overview count on band 1, or 0 when the file cannot be read."""
    ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)
    if ds is None or ds.RasterCount < 1:
        return 0
    band = ds.GetRasterBand(1)
    overview_count = band.GetOverviewCount() if band is not None else 0
    ds = None
    return int(overview_count)


def _assert_has_overviews(tiff_path: Path):
    """Fail fast when COG creation unexpectedly omitted its overview pyramid."""
    overview_count = _get_overview_count(tiff_path)
    if overview_count <= 0:
        raise ValueError(f"COG output has no overviews: {tiff_path}")
    logger.info("COG overview count: %d", overview_count)


def _ensure_cog_has_overviews(dst: Path, cog_options: dict):
    """Rebuild a COG in-place when the writer produced no overviews."""
    if _get_overview_count(dst) > 0:
        _assert_has_overviews(dst)
        return

    logger.warning("COG output has no overviews; rebuilding with forced ZSTD overviews")
    src = dst.with_suffix(".no_overviews.tif")
    repaired = dst.with_suffix(".with_overviews.tif")
    src.unlink(missing_ok=True)
    repaired.unlink(missing_ok=True)
    dst.replace(src)
    try:
        build_cog_with_overviews(src, repaired, cog_options)
        repaired.replace(dst)
        _assert_has_overviews(dst)
    finally:
        src.unlink(missing_ok=True)
        repaired.unlink(missing_ok=True)


def _build_zstd_overviews(path: Path, cog_options: dict):
    """Build internal ZSTD-compressed overviews on a temporary GeoTIFF."""
    logger.info("Building ZSTD-compressed overviews: %s", path)
    previous_options = {
        "COMPRESS_OVERVIEW": gdal.GetConfigOption("COMPRESS_OVERVIEW"),
        "ZLEVEL_OVERVIEW": gdal.GetConfigOption("ZLEVEL_OVERVIEW"),
        "BIGTIFF_OVERVIEW": gdal.GetConfigOption("BIGTIFF_OVERVIEW"),
        "GDAL_TIFF_OVR_BLOCKSIZE": gdal.GetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE"),
        "GDAL_NUM_THREADS": gdal.GetConfigOption("GDAL_NUM_THREADS"),
    }
    try:
        gdal.SetConfigOption("COMPRESS_OVERVIEW", cog_options.get("OVERVIEW_COMPRESS", "ZSTD"))
        gdal.SetConfigOption("ZLEVEL_OVERVIEW", cog_options.get("LEVEL", "1"))
        gdal.SetConfigOption("BIGTIFF_OVERVIEW", "YES")
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", cog_options.get("BLOCKSIZE", "512"))
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            raise ValueError(f"Failed to open temporary TIFF for overviews: {path}")
        result = ds.BuildOverviews(
            cog_options.get("OVERVIEW_RESAMPLING", "AVERAGE"),
            [2, 4, 8, 16, 32, 64, 128, 256],
        )
        ds = None
        if result != 0:
            raise ValueError(f"BuildOverviews failed: {path}")
    finally:
        for key, value in previous_options.items():
            gdal.SetConfigOption(key, value)


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
                _ensure_cog_has_overviews(dst, cog_options)
            
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
    
    # VRAM実残量に基づいてメモリ解放を判断（プール内free_bytesではなく実VRAM使用）
    try:
        _free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        available_vram_gb = _free_bytes / (1024**3)
    except Exception:
        available_vram_gb = 10.0  # 安全なデフォルト値
    if available_vram_gb < 10:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
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
        tmp_parent = _select_chunk_temp_parent(int(data.nbytes))
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            # チャンクごとに処理
            chunk_files = []

            # Dask配列でない場合は通常の処理にフォールバック
            if not hasattr(data.data, 'to_delayed'):
                logger.info("Data is not chunked with Dask, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return

            # チャンク情報の検証
            if not hasattr(data, 'chunks') or data.chunks is None:
                logger.warning("No chunk information found, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return

            delayed_chunks = data.data.to_delayed()
            n_rows = int(delayed_chunks.shape[0])
            n_cols = int(delayed_chunks.shape[1])
            total_chunks = n_rows * n_cols

            y_dim, x_dim = data.dims[0], data.dims[1]
            src_crs = data.rio.crs if hasattr(data, 'rio') else None

            def _persist_chunk(idx: int, i: int, j: int, chunk_data) -> Path:
                """Write one computed chunk to a temporary GeoTIFF; return its path."""
                chunk_height, chunk_width = chunk_data.shape[0], chunk_data.shape[1]
                y_start = sum(data.chunks[0][:i])
                x_start = sum(data.chunks[1][:j])
                y_end = y_start + chunk_height
                x_end = x_start + chunk_width
                chunk_da = xr.DataArray(
                    chunk_data,
                    dims=data.dims,
                    coords={
                        y_dim: data.coords[y_dim].isel({y_dim: slice(y_start, y_end)}),
                        x_dim: data.coords[x_dim].isel({x_dim: slice(x_start, x_end)}),
                    },
                    attrs=data.attrs,
                )
                if src_crs is not None:
                    chunk_da.rio.write_crs(src_crs, inplace=True)
                chunk_file = Path(tmpdir) / f"chunk_{idx}.tif"
                chunk_da.rio.to_raster(
                    chunk_file,
                    driver="GTiff",
                    compress="ZSTD",
                    zstd_level=1,
                    predictor=3,
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    BIGTIFF="YES",
                    num_threads="ALL_CPUS",
                )
                del chunk_da
                return chunk_file

            # 並列ストリーミング書き込み:
            # チャンクグラフは embarrassingly parallel なので、複数の Dask-CUDA
            # ワーカー（=GPU）が別チャンクを並行計算する間にクライアントが完了分を
            # 書き出す。distributed クライアントが取得できない場合は直列処理に戻す。
            client = None
            try:
                client = get_client()
            except Exception:
                client = None

            if client is not None:
                try:
                    n_workers = max(1, len(client.scheduler_info().get("workers", {})))
                except Exception:
                    n_workers = 1
                max_inflight = max(2, n_workers * 2)
                from distributed import as_completed as _as_completed

                coords_flat = [(i, j) for i in range(n_rows) for j in range(n_cols)]
                task_iter = iter(enumerate(coords_flat))
                fut_meta = {}
                inflight = _as_completed()

                def _submit_next() -> bool:
                    try:
                        idx, (i, j) = next(task_iter)
                    except StopIteration:
                        return False
                    fut = client.compute(delayed_chunks[i, j])
                    fut_meta[fut] = (idx, i, j)
                    inflight.add(fut)
                    return True

                logger.info(
                    "Parallel chunk write: %d chunks, %d worker(s), up to %d in flight",
                    total_chunks, n_workers, max_inflight,
                )
                done = 0
                with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                    for _ in range(min(max_inflight, total_chunks)):
                        if not _submit_next():
                            break
                    for fut in inflight:
                        idx, i, j = fut_meta.pop(fut)
                        try:
                            chunk_data = fut.result()
                            chunk_files.append(_persist_chunk(idx, i, j, chunk_data))
                            del chunk_data
                        except Exception as e:
                            logger.error(f"Failed to process chunk {i},{j}: {e}")
                            raise
                        finally:
                            del fut
                        done += 1
                        if done % 10 == 0:
                            try:
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass
                        pbar.update(1)
                        pbar.set_postfix({"saved": f"{len(chunk_files)}"})
                        _submit_next()
            else:
                # 直列フォールバック（distributed クライアントが無い場合）
                idx = 0
                with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                    for i in range(n_rows):
                        for j in range(n_cols):
                            try:
                                chunk_data = delayed_chunks[i, j].compute()
                                chunk_files.append(_persist_chunk(idx, i, j, chunk_data))
                                del chunk_data
                            except Exception as e:
                                logger.error(f"Failed to process chunk {i},{j}: {e}")
                                raise
                            idx += 1
                            if idx % 10 == 0:
                                cp.get_default_memory_pool().free_all_blocks()
                            pbar.update(1)
                
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
            
            if use_cog_driver:
                result = gdal.Translate(
                    str(dst),
                    str(vrt_file),
                    format="COG",
                    creationOptions=list(f"{k}={v}" for k, v in cog_options.items()),
                    callback=gdal_progress_callback
                )
                if result is None:
                    raise ValueError("COG conversion failed")
                result = None
                _ensure_cog_has_overviews(dst, cog_options)
            else:
                tmp_tif = Path(tmpdir) / "merged_tmp.tif"
                result = gdal.Translate(
                    str(tmp_tif),
                    str(vrt_file),
                    format="GTiff",
                    creationOptions=[
                        "TILED=YES",
                        "BLOCKXSIZE=512",
                        "BLOCKYSIZE=512",
                        "COMPRESS=ZSTD",
                        f"ZLEVEL={cog_options.get('LEVEL', '1')}",
                        "BIGTIFF=YES",
                        "NUM_THREADS=ALL_CPUS",
                    ],
                    callback=gdal_progress_callback
                )
                if result is None:
                    raise ValueError("Temporary GeoTIFF conversion failed")
                result = None
                build_cog_with_overviews(tmp_tif, dst, cog_options)
            
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
        tiff_options = {
            k: v for k, v in cog_options.items()
            if k not in [
                'OVERVIEWS',
                'OVERVIEW_COMPRESS',
                'OVERVIEW_COUNT',
                'OVERVIEW_RESAMPLING',
            ]
        }
        
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
    """旧版 GDAL 用: 一時 TIFF にoverviewを作成してからCOGへ変換"""
    # CPU数を取得して並列処理
    num_cpus = os.cpu_count() or 1

    _build_zstd_overviews(src, cog_options)
    _assert_has_overviews(src)
    logger.info("Converting overview-backed temporary TIFF to COG: %s", dst)

    translate_options = [
        f"COMPRESS={cog_options.get('COMPRESS', 'ZSTD')}",
        f"LEVEL={cog_options.get('LEVEL', '1')}",
        f"OVERVIEW_COMPRESS={cog_options.get('OVERVIEW_COMPRESS', 'ZSTD')}",
        f"BLOCKSIZE={cog_options.get('BLOCKSIZE', '512')}",
        "BIGTIFF=YES",
        f"NUM_THREADS={num_cpus}",
        "OVERVIEWS=FORCE_USE_EXISTING",
        f"OVERVIEW_RESAMPLING={cog_options.get('OVERVIEW_RESAMPLING', 'AVERAGE')}",
    ]
    
    # PREDICTORがある場合のみ追加
    if 'PREDICTOR' in cog_options:
        translate_options.append(f"PREDICTOR={cog_options['PREDICTOR']}")

    result = gdal.Translate(
        str(dst),
        str(src),
        format="COG",
        creationOptions=translate_options,
    )
    if result is None:
        raise ValueError(f"COG translate failed: {dst}")
    result = None
    _assert_has_overviews(dst)

###############################################################################
# 6. メインパイプライン
###############################################################################

def validate_inputs(src_cog: str):
    """入力パラメータの検証"""
    if not Path(src_cog).exists():
        raise FileNotFoundError(f"Input file not found: {src_cog}")


def _compute_rvi_global_stats_from_overview(
    src_cog: str,
    *,
    radii: List[int],
    weights: Optional[List[float]],
    pixel_size: float,
    sample_max: int = 2048,
) -> Optional[tuple]:
    """Estimate the RVI normalization scale from a decimated overview read.

    Striding the full-resolution Dask array forces every chunk (the entire
    dataset) to be read and copied to the GPU before any write progress is
    visible, which stalls on very large rasters.  A decimated rasterio read uses
    the COG overview pyramid and returns a representative full-extent sample at a
    tiny fraction of the cost -- mirroring the tile backend's strategy.
    """
    try:
        from ..algorithms._impl_rvi import compute_rvi_efficient_block
        from ..algorithms._normalization import rvi_stat_func
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("RVI overview stats helpers unavailable: %s", exc)
        return None

    if not radii:
        return None

    try:
        with rasterio.open(src_cog) as src:
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale))
            sample_h = max(128, int(src.height / scale))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)
            nodata = src.nodata

        if nodata is not None and not np.isnan(float(nodata)):
            sample = np.where(
                np.isclose(sample, float(nodata), rtol=0.0, atol=1e-6),
                np.nan,
                sample,
            ).astype(np.float32, copy=False)

        sample_pixel_size = float(pixel_size) * float(scale)
        scaled_radii = [max(1, int(round(float(r) / scale))) for r in radii]

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        rvi_sample = compute_rvi_efficient_block(
            sample_gpu,
            radii=scaled_radii,
            weights=weights,
            pixel_size=sample_pixel_size,
        )
        stats = rvi_stat_func(rvi_sample)
        if not stats or not np.isfinite(float(stats[0])) or float(stats[0]) <= 1e-9:
            return None
        logger.info(
            "RVI global stats from overview: decimation=%.1fx, radii=%s -> %s, abs_p80=%.6f",
            scale,
            list(radii),
            scaled_radii,
            float(stats[0]),
        )
        return stats
    except Exception as exc:
        logger.warning(
            "Failed to compute RVI overview stats; falling back to window sampling: %s",
            exc,
        )
        return None




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

    # GDAL読み込み最適化（クラスタ生成前に設定 → spawnされるワーカーが継承）
    _configure_gdal_read_performance()
    _log_overview_availability(src_cog)

    # メモリ状況の確認
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.total / 2**30:.1f} GB total, "
                f"{mem.available / 2**30:.1f} GB available")

    # メモリフラクションの取得（algo_paramsから、なければデフォルト）
    memory_fraction = algo_params.pop('memory_fraction', None)
    cluster, client = make_cluster(memory_fraction)  # dask_cluster.py 直接呼び出し
    
    try:
        # チャンクサイズの自動決定
        if chunk is None:
            if is_zarr_path(src_cog):
                dem_probe = load_input_dataarray(src_cog, 1024)
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
        dem: xr.DataArray = load_input_dataarray(src_cog, chunk)
        
        logger.info(f"DEM shape: {dem.shape}, dtype: {dem.dtype}, "
                   f"chunks: {dem.chunks}")

        # アルゴリズム固有のデフォルトパラメータを取得
        default_params = algo.get_default_params()

        # パラメータの準備
        params = {
            **default_params,
            **algo_params,
            'show_progress': show_progress,
            'agg': agg
        }

        # NoData void filling is owned by the preprocessing command
        # (`python -m FujiShaderGPU.prepare`); the pipeline no longer fills holes.

        # 6-2) CuPy 配列へ変換（改善：メタデータ指定）
        gpu_arr: da.Array = dem.data.map_blocks(
            cp.asarray,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )

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
                terrain_stats = analyze_terrain_characteristics(gpu_arr, sample_ratio=0.01)
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

        # CLI passes explicit radii through run_pipeline's top-level radii
        # parameter, so restore it for non-RVI spatial algorithms.
        if algorithm != "rvi" and radii is not None:
            params['radii'] = radii
            logger.info(f"Setting radii for {algorithm}: {radii}")

        # RVI: derive the global normalization scale from a fast decimated
        # overview read instead of striding the full-resolution array (which
        # would read the entire dataset before any write progress is visible).
        if (
            algorithm == "rvi"
            and "global_stats" not in params
            and not is_zarr_path(src_cog)
        ):
            rvi_global_stats = _compute_rvi_global_stats_from_overview(
                src_cog,
                radii=params.get("radii"),
                weights=params.get("weights"),
                pixel_size=float(params.get("pixel_size", 1.0)),
            )
            if rvi_global_stats is not None:
                params["global_stats"] = rvi_global_stats

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
        # 6-5) xarray ラップ（座標構築）
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
        elif algorithm in ['lrm', 'rvi', 'fractal_anomaly']:
            # Signed terrain anomaly outputs map p80(abs(value)) to +/-1,
            # with overflow preserved for strong extrema.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "-1.5 to +1.5",
                "normal_range": "-1 to +1",
                "normal_percentile": "80",
                "data_type": "float32"
            }
        elif algorithm in [
            'atmospheric_scattering',
            'multiscale_terrain',
            'curvature',
            'visual_saliency',
            'ambient_occlusion',
            'openness',
            'scale_space_surprise',
            'multi_light_uncertainty',
        ]:
            # Unsigned analysis outputs map p80(value) to +1,
            # with overflow preserved for strong extrema.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1.5",
                "normal_range": "0 to 1",
                "normal_percentile": "80",
                "data_type": "float32"
            }
        else:
            # Display/stylized outputs keep their native display range.
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
        if is_zarr_path(str(dst_path)):
            logger.info("Writing output as Zarr: %s", dst_path)
            write_zarr_output(result_da, dst_path, show_progress=show_progress)
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

