# -*- coding: utf-8 -*-
"""
FujiShaderGPU/core/dask_processor.py
Dask-CUDA地形解析処理のコア実装
"""

###############################################################################
# 依存ライブラリ
###############################################################################
from __future__ import annotations

import os, sys, time, subprocess, gc, warnings, logging, rasterio, psutil, GPUtil, rmm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from osgeo import gdal
import cupy as cp
import numpy as np
import dask.array as da
from dask_cuda import LocalCUDACluster
from dask import config as dask_config
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import Client, get_client
from rasterio.windows import Window
import xarray as xr
import rioxarray as rxr
from tqdm.auto import tqdm

# アルゴリズムのインポート
from ..config.gpu_config_manager import _gpu_config_manager
from ..algorithms.dask_algorithms import ALGORITHMS, determine_optimal_radii, determine_optimal_sigmas

# ロギング設定
logger = logging.getLogger(__name__)

###############################################################################
# 1. Dask‑CUDA クラスタ（改善版）
###############################################################################

def get_optimal_chunk_size(gpu_memory_gb: float = 40, gpu_name: str = "") -> int:
    gpu_type = _gpu_config_manager.detect_gpu_type(gpu_memory_gb, gpu_name)
    preset = _gpu_config_manager.get_preset(gpu_type)
    return preset["chunk_size"]
    
def get_rmm_cupy_allocator():
    """RMM CuPy allocatorを取得する共通関数"""
    try:
        # RMM ≥22.12 の新しいimportパス
        from rmm.allocators.cupy import rmm_cupy_allocator
        return rmm_cupy_allocator
    except ImportError:
        # 旧バージョンのfallback
        rmm_cupy_allocator = getattr(rmm, "rmm_cupy_allocator", None)
        if rmm_cupy_allocator is None:
            raise RuntimeError(
                "RMM のバージョンが古いかインストールが不完全です。"
                "'pip install --extra-index-url https://pypi.nvidia.com rmm-cu12==25.06.*' "
                "で再インストールしてください。"
            )
        return rmm_cupy_allocator

def get_pixel_size_from_dataarray(dem: xr.DataArray) -> float:
    """DataArrayから pixel_size を取得する共通関数"""
    try:
        x_res = abs(float(dem.rio.resolution()[0]))
        y_res = abs(float(dem.rio.resolution()[1]))
        pixel_size = (x_res + y_res) / 2
        logger.info(f"Detected pixel size: {pixel_size:.2f}")
        return pixel_size
    except Exception as e:
        logger.warning(f"Could not determine pixel size: {e}, using 1.0")
        return 1.0
    
def cleanup_gpu_memory():
    """GPU メモリを解放する共通関数"""
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        # メモリ使用量を記録してからクリーンアップ
        used_bytes = mempool.used_bytes()
        
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        if used_bytes > 0:
            logger.debug(f"GPU memory cleaned up: {used_bytes / (1024**3):.2f} GB freed")
    except Exception as e:
        logger.warning(f"GPU memory cleanup failed: {e}")
        # CuPyが初期化されていない場合は無視
        if "CuPy is not initialized" not in str(e):
            raise

def _copy_crs_info(source_da: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
    """DataArray間でCRS情報をコピーする"""
    if hasattr(source_da, 'rio') and source_da.rio.crs is not None:
        target_da.rio.write_crs(source_da.rio.crs, inplace=True)
    return target_da

def make_cluster(memory_fraction: float = 0.6) -> Tuple[LocalCUDACluster, Client]:
    try:
        # 設定マネージャから環境設定を取得
        config_mgr = _gpu_config_manager
        
        # Colab環境の検出と設定の取得
        is_colab = config_mgr.is_colab()
        if is_colab:
            env_config = config_mgr.get_environment_config("colab")
            memory_fraction = min(memory_fraction, env_config.get("memory_fraction_limit", 0.5))
            death_timeout = env_config.get("death_timeout", "60s")
            interface = env_config.get("interface", "lo")
            logger.info("Google Colab環境を検出: メモリ設定を調整")
            # Colabではスレッドベースワーカーを使用
            use_processes = False
        else:
            death_timeout = "30s"
            interface = None
            # 通常環境ではプロセスベース（ただしforkを強制）
            use_processes = True
            # forkメソッドを強制（Linux環境でのみ有効）
            import multiprocessing
            if hasattr(multiprocessing, 'set_start_method'):
                try:
                    multiprocessing.set_start_method('fork', force=True)
                    logger.info("Multiprocessing start method set to 'fork'")
                except RuntimeError:
                    # 既に設定されている場合は無視
                    pass
        
        # GPU情報を取得（gpu_config_managerを使用）
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.warning("No GPU detected by GPUtil, trying CuPy detection")
                # CuPyから直接GPU情報を取得
                try:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    gpu_memory_gb = meminfo[1] / (1024**3)
                    gpu_name = ""
                    logger.info(f"GPU memory detected via CuPy: {gpu_memory_gb:.1f} GB")
                except:
                    gpu_memory_gb = 40  # 最終的なフォールバック
                    gpu_name = ""
            else:
                gpu_memory_gb = gpus[0].memoryTotal / 1024
                gpu_name = gpus[0].name
                logger.info(f"GPU detected: {gpu_name}, Memory: {gpu_memory_gb:.1f} GB")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, using default configuration")
            gpu_memory_gb = 40  # デフォルトA100想定
            gpu_name = ""

        # 実際に利用可能なメモリを確認
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            available_gb = meminfo[0] / (1024**3)
            logger.info(f"Available GPU memory: {available_gb:.1f} GB")
        except:
            available_gb = gpu_memory_gb * 0.8  # フォールバック
        
        # プリセットを取得
        gpu_type = config_mgr.detect_gpu_type(gpu_memory_gb, gpu_name)
        preset = config_mgr.get_preset(gpu_type)
        rmm_size = preset["rmm_pool_size_gb"]

        # 実際に利用可能なメモリに基づいて調整
        if is_colab:
            # Colab環境では非常に控えめに（利用可能メモリの40%）
            max_rmm_size = int(available_gb * 0.4)
            # さらに最大値を制限
            max_rmm_size = min(max_rmm_size, 16)  # 最大16GBに制限
            rmm_size = min(rmm_size, max_rmm_size)
            logger.info(f"Colab environment: Limiting RMM pool to {rmm_size}GB (available: {available_gb:.1f}GB)")
        else:
            # 通常環境でも利用可能メモリの60%を上限とする
            max_rmm_size = int(available_gb * 0.6)
            if rmm_size > max_rmm_size:
                rmm_size = max_rmm_size
                logger.info(f"Adjusting RMM pool size to {rmm_size}GB based on available memory")

        # 最小値も設定（最低2GB）
        rmm_size = max(2, rmm_size)

        # さらに安全のため、既に使用されているメモリを考慮
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            free_gb = meminfo[0] / (1024**3)
            # 現在の空きメモリの80%を上限とする
            rmm_size = min(rmm_size, int(free_gb * 0.8))
            logger.info(f"Final RMM pool size: {rmm_size}GB (free memory: {free_gb:.1f}GB)")
        except:
            pass

        # Worker の terminate 閾値は Config で与える
        # ────────── メモリ管理パラメータを Config で一括設定 ──────────
        dask_config.set({
            # ■ メモリしきい値
            # 環境変数で設定されていない場合のみデフォルト値を設定
            "distributed.worker.memory.target": float(os.environ.get("DASK_DISTRIBUTED__WORKER__MEMORY__TARGET", "0.70")),
            "distributed.worker.memory.spill": float(os.environ.get("DASK_DISTRIBUTED__WORKER__MEMORY__SPILL", "0.75")),
            "distributed.worker.memory.pause": float(os.environ.get("DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE", "0.85")),
            "distributed.worker.memory.terminate": float(os.environ.get("DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE", "0.95")),

            # ■ イベントループ警告を 15 s まで黙らせる
            "distributed.admin.tick.limit": "15s",

            # ■ スケジューラの同期を改善
            "distributed.scheduler.work-stealing": True,

            # ■ Colab環境でのセマフォリーク対策
            # ワーカーを定期的に再起動（最も効果的）
            "distributed.worker.lifetime.duration": "10 minutes" if is_colab else None,
            "distributed.worker.lifetime.stagger": "1 minute" if is_colab else None,
            "distributed.worker.lifetime.restart": True if is_colab else False,
        })

        # distributed.core の INFO スパムを抑制
        logging.getLogger("distributed.core").setLevel(logging.WARNING)
        # distributed.scheduler のエラーも抑制
        logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)

        cluster = LocalCUDACluster(
            device_memory_limit="0.95",
            jit_unspill=True,
            threads_per_worker=1,
            silence_logs=logging.WARNING,
            death_timeout=death_timeout,
            interface=interface,
            enable_cudf_spill=True,
            local_directory='/tmp',
            processes=use_processes,
        )

        client = Client(cluster)
        try:
            rmm.reinitialize(
                pool_allocator=True,
                managed_memory=False,
                initial_pool_size=f"{rmm_size}GB",
            )
            rmm_cupy_allocator = get_rmm_cupy_allocator()
            cp.cuda.set_allocator(rmm_cupy_allocator)

            def _enable_rmm_on_worker():
                rmm.reinitialize(pool_allocator=True, managed_memory=False)
                rmm_cupy_allocator = get_rmm_cupy_allocator()
                cp.cuda.set_allocator(rmm_cupy_allocator)

            client.run(_enable_rmm_on_worker)

            logger.info("CuPy allocator switched to RMM – GPU memory is now managed by Dask")
        except ImportError:
            logger.warning("rmm not found – falling back to default CuPy allocator")

        logger.info(f"Dask dashboard: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to create cluster: {e}")
        raise

###############################################################################
# 3. 地形解析によるパラメータ自動決定
###############################################################################

def analyze_terrain_characteristics(dem_arr: da.Array, sample_ratio: float = 0.01, 
                                    include_fft: bool = False) -> Dict[str, Any]:
    """地形の特性を統合的に解析"""
    sample = None
    fft = None
    power = None
    try:
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
            stats['dominant_scales'] = []
            stats['dominant_freqs'] = []
            stats['mean_curvature'] = 0.0
    finally:
        # CuPyメモリの明示的な解放
        if sample is not None and hasattr(sample, '__cuda_array_interface__'):
            del sample
        if fft is not None:
            del fft
        if power is not None:
            del power
        cp.get_default_memory_pool().free_all_blocks()
    return stats

###############################################################################
# 4. 直接 COG 出力 (GDAL ≥ 3.8) - 改善版
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
    
    # Dask配列の場合は、実際のdtypeを正しく取得
    if hasattr(data.data, 'dtype'):
        dtype_str = str(data.data.dtype)
    else:
        dtype_str = str(data.dtype)
        
    # NumPy dtypeオブジェクトの場合は名前を取得
    if hasattr(dtype_str, 'name'):
        dtype_str = dtype_str.name

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
                            
                        def _posttask(self, key, result, dsk, state, worker_id):
                            self.tqdm.update(1)
                            
                        def _finish(self, dsk, state, failed):
                            self.tqdm.close()
                    
                    with TqdmCallback():
                        computed_data = data.compute()
                else:
                    computed_data = data.compute()
                
                # 計算済みデータをxarrayに戻す（computed_dataをcomputed_daに変換）
                computed_da = xr.DataArray(
                    computed_data,
                    dims=data.dims,
                    coords=data.coords,
                    attrs=data.attrs
                )
                computed_da = _copy_crs_info(data, computed_da)
                
                # COG書き込み
                # NoData値を設定（int16の場合は-32768）
                if computed_da.dtype == np.int16:
                    computed_da.rio.write_nodata(-32768, inplace=True)
                logger.info("Writing to COG...")
                computed_da.rio.to_raster(
                    dst,
                    driver="COG",
                    **cog_options,
                )
            
            size_mb = os.path.getsize(dst) / 2**20
            logger.info(f"✔ COG written: {dst} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.warning(f"COG driver failed: {e}, falling back to gdal_translate")
            _fallback_cog_write(data, dst, cog_options)
    else:
        logger.warning(f"GDAL {major}.{minor} < 3.8, using fallback method")
        _fallback_cog_write(data, dst, cog_options)

def _write_cog_da_chunked_impl(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """大規模データ用のチャンク単位書き込み実装（改善版：windowed writing）"""
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    
    # Dask配列の場合は、実際のdtypeを正しく取得
    if hasattr(data.data, 'dtype'):
        dtype_str = str(data.data.dtype)
    else:
        dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    # メモリ制限を考慮してチャンクサイズを動的に調整（既存のコードと同じ）
    meminfo = cp.cuda.runtime.memGetInfo()
    available_memory = meminfo[0] / (1024**3)  # 空きメモリ
    total_memory = meminfo[1] / (1024**3)      # 総メモリ
    if available_memory < 10:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        try:
            client = get_client()
            client.run(lambda: gc.collect())
        except:
            pass

    # DRAMの空き容量を確認してプリフェッチを有効化（既存のコードと同じ）
    mem_info = psutil.virtual_memory()
    available_ram_gb = mem_info.available / (1024**3)
    
    if available_ram_gb > 20:
        logger.info(f"Enabling chunk prefetching (available DRAM: {available_ram_gb:.1f}GB)")
        prefetch_config = {
            "optimization.fuse.active": False,
            "distributed.worker.memory.pause": 0.90,
            "distributed.worker.memory.spill": 0.95,
        }
    else:
        logger.info(f"Prefetching disabled (available DRAM: {available_ram_gb:.1f}GB < 20GB)")
        prefetch_config = {}
    
    with dask_config.set(prefetch_config):
        # ======== ここから変更 ========
        # 一時ファイルを使用（COGドライバが使えない場合のフォールバック用）
        tmp_file = dst.with_suffix('.tmp.tif') if not use_cog_driver else dst
        
        try:
            # rasterioでファイルを書き込みモードで開く
            profile = {
                'driver': 'COG' if use_cog_driver else 'GTiff',
                'dtype': data.dtype,
                'width': data.shape[1],
                'height': data.shape[0],
                'count': 1,
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': cog_options.get('COMPRESS', 'ZSTD'),
                'photometric': 'MINISBLACK',
            }
            
            # CRS情報があれば追加
            if hasattr(data, 'rio') and data.rio.crs is not None:
                profile['crs'] = data.rio.crs
                profile['transform'] = data.rio.transform()
            
            # COGオプションの追加
            if use_cog_driver:
                # COGドライバーの場合、creation_optionsとして設定
                creation_options = []
                for key, value in cog_options.items():
                    if key in ['COMPRESS', 'LEVEL', 'PREDICTOR', 'BIGTIFF', 'NUM_THREADS', 
                            'BLOCKSIZE', 'OVERVIEWS', 'OVERVIEW_RESAMPLING']:
                        creation_options.append(f"{key}={value}")
                profile['options'] = creation_options
            
            # Dask配列のチャンクを取得
            if hasattr(data.data, 'to_delayed'):
                delayed_chunks = data.data.to_delayed()
                
                if not hasattr(data, 'chunks') or data.chunks is None:
                    logger.warning("No chunk information found, falling back to regular processing")
                    _write_cog_da_original(data, dst, show_progress)
                    return
                
                total_chunks = delayed_chunks.shape[0] * delayed_chunks.shape[1]
                
                with rasterio.open(tmp_file, 'w', **profile) as dst_dataset:
                    chunk_idx = 0
                    # Colab環境でのセマフォクリーンアップ用
                    is_colab = _gpu_config_manager.is_colab()
                    # 478-497行目あたりの修正案
                    with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk", 
                            disable=not show_progress) as pbar:
                        for i in range(delayed_chunks.shape[0]):
                            for j in range(delayed_chunks.shape[1]):
                                try:
                                    # チャンクを計算
                                    chunk_data = delayed_chunks[i, j].compute()
                                    
                                    # チャンクの開始位置を計算
                                    # data.dataがdask配列の場合、chunksは必ず存在する
                                    if hasattr(data.data, 'chunks'):
                                        y_chunks = data.data.chunks[0]
                                        x_chunks = data.data.chunks[1]
                                    else:
                                        # chunksがない場合は全体を1チャンクとして扱う
                                        y_chunks = (data.shape[0],)
                                        x_chunks = (data.shape[1],)
                                    y_start = sum(y_chunks[:i]) if i > 0 else 0
                                    x_start = sum(x_chunks[:j]) if j > 0 else 0
                                    
                                    # rasterioのWindowを作成
                                    window = Window(
                                        col_off=x_start,
                                        row_off=y_start,
                                        width=chunk_data.shape[1],
                                        height=chunk_data.shape[0]
                                    )
                                    
                                    # チャンクをウィンドウに書き込む
                                    dst_dataset.write(chunk_data, 1, window=window)
                                    
                                    # メモリ使用量を取得
                                    mem_used = cp.get_default_memory_pool().used_bytes() / (1024**3)
                                    
                                    # メモリ解放
                                    del chunk_data
                                    chunk_idx += 1
                                    
                                    # 10チャンクごとに軽量クリーンアップ
                                    if chunk_idx % 10 == 0:
                                        cp.get_default_memory_pool().free_all_blocks()
                                        # Colabでは追加のクリーンアップ
                                        if is_colab and chunk_idx % 50 == 0:
                                            # ガベージコレクション強制
                                            gc.collect()
                                            # ワーカーのメモリクリーンアップ
                                            try:
                                                client = get_client()
                                                client.run(gc.collect)
                                                # セマフォの手動クリーンアップ（可能な場合）
                                                def cleanup_semaphores():
                                                    gc.collect()
                                                    # resource_trackerの内部をクリーンアップ
                                                    return True
                                                client.run(cleanup_semaphores)
                                            except:
                                                pass
                                    
                                    # 進捗更新（メモリ使用量も表示）
                                    pbar.update(1)
                                    pbar.set_postfix({
                                        "completed": f"{chunk_idx}/{total_chunks}",
                                        "gpu_mem_gb": f"{mem_used:.1f}"
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to write chunk at position ({i}, {j}): {e}")
                                    # メモリ不足の可能性があるため、強制的にクリーンアップ
                                    cp.get_default_memory_pool().free_all_blocks()
                                    raise
                
                # COGドライバが使えない場合は、オーバービューを追加
                if not use_cog_driver:
                    logger.info("Building overviews...")
                    build_cog_with_overviews(tmp_file, dst, cog_options)
                else:
                    logger.info(f"Successfully created COG: {dst}")
                    
            else:
                # Dask配列でない場合は通常の処理にフォールバック
                logger.info("Data is not chunked with Dask, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return
                
        finally:
            # 一時ファイルのクリーンアップ（COGドライバが使えない場合）
            if not use_cog_driver and tmp_file != dst:
                if tmp_file.exists():
                    try:
                        # 最終的なファイルサイズをログ出力
                        size_mb = tmp_file.stat().st_size / (1024**2)
                        logger.info(f"COG created: {dst} ({size_mb:.1f} MB)")
                        tmp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file: {e}")

def write_cog_da_chunked(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """COG書き出し（メモリ容量に応じて自動選択）"""
    total_gb = data.nbytes / (1024**3)
    
    # 設定マネージャから環境設定を取得
    config_mgr = _gpu_config_manager
    chunk_config = config_mgr.get_environment_config("chunk_threshold")

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
    except:
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
            attrs=data.attrs
        )
        computed_da = _copy_crs_info(data, computed_da)
        
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

def validate_inputs(src_cog: str, sigmas: Optional[List[float]] = None):
    """入力パラメータの検証"""
    if not Path(src_cog).exists():
        raise FileNotFoundError(f"Input file not found: {src_cog}")
    
    if sigmas is not None:
        if not sigmas:
            raise ValueError("At least one sigma value must be provided")
        
        if any(s <= 0 for s in sigmas):
            raise ValueError("All sigma values must be positive")

def run_pipeline(
    src_cog: str,
    dst_cog: str,
    algorithm: str = "rvi",
    sigmas: Optional[List[float]] = None,
    radii: Optional[List[int]] = None,
    agg: str = "mean",
    chunk: Optional[int] = None,
    show_progress: bool = True,
    auto_sigma: bool = False,
    auto_radii: bool = True,
    **algo_params
):
    """改善されたメインパイプライン"""
    # アルゴリズムの確認
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    
    algo = ALGORITHMS[algorithm]
    
    # アルゴリズム固有の必須パラメータチェック
    required_params = getattr(algo, 'required_params', [])
    missing_params = [p for p in required_params if p not in algo_params]
    if missing_params:
        raise ValueError(f"Missing required parameters for {algorithm}: {missing_params}")

    # 入力検証（sigmasはNoneでもOK）
    validate_inputs(src_cog, sigmas)
    
    # メモリ状況の確認
    mem = psutil.virtual_memory()
    logger.info(f"System memory: {mem.total / 2**30:.1f} GB total, "
                f"{mem.available / 2**30:.1f} GB available")
    
    # メモリフラクションの取得（algo_paramsから、なければデフォルト）
    memory_fraction = algo_params.pop('memory_fraction', 0.8)
    cluster, client = make_cluster(memory_fraction)
    
    try:
        # ===== 統計ログの収集開始 ===== #
        # GPU情報の取得
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {
                    "gpu_name": gpus[0].name,
                    "gpu_memory_gb": round(gpus[0].memoryTotal / 1024, 1),
                    "gpu_driver": gpus[0].driver
                }
            else:
                meminfo = cp.cuda.runtime.memGetInfo()
                gpu_info = {
                    "gpu_name": "Unknown",
                    "gpu_memory_gb": round(meminfo[1] / (1024**3), 1),
                    "gpu_driver": "Unknown"
                }
        except:
            gpu_info = {"gpu_name": "Unknown", "gpu_memory_gb": 0, "gpu_driver": "Unknown"}
        
        # GPU設定の取得
        gpu_type = _gpu_config_manager.detect_gpu_type(
            gpu_info["gpu_memory_gb"], 
            gpu_info["gpu_name"]
        )
        gpu_preset = _gpu_config_manager.get_preset(gpu_type)
        
        # チャンクサイズの自動決定
        if chunk is None:
            chunk = get_optimal_chunk_size(gpu_info["gpu_memory_gb"], gpu_info["gpu_name"])
            logger.info(f"Using GPU-optimized chunk size: {chunk}x{chunk}")
            
        # 入力ファイル情報
        with rasterio.open(src_cog) as src:
            input_info = {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else "None",
                "dtype": str(src.dtypes[0])
            }
        
        # アルゴリズムパラメータの整理
        algo_params_log = {
            "algorithm": algorithm,
            "agg": agg,
            "auto_sigma": auto_sigma,
            "auto_radii": auto_radii,
            **{k: str(v) for k, v in algo_params.items()}  # 全パラメータを文字列化
        }
        if sigmas is not None:
            algo_params_log["sigmas"] = str(sigmas)
        if radii is not None:
            algo_params_log["radii"] = str(radii)
        
        # 統計ログエントリの構築
        import json
        import datetime
        
        stats_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "env": {
                "os": os.name,
                "python": sys.version.split()[0],
                "cpu_count": os.cpu_count(),
                "ram_gb": round(mem.total / (1024**3), 1),
                "ram_available_gb": round(mem.available / (1024**3), 1),
                **gpu_info,
                "gpu_type_detected": gpu_type,
                "is_colab": _gpu_config_manager.is_colab()
            },
            "performance": {
                "chunk_size": chunk,
                "memory_fraction": memory_fraction,
                "rmm_pool_gb": gpu_preset["rmm_pool_size_gb"],
                "rmm_pool_fraction": gpu_preset["rmm_pool_fraction"],
                "preset_chunk_size": gpu_preset["chunk_size"]
            },
            "algorithm": algo_params_log,
            "io": {
                "input": os.path.basename(src_cog),
                "output": os.path.basename(dst_cog),
                "input_size": f"{input_info['width']}x{input_info['height']}",
                "input_pixels": input_info['width'] * input_info['height'],
                "input_gb": round((input_info['width'] * input_info['height'] * 4) / (1024**3), 2)
            }
        }
        
        # 1行のJSON形式でログ出力
        logger.info(f"STATS_LOG: {json.dumps(stats_log, ensure_ascii=False)}")
        # ===== 統計ログの収集終了 ===== #

        # チャンクサイズの自動決定
        if chunk is None:
            chunk = get_optimal_chunk_size()
            logger.info(f"Using GPU-optimized chunk size: {chunk}x{chunk}")
        
        # 6‑1) DEM 遅延ロード
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
            dem: xr.DataArray = (
                rxr.open_rasterio(
                    src_cog, 
                    masked=True, 
                    chunks={"y": chunk, "x": chunk},
                    lock=False,  # 並列読み込みを許可
                )
                .squeeze()
                .astype("float32")
            )
        
        logger.info(f"DEM shape: {dem.shape}, dtype: {dem.dtype}, "
                   f"chunks: {dem.chunks}")
        
        # 6‑2) CuPy 配列へ変換（改善：メタデータ指定）
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
        
        # 6‑2.5) 自動決定（RVIアルゴリズムの場合）
        if algorithm == "rvi":
            # ピクセルサイズを先に取得（座標系から）
            pixel_size = get_pixel_size_from_dataarray(dem)
            
            # パラメータに設定
            params['pixel_size'] = pixel_size

            # terrain_statsを初期化
            terrain_stats = None

            # terrain_statsを一度だけ計算して再利用
            if terrain_stats is None and (auto_radii or auto_sigma):
                terrain_stats = analyze_terrain_characteristics(gpu_arr, sample_ratio=0.01, include_fft=auto_sigma)
                terrain_stats['pixel_size'] = pixel_size
            
            # 新しい効率的なモードをデフォルトに
            if radii is None and sigmas is None and auto_radii:
                logger.info("Analyzing terrain for automatic radii determination...")
                
                # 最適な半径を決定
                radii, weights = determine_optimal_radii(terrain_stats)
                
                logger.info(f"Terrain analysis results:")
                logger.info(f"  - Elevation range: {terrain_stats['elevation_range']:.1f} m")
                logger.info(f"  - Mean slope: {terrain_stats['mean_slope']:.3f}")
                logger.info(f"  - Auto-determined radii: {radii} pixels")
                logger.info(f"  - Weights: {[f'{w:.2f}' for w in weights]}")
                
                # パラメータに設定
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = weights
                params['terrain_stats'] = terrain_stats  # terrain_statsも渡す
                
            elif sigmas is not None or (sigmas is None and auto_sigma):
                # 従来のsigmaモード（互換性）
                params['mode'] = 'sigma'
                
                if sigmas is not None:
                    params['sigmas'] = sigmas
                    params['agg'] = agg
                    logger.warning("Using legacy sigma mode. Consider switching to radius mode for better performance.")
                else:
                    # auto_sigmaがTrueの場合
                    logger.info("Analyzing terrain for automatic sigma determination...")
                    
                    # 最適なsigmaを決定
                    sigmas = determine_optimal_sigmas(terrain_stats)
                    
                    logger.info(f"Terrain analysis results:")
                    logger.info(f"  - Elevation range: {terrain_stats['elevation_range']:.1f} m")
                    logger.info(f"  - Mean slope: {terrain_stats['mean_slope']:.3f}")
                    logger.info(f"  - Detected scales: {terrain_stats.get('dominant_scales', [])}")
                    logger.info(f"  - Auto-determined sigmas: {sigmas}")
                    
                    # パラメータに設定
                    params['sigmas'] = sigmas
                    params['agg'] = agg
                
            elif radii is not None:
                # 手動指定の半径モード
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = algo_params.get('weights', None)
            
            else:
                raise ValueError("Either provide radii/sigmas or enable auto_radii/auto_sigma")

        # 多くのアルゴリズムでピクセルサイズが必要（RVI以外の場合）
        elif algorithm != "rvi" and ('pixel_size' not in params or params['pixel_size'] == 1.0):
            params['pixel_size'] = get_pixel_size_from_dataarray(dem)

        # 追加：fractal_anomaly用のradii処理
        if algorithm == "fractal_anomaly" and radii is not None:
            params['radii'] = radii
            logger.info(f"Setting radii for fractal_anomaly: {radii}")
        
        # 6-3) アルゴリズム実行（run_pipeline内）
        logger.info(f"Computing {algorithm} with parameters: {params}")

        # アルゴリズムを適用（遅延評価）
        result_gpu: da.Array = algo.process(gpu_arr, **params)

        # 6-4) GPU→CPU 戻し（改善：明示的なdtype）
        # アルゴリズムが返すdtypeを保持
        if hasattr(result_gpu, 'dtype'):
            output_dtype = result_gpu.dtype
        else:
            output_dtype = cp.float32  # フォールバック
            
        # NumPy型に変換
        if output_dtype == cp.int16:
            numpy_dtype = "int16"
        elif output_dtype == cp.int32:
            numpy_dtype = "int32"  
        elif output_dtype == cp.uint16:
            numpy_dtype = "uint16"
        elif output_dtype == cp.uint32:
            numpy_dtype = "uint32"
        elif output_dtype == cp.float64:
            numpy_dtype = "float64"
        else:
            numpy_dtype = "float32"  # デフォルト

        result_cpu = result_gpu.map_blocks(
            cp.asnumpy, 
            dtype=numpy_dtype,
            meta=cp.empty((0, 0), dtype=output_dtype).get()
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
            logger.info(f"Large dataset ({total_gb:.1f} GB) – skip persist; "
                        "chunked writer will stream-compute each tile")
    
        # 6‑5) xarray ラップ（改善：座標構築の簡略化）
        if agg == "stack" and 'sigmas' in params and params['sigmas'] is not None:
            dims = ("scale", *dem.dims)
            coords = {"scale": params['sigmas'], **dem.coords}
        else:
            dims = dem.dims
            coords = dem.coords
        
        # アルゴリズムによって適切なデータ範囲を設定
        if algorithm in ['hillshade']:
            # Hillshadeは0-255のまま
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0-255",
                "data_type": "uint8-like"
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
        elif algorithm in ['tpi', 'lrm', 'rvi']:
            # TPI、LRM、RVIは-1から+1
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
        
        # CRS情報を元のDEMから引き継ぐ
        if hasattr(dem, 'rio') and dem.rio.crs is not None:
            result_da.rio.write_crs(dem.rio.crs, inplace=True)
        
        # 6‑6) 出力 (直接 COG)
        write_cog_da_chunked(result_da, Path(dst_cog), show_progress=show_progress)
        logger.info("Pipeline completed successfully!")
        gc.collect()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # GPUメモリのクリーンアップ
        cleanup_gpu_memory()

        # より確実なクリーンアップ
        try:
            # ワーカーのメモリを強制的にクリア
            def clear_worker_memory():
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
