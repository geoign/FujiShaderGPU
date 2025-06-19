# -*- coding: utf-8 -*-
"""
FujiShaderGPU/core/dask_processor.py
Dask-CUDA地形解析処理のコア実装
"""

###############################################################################
# 依存ライブラリ
###############################################################################
from __future__ import annotations

import os, sys, time, subprocess, gc, warnings, logging, rasterio, psutil, GPUtil, tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from osgeo import gdal
import cupy as cp
import dask.array as da
from dask_cuda import LocalCUDACluster
from dask import config as dask_config
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import Client, get_client
import xarray as xr
import rioxarray as rxr
from tqdm.auto import tqdm


# アルゴリズムのインポート
try:
    from ..algorithms.dask_algorithms import ALGORITHMS
except ImportError:
    # algorithms/dask_algorithms.py が存在しない場合の仮の定義
    ALGORITHMS = {}
    logging.warning("dask_algorithms module not found. No algorithms available.")

# ロギング設定
logger = logging.getLogger(__name__)

###############################################################################
# 1. Dask‑CUDA クラスタ（改善版）
###############################################################################

def get_optimal_chunk_size(gpu_memory_gb: float = 40) -> int:
    """GPU メモリサイズに基づいて最適なチャンクサイズを計算"""
    # 経験的な計算式：利用可能メモリの約1/10をチャンクに割り当て
    base_chunk = int((gpu_memory_gb * 1024) ** 0.5 * 15)
    # 512の倍数に丸める（COGブロックサイズとの整合性）
    if gpu_memory_gb >= 40:  # A100
        return min(16384, (base_chunk // 512) * 512)  # 最大8192に拡大
    else:
        return min(8192, (base_chunk // 512) * 512)  # その他は4096に拡大

def make_cluster(memory_fraction: float = 0.6) -> Tuple[LocalCUDACluster, Client]:
    try:
        # Google Colab環境の検出
        is_colab = 'google.colab' in sys.modules
        
        if is_colab:
            # Colabではより保守的な設定
            memory_fraction = min(memory_fraction, 0.5)
            logger.info("Google Colab環境を検出: メモリ設定を調整")
        
        # GPU情報を取得
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory_gb = gpus[0].memoryTotal / 1024
            logger.info(f"GPU detected: {gpus[0].name}, Memory: {gpu_memory_gb:.1f} GB")
        else:
            gpu_memory_gb = 40  # デフォルトA100想定

        # 実際に利用可能なメモリを確認
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            available_gb = meminfo[0] / (1024**3)
            logger.info(f"Available GPU memory: {available_gb:.1f} GB")
        except:
            available_gb = gpu_memory_gb * 0.8  # フォールバック
        
        # RMMプールサイズを動的に調整
        if gpu_memory_gb >= 40:  # A100
            # 利用可能メモリの50%程度を確保（安全マージンを持たせる）
            rmm_size = min(int(available_gb * 0.7), 20)  # 最大20GBに制限
        else:
            rmm_size = min(int(available_gb * 0.6), 12)  # より保守的に
        
        # Worker の terminate 閾値は Config で与える
        # ────────── メモリ管理パラメータを Config で一括設定 ──────────
        dask_config.set({
            # ■ メモリしきい値
            "distributed.worker.memory.target":     0.70,  # 70 % で spill 開始
            "distributed.worker.memory.spill":      0.75,  # 75 % でディスク spill
            "distributed.worker.memory.pause":      0.85,  # 85 % でタスク一時停止
            "distributed.worker.memory.terminate":  0.95,  # 95 % でワーカ kill

            # ■ イベントループ警告を 15 s まで黙らせる
            "distributed.admin.tick.limit": "15s",
        })

        # ────────── distributed.core の INFO スパムを抑制 ──────────
        logging.getLogger("distributed.core").setLevel(logging.WARNING)

        cluster = LocalCUDACluster(
            device_memory_limit="0.95",  # 95%まで使用可能に
            jit_unspill=True,
            rmm_pool_size=f"{rmm_size}GB",
            threads_per_worker=1,
            silence_logs=logging.WARNING,
            death_timeout="60s" if is_colab else "30s",
            interface="lo" if is_colab else None,
            rmm_maximum_pool_size=f"{int(rmm_size * 1.2)}GB",  # より控えめに
            enable_cudf_spill=True,
            local_directory='/tmp',
        )

        client = Client(cluster)
        try:
            import rmm, cupy as cp
            rmm.reinitialize(
                pool_allocator=True,
                managed_memory=False,
                initial_pool_size=f"{rmm_size}GB",
            )
            # --- RMM ≥22.12 では allocator の import パスが変更 ---
            try:
                from rmm.allocators.cupy import rmm_cupy_allocator
            except ImportError:                    # 旧バージョン fallback
                rmm_cupy_allocator = getattr(rmm, "rmm_cupy_allocator", None)
            if rmm_cupy_allocator is None:
                raise RuntimeError(
                    "RMM のバージョンが古いかインストールが不完全です。"
                    "  'pip install --extra-index-url https://pypi.nvidia.com rmm-cu12==25.06.*' "
                    "で再インストールしてください。"
                )
            cp.cuda.set_allocator(rmm_cupy_allocator)

            def _enable_rmm_on_worker():
                import rmm, cupy as _cp
                rmm.reinitialize(pool_allocator=True, managed_memory=False)
                
                # 新しいRMM APIに対応
                try:
                    from rmm.allocators.cupy import rmm_cupy_allocator
                except ImportError:
                    rmm_cupy_allocator = getattr(rmm, "rmm_cupy_allocator", None)
                
                if rmm_cupy_allocator:
                    _cp.cuda.set_allocator(rmm_cupy_allocator)

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
# 3. 地形解析によるsigma自動決定
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
        else:
            dominant_scales = []
    return stats

def determine_optimal_radii(terrain_stats: dict) -> tuple[List[int], List[float]]:
    """地形統計に基づいて最適な半径を決定"""
    pixel_size = terrain_stats.get('pixel_size', 1.0)
    mean_slope = terrain_stats['mean_slope']
    std_dev = terrain_stats['std_dev']
    
    # 地形の複雑さ
    complexity = mean_slope * std_dev
    
    # 基本的な実世界距離（メートル）
    if complexity < 0.1:
        # 平坦な地形：大きめのスケール
        base_distances = [10, 40, 160, 640]
    elif complexity < 0.3:
        # 緩やかな地形：中程度のスケール
        base_distances = [5, 20, 80, 320]
    else:
        # 複雑な地形：細かいスケール
        base_distances = [2.5, 10, 40, 160]
    
    # ピクセル単位の半径に変換
    radii = []
    for dist in base_distances:
        radius = int(dist / pixel_size)
        # 現実的な範囲に制限（2-256ピクセル）
        radius = max(2, min(radius, 256))
        radii.append(radius)
    
    # 重複削除とソート
    radii = sorted(list(set(radii)))
    
    # 最大4つまでに制限
    if len(radii) > 4:
        # 対数的に分布
        indices = cp.logspace(0, cp.log10(len(radii)-1), 4).astype(int)
        radii = [radii[i] for i in indices]
    
    # 重みの決定（小さいスケールを重視）
    weights = []
    for i, r in enumerate(radii):
        weight = 1.0 / (i + 1)  # 1, 1/2, 1/3, 1/4
        weights.append(weight)
    
    # 正規化
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return radii, weights

def determine_optimal_sigmas(terrain_stats: dict, pixel_size: float = 1.0) -> List[float]:
    """地形統計に基づいて最適なsigma値を決定"""
    sigmas_set = set()  # setを使って重複を確実に排除
    
    # 1. 標高レンジと勾配に基づく基本スケール
    elev_range = terrain_stats['elevation_range']
    mean_slope = terrain_stats['mean_slope']
    
    # 地形の複雑さの指標
    terrain_complexity = mean_slope * terrain_stats['std_dev'] / (elev_range + 1e-6)
    
    # 基本スケール（地形の複雑さに応じて調整）
    if terrain_complexity < 0.1:  # 平坦な地形
        # base_scales = [100, 200, 400]  # この行を削除
        base_scales = [50, 100, 150]  # より小さい値に制限
    elif terrain_complexity < 0.3:  # 緩やかな地形
        base_scales = [50, 100, 200]
    else:  # 複雑な地形
        base_scales = [25, 50, 100, 200]
    
    # 2. FFT解析から得られたスケールを追加
    if terrain_stats['dominant_scales']:
        for scale in terrain_stats['dominant_scales']:
            if 10 < scale < 500:  # 現実的な範囲のスケールのみ
                # Gaussianフィルタのsigmaは、検出されたスケールの約1/4
                sigma_candidate = round(scale / 4, 0)  # 整数に丸める
                # if 5 <= sigma_candidate <= 500:  # この行を削除
                if 5 <= sigma_candidate <= 150:  # 最大値を150に制限
                    sigmas_set.add(sigma_candidate)
    
    # 3. 曲率に基づく微細スケール
    mean_curv = terrain_stats['mean_curvature']
    if mean_curv > 0.01:  # 曲率が高い場合は細かいスケールも追加
        sigmas_set.add(10)
    
    # 基本スケールを追加
    for scale in base_scales:
        # if 5 <= scale <= 500:  # この行を削除
        if 5 <= scale <= 150:  # 最大値を150に制限
            sigmas_set.add(scale)
    
    # setをリストに変換してソート
    sigmas = sorted(list(sigmas_set))
    
    # 最大3つまでに制限（メモリ効率のため、5→3に削減）
    if len(sigmas) > 3:
        indices = cp.linspace(0, len(sigmas)-1, 3).astype(int)
        sigmas = [sigmas[i] for i in indices]
    
    return [float(s) for s in sigmas]


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
                            
                        def _posttask(self, key, result, dsk, state, worker_id):
                            self.tqdm.update(1)
                            
                        def _finish(self, dsk, state, failed):
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
            logger.info(f"✔ COG written: {dst} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.warning(f"COG driver failed: {e}, falling back to gdal_translate")
            _fallback_cog_write(data, dst, cog_options)
    else:
        logger.warning(f"GDAL {major}.{minor} < 3.8, using fallback method")
        _fallback_cog_write(data, dst, cog_options)

def write_cog_da_chunked(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """大規模データ対応のチャンク単位COG書き出し"""   
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    # データサイズをチェック
    total_gb = data.nbytes / (1024**3)
    
    if total_gb > 50:  # 50GB以上の場合のみチャンク処理（閾値を上げる）
        logger.info(f"Large dataset ({total_gb:.1f} GB), using chunked writing")

        # ストリーミング書き込みを試みる
        try:
            # 直接書き込み
            logger.info("Attempting direct streaming write...")
            
            # ストリーミング用のCOGオプションを準備
            streaming_options = cog_options.copy()
            # BLOCKSIZEが設定されている場合は削除（個別に指定するため）
            if 'BLOCKSIZE' in streaming_options:
                del streaming_options['BLOCKSIZE']
            
            # タイル関連のオプションを追加
            streaming_options.update({
                'TILED': 'YES',
                'BLOCKXSIZE': '1024',
                'BLOCKYSIZE': '1024'
            })
            
            with rasterio.Env(GDAL_CACHEMAX=4096):
                data.rio.to_raster(
                    dst,
                    driver="COG" if use_cog_driver else "GTiff",
                    windowed=True,  # ウィンドウ処理を有効化
                    **streaming_options
                )
            return
        except Exception as e:
            logger.warning(f"Direct streaming failed: {e}, falling back to chunked write")
        
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
            except:
                pass

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
                                
                                # 座標のスライスを作成
                                y_slice = slice(y_start, y_end)
                                x_slice = slice(x_start, x_end)
                                
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
            def gdal_progress_callback(complete, message, cb_data):
                if not hasattr(gdal_progress_callback, 'pbar'):
                    gdal_progress_callback.pbar = tqdm(total=100, desc="COG conversion", unit="%")
                gdal_progress_callback.pbar.n = int(complete * 100)
                gdal_progress_callback.pbar.refresh()
                if complete >= 1.0:
                    gdal_progress_callback.pbar.close()
                return 1
            
            gdal.Translate(
                str(dst),
                str(vrt_file),
                format="COG" if use_cog_driver else "GTiff",
                creationOptions=list(f"{k}={v}" for k, v in cog_options.items()),
                callback=gdal_progress_callback
            )
            
            logger.info(f"Successfully created COG from {len(chunk_files)} chunks")
    else:
        # 既存の処理
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
# 5. gdal_translate/gdaladdo フォールバック関数（改善版）
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
# 6. メインパイプライン（改善版）
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
        # チャンクサイズの自動決定
        if chunk is None:
            with rasterio.open(src_cog) as src:
                total_pixels = src.width * src.height
                total_gb = (total_pixels * 4) / (1024**3)
                
            # より細かい段階的な調整
            if total_gb > 100:  # 100GB以上
                chunk = 1024
            elif total_gb > 50:  # 50-100GB
                chunk = 1536
            elif total_gb > 20:  # 20-50GB
                chunk = 2048
            else:
                chunk = get_optimal_chunk_size()
            
            logger.info(f"Dataset size: {total_gb:.1f} GB, using chunk size: {chunk}x{chunk}")
        
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
            try:
                x_res = abs(float(dem.rio.resolution()[0]))
                y_res = abs(float(dem.rio.resolution()[1]))
                pixel_size = (x_res + y_res) / 2
                logger.info(f"Detected pixel size: {pixel_size:.2f}")
            except:
                pixel_size = 1.0  # デフォルト
                logger.warning("Could not determine pixel size, using 1.0")
            
            # パラメータに設定
            params['pixel_size'] = pixel_size
            
            # 新しい効率的なモードをデフォルトに
            if radii is None and sigmas is None and auto_radii:
                logger.info("Analyzing terrain for automatic radii determination...")
                
                # 地形解析
                terrain_stats = analyze_terrain_characteristics(gpu_arr, pixel_size, include_fft=False)
                
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
                    
                    # 地形解析
                    terrain_stats = analyze_terrain_characteristics(gpu_arr, pixel_size, include_fft=True)
                    
                    # 最適なsigmaを決定
                    sigmas = determine_optimal_sigmas(terrain_stats, pixel_size)
                    
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
            try:
                x_res = abs(float(dem.rio.resolution()[0]))
                y_res = abs(float(dem.rio.resolution()[1]))
                params['pixel_size'] = (x_res + y_res) / 2
                logger.info(f"Detected pixel size: {params['pixel_size']:.2f}")
            except:
                params['pixel_size'] = 1.0
                logger.warning("Could not determine pixel size, using 1.0")

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
        # CuPyメモリプールの明示的なクリア（追加）
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
