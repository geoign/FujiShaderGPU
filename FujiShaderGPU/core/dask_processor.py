# -*- coding: utf-8 -*-
"""
FujiShaderGPU/core/dask_processor.py
Dask-CUDA地形解析処理のコア実装
"""

###############################################################################
# 依存ライブラリ
###############################################################################
from __future__ import annotations

import os, subprocess, warnings, logging, rasterio, psutil, GPUtil
from pathlib import Path
from typing import List, Tuple, Optional

import cupy as cp
import numpy as np
import dask.array as da
from dask_cuda import LocalCUDACluster
from distributed import Client
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
    base_chunk = int((gpu_memory_gb * 1024) ** 0.5 * 10)
    # 512の倍数に丸める（COGブロックサイズとの整合性）
    # return max(2048, min(8192, (base_chunk // 512) * 512))  # この行を削除
    return max(4096, min(8192, (base_chunk // 512) * 512))  # 最小値を4096に増加

def make_cluster(memory_fraction: float = 0.6) -> Tuple[LocalCUDACluster, Client]:  # 0.8 → 0.6に削減
    """Colab A100 (40 GB VRAM) 用の最適化されたクラスタを構築"""
    try:
        # GPU情報を取得
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory_gb = gpus[0].memoryTotal / 1024
            logger.info(f"GPU detected: {gpus[0].name}, Memory: {gpu_memory_gb:.1f} GB")
        else:
            gpu_memory_gb = 40  # デフォルトA100想定
            
        # rmm_size = int(gpu_memory_gb * memory_fraction * 0.7)  # さらに保守的に  # この行を削除
        rmm_size = int(gpu_memory_gb * memory_fraction * 0.5)  # さらに保守的に（0.7→0.5）
        
        cluster = LocalCUDACluster(
            device_memory_limit=str(memory_fraction),
            jit_unspill=True,
            rmm_pool_size=f"{rmm_size}GB",
            threads_per_worker=1,  # メモリ競合を避ける
            silence_logs=logging.WARNING,
        )
        client = Client(cluster)
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        return cluster, client
    except Exception as e:
        logger.error(f"Failed to create cluster: {e}")
        raise

###############################################################################
# 3. 地形解析によるsigma自動決定
###############################################################################

def analyze_terrain_scales(dem_arr: da.Array, sample_ratio: float = 0.01) -> dict:
    """地形の特性を解析してスケールパラメータを推定"""
    # サンプリングして計算を高速化
    h, w = dem_arr.shape
    sample_size = int(min(h, w) * sample_ratio)
    sample_size = max(512, min(4096, sample_size))  # 512-4096の範囲に制限
    
    # 中央部分をサンプリング
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - sample_size // 2)
    y2 = min(h, cy + sample_size // 2)
    x1 = max(0, cx - sample_size // 2)
    x2 = min(w, cx + sample_size // 2)
    
    sample = dem_arr[y1:y2, x1:x2].compute()
    
    # NaN除去
    valid_mask = ~cp.isnan(sample)
    if not valid_mask.any():
        raise ValueError("No valid elevation data found")
    
    # 基本統計量
    elevations = sample[valid_mask]
    elev_range = float(cp.ptp(elevations))
    std_dev = float(cp.std(elevations))
    
    # 勾配計算
    dy, dx = cp.gradient(sample)
    slope = cp.sqrt(dy**2 + dx**2)
    valid_slope = slope[valid_mask]
    mean_slope = float(cp.mean(valid_slope))
    max_slope = float(cp.percentile(valid_slope, 95))  # 95パーセンタイル
    
    # 2次微分（曲率の代理）
    dyy, dyx = cp.gradient(dy)
    dxy, dxx = cp.gradient(dx)
    curvature = cp.abs(dxx + dyy)
    valid_curv = curvature[valid_mask]
    mean_curv = float(cp.mean(valid_curv))
    
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
    
    return {
        'elevation_range': elev_range,
        'std_dev': std_dev,
        'mean_slope': mean_slope,
        'max_slope': max_slope,
        'mean_curvature': mean_curv,
        'dominant_scales': dominant_scales,
        'sample_size': sample.shape
    }

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
        import numpy as np
        indices = np.linspace(0, len(sigmas)-1, 3).astype(int)
        sigmas = [sigmas[i] for i in indices]
    
    return [float(s) for s in sigmas]


###############################################################################
# 4. 直接 COG 出力 (GDAL ≥ 3.8) - 改善版
###############################################################################

def get_cog_options(dtype: str) -> dict:
    """データ型に応じた最適なCOGオプションを返す"""
    base_options = {
        "COMPRESS": "ZSTD",
        "LEVEL": "1",
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
    import osgeo.gdal as gdal
    version = gdal.VersionInfo("VERSION_NUM")
    major = int(version[0])
    minor = int(version[1:3])
    return major, minor

def write_cog_da(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """DataArray を直接 COG として保存（進捗表示付き）"""
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    
    # データ型を取得
    dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    if use_cog_driver:
        try:
            logger.info(f"Using COG driver (GDAL {major}.{minor}) with dtype={dtype_str}")
            with rasterio.Env(GDAL_CACHEMAX=512):
                if show_progress:
                    # Daskの計算進捗を表示
                    with tqdm(total=100, desc="Writing COG") as pbar:
                        def callback(future):
                            pbar.update(10)
                        
                        future = data.rio.to_raster(
                            dst,
                            driver="COG",
                            **cog_options,
                        )
                else:
                    data.rio.to_raster(
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

def _fallback_cog_write(data: xr.DataArray, dst: Path, cog_options: dict):
    """フォールバック：一時ファイル経由でCOG作成"""
    tmp = dst.with_suffix(".tmp.tif")
    try:
        # COG固有のオプションを除外
        tiff_options = {k: v for k, v in cog_options.items() 
                       if k not in ['OVERVIEWS', 'OVERVIEW_RESAMPLING']}
        
        data.rio.to_raster(
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
    agg: str = "mean",
    chunk: Optional[int] = None,
    show_progress: bool = True,
    auto_sigma: bool = True,
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
            chunk = get_optimal_chunk_size()
            logger.info(f"Auto-selected chunk size: {chunk}x{chunk}")
        
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
        
        # 地理座標系の大きなデータの場合はチャンクサイズを大幅に調整
        if dem.shape[0] > 30000 or dem.shape[1] > 30000:
            logger.info("Very large geographic dataset detected, using minimal chunk size")
            # chunk = 512  # より小さなチャンクサイズ  # この行を削除
            chunk = 2048  # より大きなチャンクサイズに変更
            # データを再チャンク
            dem = dem.chunk({"y": chunk, "x": chunk})
            logger.info(f"Rechunked to {chunk}x{chunk}")
        
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
        
        # 6‑2.5) 自動sigma決定（必要な場合、RVIアルゴリズムのみ）
        if algorithm == "rvi" and sigmas is None and auto_sigma:
            logger.info("Analyzing terrain for automatic sigma determination...")
            
            # 地形解析
            terrain_stats = analyze_terrain_scales(gpu_arr)
            
            # ピクセルサイズを取得（座標系から）
            try:
                x_res = abs(float(dem.rio.resolution()[0]))
                y_res = abs(float(dem.rio.resolution()[1]))
                pixel_size = (x_res + y_res) / 2
            except:
                pixel_size = 1.0  # デフォルト
                logger.warning("Could not determine pixel size, using 1.0")
            
            # 最適なsigmaを決定
            sigmas = determine_optimal_sigmas(terrain_stats, pixel_size)
            
            logger.info(f"Terrain analysis results:")
            logger.info(f"  - Elevation range: {terrain_stats['elevation_range']:.1f} m")
            logger.info(f"  - Mean slope: {terrain_stats['mean_slope']:.3f}")
            logger.info(f"  - Detected scales: {terrain_stats.get('dominant_scales', [])}")
            logger.info(f"  - Auto-determined sigmas: {sigmas}")
            
            # パラメータに設定
            params['sigmas'] = sigmas
        
        elif algorithm == "rvi" and sigmas is None:
            # auto_sigmaがFalseでsigmasも指定されていない場合
            raise ValueError("Either provide sigmas or enable auto_sigma")
        
        # RVIの場合は手動指定のsigmasを設定
        if algorithm == "rvi" and sigmas is not None:
            params['sigmas'] = sigmas
        
        # 多くのアルゴリズムでピクセルサイズが必要
        if 'pixel_size' not in params or params['pixel_size'] == 1.0:
            try:
                x_res = abs(float(dem.rio.resolution()[0]))
                y_res = abs(float(dem.rio.resolution()[1]))
                params['pixel_size'] = (x_res + y_res) / 2
                logger.info(f"Detected pixel size: {params['pixel_size']:.2f}")
            except:
                params['pixel_size'] = 1.0
                logger.warning("Could not determine pixel size, using 1.0")
        
        # 6‑3) アルゴリズム実行
        logger.info(f"Computing {algorithm} with parameters: {params}")
        result_gpu: da.Array = algo.process(gpu_arr, **params)
        
        # 6‑4) GPU→CPU 戻し（改善：明示的なdtype）
        result_cpu = result_gpu.map_blocks(
            cp.asnumpy, 
            dtype="float32",
            meta=cp.empty((0, 0), dtype=cp.float32).get()
        )
        
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
        
        # 6‑6) 出力 (直接 COG)
        write_cog_da(result_da, Path(dst_cog), show_progress=show_progress)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        client.close()
        cluster.close()