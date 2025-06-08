"""
FujiShaderGPU/io/cog_builder.py
"""
from ..config.gdal_config import _configure_gdal_ultra_performance
import os, glob, shutil
from typing import List
from osgeo import gdal

def _create_vrt_command_line_ultra(tile_files: List[str], vrt_path: str):
    """
    コマンドライン版超高速VRT
    """
    file_list_path = vrt_path.replace('.vrt', '_files.txt')
    
    try:
        with open(file_list_path, 'w') as f:
            for tile_file in tile_files:
                f.write(tile_file + '\n')
        
        import subprocess
        cmd = [
            'gdalbuildvrt',
            '-resolution', 'highest',
            '-r', 'nearest',
            '-hidenodata',
            '-input_file_list', file_list_path,
            vrt_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)



def _create_qgis_optimized_overviews(tiff_path: str):
    """
    QGIS表示最適化オーバービュー生成
    """
    try:
        import subprocess
        
        # QGISに最適な多段階オーバービュー
        overview_levels = ["2", "4", "8", "16", "32", "64", "128", "256", "512"]
        
        overview_cmd = [
            "gdaladdo", 
            "-r", "average",                    # 高品質リサンプリング
            "--config", "COMPRESS_OVERVIEW", "ZSTD",
            "--config", "ZLEVEL_OVERVIEW", "1",
            "--config", "BIGTIFF_OVERVIEW", "YES",
            "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
            "--config", "GDAL_CACHEMAX", "4096",
            "--config", "GDAL_TIFF_OVR_BLOCKSIZE", "512",  # オーバービューブロックサイズ
            tiff_path
        ] + overview_levels
        
        result = subprocess.run(overview_cmd, check=True, capture_output=True, text=True)
        print(f"オーバービュー生成完了: {len(overview_levels)}レベル")
        
    except subprocess.CalledProcessError as e:
        print(f"警告: gdaladdoでのオーバービュー生成失敗: {e}")
        # GDAL APIでのフォールバック
        _create_overviews_gdal_api(tiff_path)
    except FileNotFoundError:
        print("gdaladdoコマンド未発見、GDAL APIにフォールバック")
        _create_overviews_gdal_api(tiff_path)

def _create_overviews_gdal_api(tiff_path: str):
    """
    GDAL APIによるオーバービュー生成（フォールバック）
    """
    try:
        # GDAL設定
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'ZSTD')
        gdal.SetConfigOption('GDAL_TIFF_OVR_BLOCKSIZE', '512')
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        
        ds = gdal.Open(tiff_path, gdal.GA_Update)
        if ds is None:
            raise ValueError("ファイルを開けません")
        
        # 多段階オーバービュー生成
        overview_levels = [2, 4, 8, 16, 32, 64, 128, 256]
        result = ds.BuildOverviews("AVERAGE", overview_levels)
        
        if result != 0:
            raise ValueError("オーバービュー生成失敗")
        
        ds = None  # ファイルを閉じる
        print(f"GDAL APIオーバービュー生成完了: {len(overview_levels)}レベル")
        
    except Exception as e:
        print(f"GDAL APIオーバービュー生成エラー: {e}")
        print("オーバービューなしで継続します")

def _build_vrt_and_cog_ultra_fast(tmp_tile_dir: str, output_cog_path: str, gpu_config: dict):
    """
    超高速VRT構築とCOG生成
    """
    print("=== 超高速COG生成開始 ===")
    
    # GDAL超高速設定
    _configure_gdal_ultra_performance(gpu_config)
    
    vrt_path = os.path.join(tmp_tile_dir, "rvi_tiles.vrt")
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))

    if not tile_files:
        raise ValueError(f"タイルファイルが見つかりません: {tmp_tile_dir}")

    print(f"VRT作成: {len(tile_files)}個のタイルを統合")
    
    # 超高速VRT作成
    _create_vrt_ultra_fast(tile_files, vrt_path)
    
    # COGドライバー確認と最適生成
    cog_driver = gdal.GetDriverByName("COG")
    if cog_driver:
        _create_cog_ultra_fast(vrt_path, output_cog_path, gpu_config)
    else:
        _create_cog_gtiff_ultra_fast(vrt_path, output_cog_path, gpu_config)
    
    print("=== 超高速COG生成完了 ===")

def _create_vrt_ultra_fast(tile_files: List[str], vrt_path: str):
    """
    超高速VRT作成
    """
    import time
    start_time = time.time()
    
    # コマンドライン版を優先使用
    if len(tile_files) > 20:
        try:
            _create_vrt_command_line_ultra(tile_files, vrt_path)
            elapsed = time.time() - start_time
            print(f"コマンドライン超高速VRT: {elapsed:.1f}秒")
            return
        except Exception as e:
            print(f"コマンドライン失敗、Python版にフォールバック: {e}")
    
    # Python版超高速VRT
    vrt_options = gdal.BuildVRTOptions(
        resolution="highest",
        resampleAlg="nearest",
        allowProjectionDifference=True,
        addAlpha=False,
        hideNodata=True,
        srcNodata=None,
    )
    
    gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)
    elapsed = time.time() - start_time
    print(f"Python超高速VRT: {elapsed:.1f}秒")

def _create_cog_ultra_fast(vrt_path: str, output_cog_path: str, gpu_config: dict):
    """
    COGドライバー版超高速生成（QGIS最適化）
    """
    import time
    start_time = time.time()
    
    # QGIS表示最適化COGオプション
    cog_options = [
        "COMPRESS=ZSTD",
        "LEVEL=1",                      # 高速圧縮レベル
        "BIGTIFF=YES",
        "BLOCKSIZE=512",                # QGIS最適ブロックサイズ（4096→512）
        "NUM_THREADS=ALL_CPUS",
        "OVERVIEW_RESAMPLING=AVERAGE",  # 高品質リサンプリング（NEAREST→AVERAGE）
        "OVERVIEW_COUNT=8",             # 十分なオーバービュー数（3→8）
        "OVERVIEW_LEVELS=2,4,8,16,32,64,128,256",  # 多段階ピラミッド
        "ALIGNED_LEVELS=4",             # アライメント最適化
        "TILING_SCHEME=GoogleMapsCompatible",  # 標準タイリング
    ]
    
    def progress_callback(complete, message, cb_data):
        if int(complete * 100) % 10 == 0:
            print(f"COG変換: {complete*100:.0f}%完了")
        return 1
    
    try:
        vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
        if vrt_ds is None:
            raise ValueError(f"VRT読み込み失敗: {vrt_path}")
        
        result = gdal.Translate(
            output_cog_path,
            vrt_ds,
            format="COG",
            creationOptions=cog_options,
            callback=progress_callback
        )
        
        if result is None:
            raise ValueError("COG変換失敗")
        
        result = None
        vrt_ds = None
        
        elapsed = time.time() - start_time
        size_mb = os.path.getsize(output_cog_path) / (1024 * 1024)
        throughput = size_mb / elapsed if elapsed > 0 else 0
        
        print(f"超高速COG完了: {elapsed:.1f}秒, {size_mb:.1f}MB, {throughput:.1f}MB/s")
        
    except Exception as e:
        raise ValueError(f"COG生成エラー: {e}")

def _create_cog_gtiff_ultra_fast(vrt_path: str, output_cog_path: str, gpu_config: dict):
    """
    GTiff版超高速COG生成（QGIS最適化）
    """
    import time
    start_time = time.time()
    
    temp_tiff_path = output_cog_path.replace('.tif', '_temp.tif')
    
    try:
        # QGIS最適化GTiffオプション
        gtiff_options = [
            "TILED=YES",
            "BLOCKXSIZE=512",       # QGIS最適サイズ（2048→512）
            "BLOCKYSIZE=512",
            "COMPRESS=ZSTD",
            "ZLEVEL=1",
            "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS"
        ]

        vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
        if vrt_ds is None:
            raise ValueError(f"VRT読み込み失敗")
        
        temp_result = gdal.Translate(
            temp_tiff_path,
            vrt_ds,
            format="GTiff",
            creationOptions=gtiff_options
        )
        
        if temp_result is None:
            raise ValueError("GeoTIFF作成失敗")
        
        temp_result = None
        vrt_ds = None
        
        # QGIS最適化オーバービュー生成
        print("QGIS最適化オーバービュー生成中...")
        _create_qgis_optimized_overviews(temp_tiff_path)
        
        shutil.move(temp_tiff_path, output_cog_path)
        
        elapsed = time.time() - start_time
        print(f"QGIS最適化COG完了: {elapsed:.1f}秒")
        
    except Exception as e:
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)
        raise ValueError(f"GTiff COG生成エラー: {e}")
