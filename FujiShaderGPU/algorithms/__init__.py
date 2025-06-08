"""
FujiShaderGPU/algorithms/__init__.py
"""
from .base import BaseAlgorithm
from .rvi_gaussian import _compute_multiscale_rvi_ultra_fast
from .hillshade import HillshadeAlgorithm
from .atmospheric_scattering import AtmosphericScatteringAlgorithm
from .composite_terrain import CompositeTerrainAlgorithm
from .curvature import CurvatureAlgorithm
from .frequency_enhancement import FrequencyEnhancementAlgorithm, DirectionalFrequencyFilter
from .visual_saliency import VisualSaliencyAlgorithm, MultiResolutionSaliency
from .utils import calculate_padding

__all__ = [
    'BaseAlgorithm',
    'HillshadeAlgorithm',
    'AtmosphericScatteringAlgorithm',
    'CompositeTerrainAlgorithm',
    'CurvatureAlgorithm',
    'FrequencyEnhancementAlgorithm',
    'DirectionalFrequencyFilter',
    'VisualSaliencyAlgorithm',
    'MultiResolutionSaliency',
    'calculate_padding',
    '_compute_multiscale_rvi_ultra_fast',
]

_REGISTRY = {
    "rvi_gaussian": _compute_multiscale_rvi_ultra_fast,
    "hillshade": HillshadeAlgorithm,
    "atmospheric_scattering": AtmosphericScatteringAlgorithm,
    "composite_terrain": CompositeTerrainAlgorithm,
    "curvature": CurvatureAlgorithm,
    "frequency_enhancement": FrequencyEnhancementAlgorithm,
    "visual_saliency": VisualSaliencyAlgorithm,
    "multi_resolution_saliency": MultiResolutionSaliency,
}
def get(name): return _REGISTRY[name]

# ===== 使用例 =====
"""
使用例: FujiShaderGPUで地形可視化を実行

import cupy as cp
from osgeo import gdal
from FujiShaderGPU.algorithms import (
    HillshadeAlgorithm,
    AtmosphericScatteringAlgorithm,
    CompositeTerrainAlgorithm,
    CurvatureAlgorithm,
    FrequencyEnhancementAlgorithm
)

# 1. DEMデータの読み込み
def load_dem_to_gpu(dem_path):
    dataset = gdal.Open(dem_path, gdal.GA_ReadOnly)
    dem_array = dataset.GetRasterBand(1).ReadAsArray()
    
    # GPUに転送
    dem_gpu = cp.asarray(dem_array, dtype=cp.float32)
    
    # ピクセルサイズの取得
    geotransform = dataset.GetGeoTransform()
    pixel_size = abs(geotransform[1])
    
    return dem_gpu, pixel_size, dataset

# 2. 単一アルゴリズムの使用例
def example_single_algorithm():
    # DEM読み込み
    dem_gpu, pixel_size, dataset = load_dem_to_gpu("input_dem.tif")
    
    # Hillshadeの計算
    hillshade_algo = HillshadeAlgorithm()
    hillshade_result = hillshade_algo.process(
        dem_gpu,
        azimuth=315,
        altitude=45,
        pixel_size=pixel_size,
        color_mode="warm"
    )
    
    # 結果をCPUに転送
    result_cpu = cp.asnumpy(hillshade_result)
    
    # GeoTIFFとして保存
    save_as_geotiff(result_cpu, dataset, "hillshade_output.tif")

# 3. 複合アルゴリズムの使用例
def example_composite_algorithm():
    # DEM読み込み
    dem_gpu, pixel_size, dataset = load_dem_to_gpu("input_dem.tif")
    
    # 複合地形可視化
    composite_algo = CompositeTerrainAlgorithm()
    
    # カスタムパラメータで実行
    result = composite_algo.process(
        dem_gpu,
        pixel_size=pixel_size,
        layers={
            "hillshade": {"enabled": True, "weight": 1.2},
            "atmospheric": {"enabled": True, "weight": 0.8},
            "rvi": {"enabled": True, "weight": 0.5},
        },
        hillshade_params={
            "azimuth": 315,
            "altitude": 45,
            "color_mode": "warm",
        },
        atmospheric_params={
            "tpi_radii": [4, 16, 64],
            "base_ambient": 0.35,
        },
        blend_mode="weighted",
        tone_mapping=True,
        gamma=2.0
    )
    
    # 結果をCPUに転送して保存
    result_cpu = cp.asnumpy(result)
    save_as_geotiff(result_cpu, dataset, "composite_output.tif")

# 4. バッチ処理の例（大規模データ用）
def example_batch_processing():
    # 大規模DEMをタイル単位で処理
    dem_gpu, pixel_size, dataset = load_dem_to_gpu("large_dem.tif")
    
    tile_size = 4096
    overlap = 256  # オーバーラップ（境界アーチファクト防止）
    
    rows, cols = dem_gpu.shape
    result_gpu = cp.zeros((rows, cols, 3), dtype=cp.float32)
    
    algo = CompositeTerrainAlgorithm()
    
    for y in range(0, rows, tile_size - overlap):
        for x in range(0, cols, tile_size - overlap):
            # タイル境界の計算
            y_start = max(0, y - overlap//2)
            y_end = min(rows, y + tile_size + overlap//2)
            x_start = max(0, x - overlap//2)
            x_end = min(cols, x + tile_size + overlap//2)
            
            # タイル抽出
            tile = dem_gpu[y_start:y_end, x_start:x_end]
            
            # 処理
            tile_result = algo.process(tile, pixel_size=pixel_size)
            
            # 結果をメイン配列に配置（オーバーラップ部分を除く）
            y_out_start = y
            y_out_end = min(rows, y + tile_size)
            x_out_start = x
            x_out_end = min(cols, x + tile_size)
            
            y_in_start = overlap//2 if y > 0 else 0
            y_in_end = y_in_start + (y_out_end - y_out_start)
            x_in_start = overlap//2 if x > 0 else 0
            x_in_end = x_in_start + (x_out_end - x_out_start)
            
            result_gpu[y_out_start:y_out_end, x_out_start:x_out_end] = \
                tile_result[y_in_start:y_in_end, x_in_start:x_in_end]
            
            # メモリクリーンアップ
            del tile, tile_result
            cp.cuda.Stream.null.synchronize()
    
    # 結果保存
    result_cpu = cp.asnumpy(result_gpu)
    save_as_geotiff(result_cpu, dataset, "batch_output.tif")

# 5. GeoTIFF保存用のヘルパー関数
def save_as_geotiff(rgb_array, source_dataset, output_path):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, bands = rgb_array.shape
    
    # データ型変換（float32 -> uint8）
    rgb_uint8 = (np.clip(rgb_array, 0, 1) * 255).astype(np.uint8)
    
    # 出力データセット作成
    out_dataset = driver.Create(
        output_path, 
        cols, 
        rows, 
        bands, 
        gdal.GDT_Byte,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    # 地理参照情報のコピー
    out_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
    out_dataset.SetProjection(source_dataset.GetProjection())
    
    # 各バンドの書き込み
    for i in range(bands):
        out_band = out_dataset.GetRasterBand(i + 1)
        out_band.WriteArray(rgb_uint8[:, :, i])
        
        # NoData値の設定（必要に応じて）
        source_nodata = source_dataset.GetRasterBand(1).GetNoDataValue()
        if source_nodata is not None:
            out_band.SetNoDataValue(0)
    
    # クリーンアップ
    out_dataset.FlushCache()
    out_dataset = None

# 6. パフォーマンス測定の例
def benchmark_algorithms():
    import time
    
    dem_gpu, pixel_size, _ = load_dem_to_gpu("test_dem.tif")
    print(f"DEM shape: {dem_gpu.shape}")
    
    algorithms = [
        ("Hillshade", HillshadeAlgorithm()),
        ("Atmospheric", AtmosphericScatteringAlgorithm()),
        ("Curvature", CurvatureAlgorithm()),
        ("Frequency", FrequencyEnhancementAlgorithm()),
    ]
    
    for name, algo in algorithms:
        # ウォームアップ
        _ = algo.process(dem_gpu, pixel_size=pixel_size)
        cp.cuda.Stream.null.synchronize()
        
        # 計測
        start = time.time()
        for _ in range(10):
            result = algo.process(dem_gpu, pixel_size=pixel_size)
            cp.cuda.Stream.null.synchronize()
        
        elapsed = (time.time() - start) / 10
        print(f"{name}: {elapsed*1000:.1f} ms")
        
        del result
"""