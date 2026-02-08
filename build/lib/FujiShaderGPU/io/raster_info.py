"""
FujiShaderGPU/io/raster_info.py
"""
import math

import rasterio

def detect_pixel_size_from_cog(input_cog_path: str) -> float:
    """
    COGファイルからピクセルサイズを自動検出（高速化）
    """
    try:
        with rasterio.open(input_cog_path) as src:
            transform = src.transform
            crs = src.crs
            
            pixel_size_x = abs(transform.a)
            pixel_size_y = abs(transform.e)
            
            if crs and crs.is_geographic:
                lat_center = (src.bounds.bottom + src.bounds.top) / 2
                meter_per_degree_lat = 111320.0
                meter_per_degree_lon = 111320.0 * math.cos(math.radians(lat_center))
                
                pixel_size_x_m = pixel_size_x * meter_per_degree_lon
                pixel_size_y_m = pixel_size_y * meter_per_degree_lat
                
                print(f"地理座標系検出: 緯度{lat_center:.3f}度")
                print(f"メートル換算: {pixel_size_x_m:.3f}m x {pixel_size_y_m:.3f}m")
            else:
                pixel_size_x_m = pixel_size_x
                pixel_size_y_m = pixel_size_y
                print(f"投影座標系: {pixel_size_x_m:.3f}m x {pixel_size_y_m:.3f}m")
            
            pixel_size = (pixel_size_x_m + pixel_size_y_m) / 2
            print(f"自動検出ピクセルサイズ: {pixel_size:.3f}m")
            return pixel_size
            
    except Exception as e:
        print(f"ピクセルサイズ自動検出エラー: {e}")
        return 0.5
