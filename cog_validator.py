"""
FujiShaderGPU/io/cog_validator.py
"""
import os
from osgeo import gdal

def _validate_cog_for_qgis(cog_path: str):
    """
    QGIS最適化COG検証
    """
    print("=== COG品質検証開始 ===")
    
    try:
        ds = gdal.Open(cog_path, gdal.GA_ReadOnly)
        if ds is None:
            print("❌ COGファイルを開けません")
            return False
        
        # 基本情報
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        print(f"📊 COG基本情報:")
        print(f"   サイズ: {width} x {height}")
        print(f"   バンド数: {bands}")
        
        # タイル構造確認
        band = ds.GetRasterBand(1)
        block_x, block_y = band.GetBlockSize()
        
        if block_x == width and block_y == 1:
            print("❌ ストライプ形式（タイル化されていません）")
            tiled = False
        else:
            print(f"✅ タイル形式: {block_x} x {block_y}")
            tiled = True
        
        # オーバービュー確認
        overview_count = band.GetOverviewCount()
        print(f"📈 オーバービュー数: {overview_count}")
        
        if overview_count == 0:
            print("❌ オーバービューがありません - QGISで表示が遅くなります")
        elif overview_count < 4:
            print("⚠️  オーバービューが少ないです - より多くのレベルが推奨")
        else:
            print("✅ 十分なオーバービューがあります")
            
        # オーバービューサイズ表示
        for i in range(overview_count):
            ovr = band.GetOverview(i)
            ovr_width = ovr.XSize
            ovr_height = ovr.YSize
            scale_factor = width // ovr_width
            print(f"   レベル{i+1}: {ovr_width} x {ovr_height} (1/{scale_factor})")
        
        # 圧縮確認
        compression = ds.GetMetadata().get('COMPRESSION', 'なし')
        print(f"🗜️  圧縮: {compression}")
        
        # COG準拠確認
        metadata = ds.GetMetadata()
        layout = metadata.get('LAYOUT', '不明')
        
        if 'COG' in layout.upper() or tiled:
            print("✅ COG形式準拠")
            cog_compliant = True
        else:
            print("❌ COG形式非準拠")
            cog_compliant = False
        
        # ファイルサイズ
        file_size_mb = os.path.getsize(cog_path) / (1024 * 1024)
        print(f"💾 ファイルサイズ: {file_size_mb:.1f} MB")
        
        # QGIS最適化スコア
        score = 0
        if tiled:
            score += 30
        score += min(overview_count * 10, 40)  # 最大40点
        if cog_compliant:
            score += 20
        if 512 <= block_x <= 1024:  # QGIS最適ブロックサイズ
            score += 10
        
        print(f"\n🎯 QGIS最適化スコア: {score}/100")
        
        if score >= 80:
            print("✅ 優秀 - QGISで高速表示されます")
        elif score >= 60:
            print("⚠️  良好 - 通常速度で表示されます")
        elif score >= 40:
            print("⚠️  要改善 - 表示が遅い可能性があります")
        else:
            print("❌ 不良 - QGISで表示が非常に遅くなります")
        
        # 改善提案
        if score < 80:
            print("\n💡 改善提案:")
            if not tiled:
                print("   - ファイルをタイル化してください")
            if overview_count < 6:
                print("   - より多くのオーバービューレベルを追加してください")
            if not cog_compliant:
                print("   - COG形式で再生成してください")
        
        ds = None
        print("=== COG品質検証完了 ===")
        return score >= 60
        
    except Exception as e:
        print(f"❌ COG検証エラー: {e}")
        return False
