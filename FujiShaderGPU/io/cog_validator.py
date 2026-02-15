"""
FujiShaderGPU/io/cog_validator.py
COG品質検証ユーティリティ
"""
import logging
import os

from osgeo import gdal

# GDAL 4.0 将来警告を抑制しつつ現行の非例外動作を維持
gdal.DontUseExceptions()

logger = logging.getLogger(__name__)


def _validate_cog_for_qgis(cog_path: str):
    """
    QGIS最適化COG検証
    """
    logger.info("=== COG品質検証開始 ===")

    try:
        ds = gdal.Open(cog_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error("COGファイルを開けません")
            return False

        # 基本情報
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount

        logger.info("COG基本情報:")
        logger.info("   サイズ: %d x %d", width, height)
        logger.info("   バンド数: %d", bands)

        # タイル構造確認
        band = ds.GetRasterBand(1)
        block_x, block_y = band.GetBlockSize()

        if block_x == width and block_y == 1:
            logger.error("ストライプ形式（タイル化されていません）")
            tiled = False
        else:
            logger.info("タイル形式: %d x %d", block_x, block_y)
            tiled = True

        # オーバービュー確認
        overview_count = band.GetOverviewCount()
        logger.info("オーバービュー数: %d", overview_count)

        if overview_count == 0:
            logger.error("オーバービューがありません - QGISで表示が遅くなります")
        elif overview_count < 4:
            logger.warning("オーバービューが少ないです - より多くのレベルが推奨")
        else:
            logger.info("十分なオーバービューがあります")

        # オーバービューサイズ表示
        for i in range(overview_count):
            ovr = band.GetOverview(i)
            ovr_width = ovr.XSize
            ovr_height = ovr.YSize
            scale_factor = width // ovr_width
            logger.info(
                "   レベル%d: %d x %d (1/%d)",
                i + 1, ovr_width, ovr_height, scale_factor,
            )

        # 圧縮確認
        img_md = ds.GetMetadata('IMAGE_STRUCTURE') or {}
        compression = img_md.get('COMPRESSION', ds.GetMetadata().get('COMPRESSION', 'なし'))
        logger.info("圧縮: %s", compression)

        # COG準拠確認
        metadata = ds.GetMetadata()
        layout = img_md.get('LAYOUT', metadata.get('LAYOUT', '不明'))

        if 'COG' in layout.upper() or tiled:
            logger.info("COG形式準拠")
            cog_compliant = True
        else:
            logger.error("COG形式非準拠")
            cog_compliant = False

        # ファイルサイズ
        file_size_mb = os.path.getsize(cog_path) / (1024 * 1024)
        logger.info("ファイルサイズ: %.1f MB", file_size_mb)

        # QGIS最適化スコア
        score = 0
        if tiled:
            score += 30
        score += min(overview_count * 10, 40)  # 最大40点
        if cog_compliant:
            score += 20
        if 512 <= block_x <= 1024:  # QGIS最適ブロックサイズ
            score += 10

        logger.info("QGIS最適化スコア: %d/100", score)

        if score >= 80:
            logger.info("優秀 - QGISで高速表示されます")
        elif score >= 60:
            logger.warning("良好 - 通常速度で表示されます")
        elif score >= 40:
            logger.warning("要改善 - 表示が遅い可能性があります")
        else:
            logger.error("不良 - QGISで表示が非常に遅くなります")

        # 改善提案
        if score < 80:
            logger.info("改善提案:")
            if not tiled:
                logger.info("   - ファイルをタイル化してください")
            if overview_count < 6:
                logger.info("   - より多くのオーバービューレベルを追加してください")
            if not cog_compliant:
                logger.info("   - COG形式で再生成してください")

        ds = None
        logger.info("=== COG品質検証完了 ===")
        return score >= 60

    except Exception as e:
        logger.error("COG検証エラー: %s", e)
        return False
