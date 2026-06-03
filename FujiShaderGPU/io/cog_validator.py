"""
FujiShaderGPU/io/cog_validator.py
COG quality validation utilities.
"""
import logging
import os

from osgeo import gdal

# Suppress GDAL 4.0 future warnings while keeping current non-exception behavior
gdal.DontUseExceptions()

logger = logging.getLogger(__name__)


def _validate_cog_for_qgis(cog_path: str):
    """
    QGIS-optimization COG validation.
    """
    logger.info("=== COG quality validation start ===")

    try:
        ds = gdal.Open(cog_path, gdal.GA_ReadOnly)
        if ds is None:
            logger.error("Cannot open the COG file")
            return False

        # Basic info
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount

        logger.info("COG basic info:")
        logger.info("   Size: %d x %d", width, height)
        logger.info("   Bands: %d", bands)

        # Check tile structure
        band = ds.GetRasterBand(1)
        block_x, block_y = band.GetBlockSize()

        if block_x == width and block_y == 1:
            logger.error("Strip layout (not tiled)")
            tiled = False
        else:
            logger.info("Tiled layout: %d x %d", block_x, block_y)
            tiled = True

        # Check overviews
        overview_count = band.GetOverviewCount()
        logger.info("Overview count: %d", overview_count)

        if overview_count == 0:
            logger.error("No overviews - display in QGIS will be slow")
        elif overview_count < 4:
            logger.warning("Few overviews - more levels are recommended")
        else:
            logger.info("Sufficient overviews are present")

        # Show overview sizes
        for i in range(overview_count):
            ovr = band.GetOverview(i)
            ovr_width = ovr.XSize
            ovr_height = ovr.YSize
            scale_factor = width // ovr_width
            logger.info(
                "   Level %d: %d x %d (1/%d)",
                i + 1, ovr_width, ovr_height, scale_factor,
            )

        # Check compression
        img_md = ds.GetMetadata('IMAGE_STRUCTURE') or {}
        compression = img_md.get('COMPRESSION', ds.GetMetadata().get('COMPRESSION', 'none'))
        logger.info("Compression: %s", compression)

        # Check COG compliance
        metadata = ds.GetMetadata()
        layout = img_md.get('LAYOUT', metadata.get('LAYOUT', 'unknown'))

        if 'COG' in layout.upper() or tiled:
            logger.info("COG-compliant")
            cog_compliant = True
        else:
            logger.error("Not COG-compliant")
            cog_compliant = False

        # File size
        file_size_mb = os.path.getsize(cog_path) / (1024 * 1024)
        logger.info("File size: %.1f MB", file_size_mb)

        # QGIS optimization score
        score = 0
        if tiled:
            score += 30
        score += min(overview_count * 10, 40)  # up to 40 points
        if cog_compliant:
            score += 20
        if 512 <= block_x <= 1024:  # optimal QGIS block size
            score += 10

        logger.info("QGIS optimization score: %d/100", score)

        if score >= 80:
            logger.info("Excellent - fast display in QGIS")
        elif score >= 60:
            logger.warning("Good - normal display speed")
        elif score >= 40:
            logger.warning("Needs improvement - display may be slow")
        else:
            logger.error("Poor - display in QGIS will be very slow")

        # Improvement suggestions
        if score < 80:
            logger.info("Improvement suggestions:")
            if not tiled:
                logger.info("   - Tile the file")
            if overview_count < 6:
                logger.info("   - Add more overview levels")
            if not cog_compliant:
                logger.info("   - Regenerate in COG format")

        ds = None
        logger.info("=== COG quality validation complete ===")
        return score >= 60

    except Exception as e:
        logger.error("COG validation error: %s", e)
        return False
