from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal


def _write_source_tiff(path: Path, width: int = 1024, height: int = 1024) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"],
    )
    arr = np.arange(width * height, dtype=np.float32).reshape(height, width)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    ds = None


@pytest.mark.skipif(gdal.GetDriverByName("COG") is None, reason="GDAL COG driver unavailable")
def test_cog_builder_writes_zstd_overviews(tmp_path):
    from FujiShaderGPU.io.cog_builder import _create_cog_ultra_fast

    src = tmp_path / "src.tif"
    vrt = tmp_path / "src.vrt"
    dst = tmp_path / "out.tif"

    _write_source_tiff(src)
    gdal.BuildVRT(str(vrt), [str(src)])

    _create_cog_ultra_fast(str(vrt), str(dst), gpu_config={})

    ds = gdal.Open(str(dst), gdal.GA_ReadOnly)
    assert ds is not None
    band = ds.GetRasterBand(1)
    assert band.GetOverviewCount() > 0
    assert (ds.GetMetadata("IMAGE_STRUCTURE") or {}).get("COMPRESSION") == "ZSTD"
    assert (ds.GetMetadata("IMAGE_STRUCTURE") or {}).get("LAYOUT") == "COG"
