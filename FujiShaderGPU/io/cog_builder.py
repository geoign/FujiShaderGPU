"""COG/VRT builder utilities."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
import time
from typing import List, Optional

from osgeo import gdal

from ..config.gdal_config import _configure_gdal_ultra_performance

# Keep current non-exception behavior and silence GDAL 4.0 future warning.
gdal.DontUseExceptions()


def _detect_nodata_from_tiles(tile_files: List[str]) -> Optional[float]:
    """Detect nodata value from first readable tile band."""
    for path in tile_files:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue() if band is not None else None
        ds = None
        if nodata is not None:
            return float(nodata)
    return None


def _create_vrt_command_line_ultra(
    tile_files: List[str],
    vrt_path: str,
    nodata: Optional[float] = None,
) -> None:
    """Create VRT by gdalbuildvrt command line (fast path for many tiles)."""
    file_list_path = vrt_path.replace(".vrt", "_files.txt")
    try:
        with open(file_list_path, "w", encoding="utf-8") as f:
            for tile_file in tile_files:
                f.write(tile_file + "\n")

        cmd = [
            "gdalbuildvrt",
            "-resolution",
            "highest",
            "-r",
            "nearest",
            "-input_file_list",
            file_list_path,
        ]
        if nodata is not None:
            nval = str(float(nodata))
            cmd.extend(["-srcnodata", nval, "-vrtnodata", nval])
        cmd.append(vrt_path)

        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)


def _create_overviews_gdal_api(tiff_path: str) -> None:
    """Fallback: build overviews via GDAL Python API."""
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "ZSTD")
    gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "512")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    ds = gdal.Open(tiff_path, gdal.GA_Update)
    if ds is None:
        raise ValueError("Failed to open TIFF for overview creation")

    levels = [2, 4, 8, 16, 32, 64, 128, 256]
    result = ds.BuildOverviews("AVERAGE", levels)
    ds = None
    if result != 0:
        raise ValueError("BuildOverviews failed")


def _create_qgis_optimized_overviews(tiff_path: str) -> None:
    """Build QGIS-friendly overview pyramid."""
    overview_levels = ["2", "4", "8", "16", "32", "64", "128", "256", "512"]
    cmd = [
        "gdaladdo",
        "-r",
        "average",
        "--config",
        "COMPRESS_OVERVIEW",
        "ZSTD",
        "--config",
        "ZLEVEL_OVERVIEW",
        "1",
        "--config",
        "BIGTIFF_OVERVIEW",
        "YES",
        "--config",
        "GDAL_NUM_THREADS",
        "ALL_CPUS",
        "--config",
        "GDAL_CACHEMAX",
        "4096",
        "--config",
        "GDAL_TIFF_OVR_BLOCKSIZE",
        "512",
        tiff_path,
    ] + overview_levels

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Overview build complete: {len(overview_levels)} levels")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"gdaladdo failed, fallback to GDAL API: {exc}")
        _create_overviews_gdal_api(tiff_path)


def _ensure_output_nodata(path: str, nodata: Optional[float]) -> None:
    """Ensure output file carries nodata metadata when nodata is known."""
    if nodata is None:
        return

    # Fast path: already set, do nothing.
    ds_ro = gdal.OpenEx(path, gdal.OF_RASTER | gdal.OF_READONLY)
    if ds_ro is None:
        return
    band_ro = ds_ro.GetRasterBand(1)
    current = band_ro.GetNoDataValue() if band_ro is not None else None
    ds_ro = None
    if current is not None:
        return

    # Only update when nodata is actually missing.
    ds = gdal.OpenEx(
        path,
        gdal.OF_RASTER | gdal.OF_UPDATE,
        open_options=["IGNORE_COG_LAYOUT_BREAK=YES"],
    )
    if ds is None:
        return
    band = ds.GetRasterBand(1)
    if band is not None:
        band.SetNoDataValue(float(nodata))
    ds = None


def _create_vrt_ultra_fast(
    tile_files: List[str],
    vrt_path: str,
    nodata: Optional[float] = None,
) -> None:
    """Create VRT from tiles."""
    start = time.time()

    if len(tile_files) > 20:
        try:
            _create_vrt_command_line_ultra(tile_files, vrt_path, nodata=nodata)
            print(f"Command line VRT: {time.time() - start:.1f}s")
            return
        except Exception as exc:
            print(f"Command line VRT failed; fallback to Python API: {exc}")

    vrt_options = gdal.BuildVRTOptions(
        resolution="highest",
        resampleAlg="nearest",
        allowProjectionDifference=True,
        addAlpha=False,
        hideNodata=False,
        srcNodata=nodata,
        VRTNodata=nodata,
    )
    gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)
    print(f"Python VRT: {time.time() - start:.1f}s")


def _create_cog_ultra_fast(
    vrt_path: str,
    output_cog_path: str,
    gpu_config: dict,
    nodata: Optional[float] = None,
) -> None:
    """Create COG directly using GDAL COG driver."""
    start = time.time()
    cog_options = [
        "COMPRESS=ZSTD",
        "LEVEL=1",
        "BIGTIFF=YES",
        "BLOCKSIZE=512",
        "NUM_THREADS=ALL_CPUS",
        "OVERVIEW_RESAMPLING=AVERAGE",
        "OVERVIEW_COUNT=8",
        "ALIGNED_LEVELS=4",
    ]

    last_progress = {"pct": -10, "ts": time.time()}

    def _estimate_written_mb(path: str) -> float:
        """Estimate bytes written during COG creation, including temporary sidecar files."""
        total = 0
        candidates = [path, f"{path}.tmp", f"{path}.ovr", f"{path}.ovr.tmp"]
        for c in candidates:
            if os.path.exists(c):
                try:
                    total += os.path.getsize(c)
                except OSError:
                    pass
        return total / (1024 * 1024)

    def progress_callback(complete, _message, _cb_data):
        pct = max(0, min(100, int(complete * 100)))
        now = time.time()
        if pct >= last_progress["pct"] + 10 or (now - last_progress["ts"] >= 30.0):
            size_mb = _estimate_written_mb(output_cog_path)
            elapsed = max(now - start, 1e-6)
            avg_mb_s = size_mb / elapsed
            print(f"COG conversion: {pct}% (written={size_mb:.1f}MB, avg={avg_mb_s:.1f}MB/s)")
            last_progress["pct"] = pct
            last_progress["ts"] = now
        return 1

    vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    if vrt_ds is None:
        raise ValueError(f"Failed to open VRT: {vrt_path}")
    print(
        "COG source size: "
        f"{vrt_ds.RasterXSize} x {vrt_ds.RasterYSize}, bands={vrt_ds.RasterCount}"
    )
    src_pixels = int(vrt_ds.RasterXSize) * int(vrt_ds.RasterYSize)

    # Python progress callbacks can dominate runtime on very large rasters because
    # GDAL calls into Python for nearly every internal chunk.
    use_python_callback = src_pixels < 1_000_000_000
    monitor_stop = threading.Event()
    monitor_thread: Optional[threading.Thread] = None

    if not use_python_callback:
        print("COG conversion: large raster detected; using low-overhead progress monitor.")

        def _monitor_progress() -> None:
            while not monitor_stop.wait(30.0):
                size_mb = _estimate_written_mb(output_cog_path)
                elapsed = max(time.time() - start, 1e-6)
                avg_mb_s = size_mb / elapsed
                print(f"COG conversion: running (written={size_mb:.1f}MB, avg={avg_mb_s:.1f}MB/s)")

        monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
        monitor_thread.start()

    try:
        translate_kwargs = dict(
            format="COG",
            creationOptions=cog_options,
            noData=nodata,
        )
        if use_python_callback:
            translate_kwargs["callback"] = progress_callback
        result = gdal.Translate(
            output_cog_path,
            vrt_ds,
            **translate_kwargs,
        )
    finally:
        if monitor_thread is not None:
            monitor_stop.set()
            monitor_thread.join(timeout=1.0)
    vrt_ds = None
    if result is None:
        raise ValueError("COG translate failed")
    result = None

    _ensure_output_nodata(output_cog_path, nodata)

    elapsed = time.time() - start
    size_mb = os.path.getsize(output_cog_path) / (1024 * 1024)
    throughput = size_mb / elapsed if elapsed > 0 else 0
    print(f"COG complete: {elapsed:.1f}s, {size_mb:.1f}MB, {throughput:.1f}MB/s")


def _create_cog_gtiff_ultra_fast(
    vrt_path: str,
    output_cog_path: str,
    gpu_config: dict,
    nodata: Optional[float] = None,
) -> None:
    """Fallback COG creation path using GTiff + overviews."""
    start = time.time()
    temp_tiff_path = output_cog_path.replace(".tif", "_temp.tif")

    gtiff_options = [
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=ZSTD",
        "ZLEVEL=1",
        "BIGTIFF=YES",
        "NUM_THREADS=ALL_CPUS",
    ]

    try:
        vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
        if vrt_ds is None:
            raise ValueError("Failed to open VRT")

        temp_result = gdal.Translate(
            temp_tiff_path,
            vrt_ds,
            format="GTiff",
            creationOptions=gtiff_options,
            noData=nodata,
        )
        vrt_ds = None
        if temp_result is None:
            raise ValueError("GTiff translate failed")
        temp_result = None

        _ensure_output_nodata(temp_tiff_path, nodata)
        print("Building overviews...")
        _create_qgis_optimized_overviews(temp_tiff_path)

        shutil.move(temp_tiff_path, output_cog_path)
        _ensure_output_nodata(output_cog_path, nodata)

        elapsed = time.time() - start
        print(f"GTiff COG complete: {elapsed:.1f}s")
    except Exception:
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)
        raise


def _prepare_external_gdal_env(gdal_bin_dir: Optional[str]) -> dict:
    """Prepare environment variables for external GDAL CLI calls."""
    env = os.environ.copy()
    if gdal_bin_dir:
        env["PATH"] = f"{gdal_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        if not env.get("GDAL_DATA"):
            for cand in (
                os.path.join(gdal_bin_dir, "gdal-data"),
                os.path.join(gdal_bin_dir, "share", "gdal"),
            ):
                if os.path.isdir(cand):
                    env["GDAL_DATA"] = cand
                    break
        if not env.get("PROJ_LIB"):
            for cand in (
                os.path.join(gdal_bin_dir, "projlib"),
                os.path.join(gdal_bin_dir, "share", "proj"),
            ):
                if os.path.isdir(cand):
                    env["PROJ_LIB"] = cand
                    break
    return env


def _resolve_external_gdal_bin_dir(gdal_bin_dir: Optional[str]) -> Optional[str]:
    """Resolve preferred external GDAL bin dir with sane Windows defaults."""
    if gdal_bin_dir and os.path.isdir(gdal_bin_dir):
        return gdal_bin_dir

    candidates = [
        os.environ.get("FUJISHADERGPU_GDAL_BIN", ""),
        r"C:\Program Files\GDAL",
        r"C:\OSGeo4W\bin",
    ]
    for cand in candidates:
        if not cand:
            continue
        if os.path.isfile(os.path.join(cand, "gdal_translate.exe")) and os.path.isfile(
            os.path.join(cand, "gdalbuildvrt.exe")
        ):
            return cand
    return None


def _find_cli_tool(tool_name: str, gdal_bin_dir: Optional[str]) -> str:
    if gdal_bin_dir:
        cand = os.path.join(gdal_bin_dir, f"{tool_name}.exe")
        if os.path.isfile(cand):
            return cand
    found = shutil.which(tool_name)
    if found:
        return found
    raise FileNotFoundError(
        f"Required GDAL CLI tool not found: {tool_name}. "
        f"Specify --gdal-bin-dir or add it to PATH."
    )


def _create_vrt_and_cog_external_cli(
    *,
    tile_files: List[str],
    vrt_path: str,
    output_cog_path: str,
    nodata: Optional[float],
    gdal_bin_dir: Optional[str],
) -> None:
    """Build VRT + COG using external GDAL executables (no Python GDAL processing path)."""
    start = time.time()
    preferred_bin = _resolve_external_gdal_bin_dir(gdal_bin_dir)
    env = _prepare_external_gdal_env(preferred_bin)
    gdalbuildvrt = _find_cli_tool("gdalbuildvrt", preferred_bin)
    gdal_translate = _find_cli_tool("gdal_translate", preferred_bin)
    print(f"External GDAL tools: buildvrt={gdalbuildvrt}, translate={gdal_translate}")
    if preferred_bin is None and gdal_bin_dir is None:
        print(
            "WARNING: Using GDAL tools from PATH. "
            "For reproducible performance, specify --gdal-bin-dir explicitly."
        )

    file_list_path = vrt_path.replace(".vrt", "_files.txt")
    with open(file_list_path, "w", encoding="utf-8", newline="\n") as f:
        for tile_file in tile_files:
            f.write(tile_file + "\n")

    try:
        cmd_vrt = [
            gdalbuildvrt,
            "-input_file_list",
            file_list_path,
            "-resolution",
            "highest",
            "-r",
            "nearest",
        ]
        if nodata is not None:
            nval = str(float(nodata))
            cmd_vrt.extend(["-srcnodata", nval, "-vrtnodata", nval])
        cmd_vrt.append(vrt_path)
        subprocess.run(cmd_vrt, check=True, env=env)
        print(f"External CLI VRT: {time.time() - start:.1f}s")

        cmd_cog = [
            gdal_translate,
            "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
            "--config", "GDAL_CACHEMAX", "16384",
            "-of", "COG",
            "-co", "COMPRESS=ZSTD",
            "-co", "LEVEL=1",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-co", "BLOCKSIZE=512",
            "-co", "BIGTIFF=YES",
            "-co", "OVERVIEW_RESAMPLING=AVERAGE",
            "-co", "OVERVIEW_COUNT=8",
            vrt_path,
            output_cog_path,
        ]
        subprocess.run(cmd_cog, check=True, env=env)
        _ensure_output_nodata(output_cog_path, nodata)
        elapsed = time.time() - start
        size_mb = os.path.getsize(output_cog_path) / (1024 * 1024)
        throughput = size_mb / elapsed if elapsed > 0 else 0.0
        print(f"External CLI COG complete: {elapsed:.1f}s, {size_mb:.1f}MB, {throughput:.1f}MB/s")
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)


def _build_vrt_and_cog_ultra_fast(
    tmp_tile_dir: str,
    output_cog_path: str,
    gpu_config: dict,
    backend: str = "internal",
    gdal_bin_dir: Optional[str] = None,
) -> None:
    """Build VRT from tile outputs and convert to COG."""
    print("=== Fast COG generation start ===")
    _configure_gdal_ultra_performance(gpu_config)

    vrt_path = os.path.join(tmp_tile_dir, "tiles.vrt")
    tile_files = sorted(glob.glob(os.path.join(tmp_tile_dir, "tile_*.tif")))
    if not tile_files:
        raise ValueError(f"No tile files found: {tmp_tile_dir}")

    nodata = _detect_nodata_from_tiles(tile_files)
    print(f"VRT create: merging {len(tile_files)} tiles")
    if nodata is not None:
        print(f"Detected tile nodata={nodata:g}; preserving nodata through VRT/COG")

    backend_norm = str(backend or "internal").lower()
    if backend_norm not in {"internal", "external", "auto"}:
        raise ValueError(f"Unknown COG backend: {backend}")

    use_external = False
    if backend_norm == "external":
        use_external = True
    elif backend_norm == "auto":
        try:
            _find_cli_tool("gdalbuildvrt", gdal_bin_dir)
            _find_cli_tool("gdal_translate", gdal_bin_dir)
            use_external = True
        except Exception:
            use_external = False

    if use_external:
        print("COG backend: external GDAL CLI")
        _create_vrt_and_cog_external_cli(
            tile_files=tile_files,
            vrt_path=vrt_path,
            output_cog_path=output_cog_path,
            nodata=nodata,
            gdal_bin_dir=gdal_bin_dir,
        )
    else:
        print("COG backend: internal GDAL Python")
        _create_vrt_ultra_fast(tile_files, vrt_path, nodata=nodata)
        if gdal.GetDriverByName("COG"):
            _create_cog_ultra_fast(vrt_path, output_cog_path, gpu_config, nodata=nodata)
        else:
            _create_cog_gtiff_ultra_fast(vrt_path, output_cog_path, gpu_config, nodata=nodata)

    print("=== Fast COG generation done ===")
