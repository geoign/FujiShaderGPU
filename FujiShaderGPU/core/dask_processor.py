# -*- coding: utf-8 -*-
"""
FujiShaderGPU/core/dask_processor.py
Core implementation of Dask-CUDA terrain analysis.
"""

###############################################################################
# Dependencies
###############################################################################
from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import psutil
import rasterio
from pathlib import Path
from typing import List, Tuple, Optional
from osgeo import gdal
import cupy as cp
import dask.array as da
from dask import config as dask_config
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import get_client
import xarray as xr
from tqdm.auto import tqdm


# Algorithm imports
try:
    from ..algorithms.dask_registry import ALGORITHMS
except ImportError:
    # Fallback definition when the algorithms registry is unavailable
    ALGORITHMS = {}
    logging.warning("dask registry module not found. No algorithms available.")

from ..algorithms.common.auto_params import determine_optimal_radii
from ..io.raster_info import metric_pixel_scales_from_metadata
from ..io.output_encoding import (
    SUPPORTED_OUTPUT_DTYPES,
    output_nodata_for_dtype,
    resolve_output_range,
    quantize_params,
)
from ..config.auto_tune import compute_dask_chunk
from ..utils.cpu import container_cpu_count
from ..utils.memory import (
    container_memory_available_gb,
    container_memory_total_gb,
)
from ..utils.paths import resolve_tmp_dir
from .dask_cluster import make_cluster
from .dask_io import (
    is_zarr_path,
    load_input_dataarray,
    write_zarr_output,
)

# Logging configuration
logger = logging.getLogger(__name__)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.1f}GB"


def _configure_gdal_read_performance() -> None:
    """Enable multi-threaded COG decoding and block caching for input reads.

    Call this *before* the Dask-CUDA cluster is created: ``LocalCUDACluster``
    spawns its worker (nanny) processes from this process, so they inherit the
    environment configured here, which is where the chunked input reads actually
    run.  ``setdefault`` is used so explicit user environment overrides win.
    """
    try:
        # Container-aware: psutil alone reports host RAM, so a cgroup-capped
        # worker would inherit an oversized GDAL cache and OOM on read.
        avail_gb = container_memory_available_gb()
    except Exception:
        avail_gb = 8.0
    # GDAL interprets a bare integer < 100000 as megabytes.
    cache_mb = int(max(1024, min(16384, avail_gb * 1024 * 0.1)))

    read_opts = {
        "GDAL_NUM_THREADS": "ALL_CPUS",          # multi-threaded (de)compression
        "GDAL_CACHEMAX": str(cache_mb),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "YES",
        "GDAL_BAND_BLOCK_CACHE": "HASHSET",
    }
    for key, value in read_opts.items():
        os.environ.setdefault(key, value)
        try:
            gdal.SetConfigOption(key, os.environ.get(key, value))
        except Exception:
            pass
    logger.info(
        "GDAL read performance configured: NUM_THREADS=%s, CACHEMAX=%sMB",
        os.environ.get("GDAL_NUM_THREADS"),
        os.environ.get("GDAL_CACHEMAX"),
    )


def _log_overview_availability(src_cog: str) -> None:
    """Log whether the input COG carries an overview pyramid.

    Decimated reads (global stats, terrain sampling) and downstream viewers are
    far faster when overviews exist; warn when they do not so the user can add
    them with ``gdaladdo`` ahead of time.
    """
    try:
        if is_zarr_path(src_cog):
            return
        with rasterio.open(src_cog) as src:
            ov = src.overviews(1) if src.count >= 1 else []
        if ov:
            logger.info("Input overviews available: %s (used for decimated sampling)", ov)
        else:
            logger.warning(
                "Input has no overviews; decimated stat/sampling reads will be slow. "
                "FujiShaderGPU expects an overview-bearing COG -- pre-process the input first:\n"
                "    python -m FujiShaderGPU.prepare %s prepared_cog.tif\n"
                "then run the pipeline on 'prepared_cog.tif'.",
                src_cog,
            )
    except Exception as exc:
        logger.debug("Overview availability check skipped: %s", exc)


def _select_chunk_temp_parent(data_nbytes: int) -> Path:
    """Choose and diagnose the temporary directory for chunk GeoTIFFs."""
    parent, selected_from = resolve_tmp_dir(Path(tempfile.gettempdir()))
    usage = shutil.disk_usage(parent)
    estimated = max(data_nbytes, int(data_nbytes * 0.75))
    origin = f" from ${selected_from}" if selected_from else " from tempfile default"
    logger.info(
        "Chunk temporary directory%s: %s (free=%s)",
        origin,
        parent,
        _format_gib(usage.free),
    )
    if usage.free < estimated:
        logger.warning(
            "Chunk temporary directory may be too small for COG staging "
            "(free=%s, output array=%s). Set FUJISHADER_TMP_DIR, TMPDIR, "
            "TMP, TEMP, and CPL_TMPDIR to a large persistent volume.",
            _format_gib(usage.free),
            _format_gib(data_nbytes),
        )
    if not selected_from and str(parent).startswith("/tmp") and Path("/content").exists():
        try:
            content_free = shutil.disk_usage("/content").free
        except OSError:
            content_free = 0
        if content_free > usage.free:
            logger.warning(
                "Runpod/Colab-compatible layout detected: /tmp has %s free, "
                "while /content has %s free. For large COG output set "
                "FUJISHADER_TMP_DIR=/content/fujishader_tmp and CPL_TMPDIR "
                "to the same path.",
                _format_gib(usage.free),
                _format_gib(content_free),
            )
    return parent


def _detect_metric_scales_from_dataarray(dem: xr.DataArray) -> Tuple[float, float, float, bool, Optional[float]]:
    """Detect signed x/y metric pixel scales from an xarray+rioxarray DataArray."""
    try:
        transform = dem.rio.transform()
        bounds = dem.rio.bounds()
        crs = dem.rio.crs
        sx, sy, mean_m, is_geo, lat = metric_pixel_scales_from_metadata(
            transform=transform, crs=crs, bounds=bounds
        )
        return float(sx), float(sy), float(mean_m), bool(is_geo), lat
    except Exception:
        try:
            x_res = abs(float(dem.rio.resolution()[0]))
            y_res = abs(float(dem.rio.resolution()[1]))
            mean_m = 0.5 * (x_res + y_res)
            return float(x_res), float(y_res), float(mean_m), False, None
        except Exception:
            return 1.0, 1.0, 1.0, False, None


###############################################################################
# 1. Dask-CUDA cluster
#    make_cluster() is imported directly from dask_cluster.py
###############################################################################

###############################################################################
# 3. Automatic parameter determination via terrain analysis
###############################################################################

def analyze_terrain_characteristics(dem_arr: da.Array, sample_ratio: float = 0.01,
                                    src_path: "str | None" = None) -> dict:
    """Analyze terrain characteristics holistically.

    Samples a coarse, FULL-EXTENT view of the DEM rather than a fixed center
    crop, so datasets whose valid data is off-center or sparse (e.g. an
    L-shaped footprint whose raster center is entirely NoData) are handled
    correctly.  When the source COG path is provided the sample is read from a
    low-resolution overview (cheap, full-extent); otherwise a strided
    decimation of the in-memory array is used.  The decimation factor ``decim``
    is divided back out of the gradient so ``mean_slope`` stays in
    per-original-pixel units and remains comparable across resolutions.
    """
    h, w = dem_arr.shape
    sample = None
    decim = 1

    # Preferred: a low-resolution, full-extent overview from the source COG.
    if src_path is not None and not is_zarr_path(src_path):
        try:
            from rasterio.enums import Resampling
            target = 2048
            with rasterio.open(src_path) as src:
                ovr = src.overviews(1)
                long_side = max(h, w)
                # Align to a real overview factor when possible (no extra
                # resampling / NoData blending); else use the plain ratio.
                cand = [f for f in ovr if f >= 1] or [max(1, long_side // target)]
                decim = min(cand, key=lambda f: abs((long_side // max(f, 1)) - target))
                decim = max(1, int(decim))
                out_h = max(1, h // decim)
                out_w = max(1, w // decim)
                ov = src.read(1, out_shape=(out_h, out_w),
                              resampling=Resampling.nearest, masked=True)
                src_nodata = src.nodata
            data = np.ma.getdata(ov).astype(np.float32, copy=False)
            mask = np.ma.getmaskarray(ov)
            arr = cp.asarray(data)
            arr = cp.where(cp.asarray(mask), cp.asarray(np.float32(np.nan)), arr)
            if src_nodata is not None and np.isfinite(src_nodata):
                arr = cp.where(arr == cp.float32(src_nodata),
                               cp.asarray(np.float32(np.nan)), arr)
            sample = arr
        except Exception:
            sample = None
            decim = 1

    # Fallback: strided full-extent decimation of the in-memory array.
    if sample is None:
        decim = max(1, min(h, w) // 2048)
        sample = dem_arr[::decim, ::decim].compute()

    # Basic statistics (shared part)
    valid_mask = ~cp.isnan(sample)
    if not valid_mask.any():
        raise ValueError("No valid elevation data found")

    elevations = sample[valid_mask]
    stats = {
        'elevation_range': float(cp.ptp(elevations)),
        'std_dev': float(cp.std(elevations)),
        'sample_size': sample.shape
    }

    # Gradient computation (shared part). NoData boundaries produce NaN
    # gradients, so aggregate over finite slope values only.  Divide by the
    # decimation factor to recover per-original-pixel slope magnitude.
    dy, dx = cp.gradient(sample)
    slope = cp.sqrt(dy**2 + dx**2) / float(max(decim, 1))
    valid_slope = slope[valid_mask]
    valid_slope = valid_slope[cp.isfinite(valid_slope)]
    if valid_slope.size == 0:
        stats['mean_slope'] = 0.0
        stats['max_slope'] = 0.0
    else:
        stats['mean_slope'] = float(cp.mean(valid_slope))
        stats['max_slope'] = float(cp.percentile(valid_slope, 95))

    # Common metrics for auto-parameter estimation
    stats["complexity_score"] = float(stats["mean_slope"] * stats["std_dev"])
    return stats




###############################################################################
# 4. Direct COG output (GDAL >= 3.8) - improved
###############################################################################

def get_cog_options(dtype: str) -> dict:
    """Return optimal COG options for the given data type."""
    base_options = {
        "COMPRESS": "ZSTD",
        "LEVEL": "1",
        "BLOCKSIZE": "512",
        "OVERVIEWS": "IGNORE_EXISTING",
        "OVERVIEW_COMPRESS": "ZSTD",
        "OVERVIEW_RESAMPLING": "AVERAGE",
        "OVERVIEW_COUNT": "8",
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }
    
    # Set PREDICTOR according to data type
    if dtype in ['float32', 'float64']:
        base_options["PREDICTOR"] = "3"  # floating point
    elif dtype in ['int16', 'int32', 'uint16', 'uint32']:
        base_options["PREDICTOR"] = "2"  # integer
    # Do not use PREDICTOR for uint8 or other types
    
    return base_options

def check_gdal_version() -> tuple:
    """Check the GDAL version (parse safely with integer arithmetic)."""
    ver_num = int(gdal.VersionInfo("VERSION_NUM"))
    major = ver_num // 1_000_000
    minor = (ver_num % 1_000_000) // 10_000
    return major, minor


def _get_overview_count(tiff_path: Path) -> int:
    """Return the overview count on band 1, or 0 when the file cannot be read."""
    ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)
    if ds is None or ds.RasterCount < 1:
        return 0
    band = ds.GetRasterBand(1)
    overview_count = band.GetOverviewCount() if band is not None else 0
    ds = None
    return int(overview_count)


def _assert_has_overviews(tiff_path: Path):
    """Fail fast when COG creation unexpectedly omitted its overview pyramid."""
    overview_count = _get_overview_count(tiff_path)
    if overview_count <= 0:
        raise ValueError(f"COG output has no overviews: {tiff_path}")
    logger.info("COG overview count: %d", overview_count)


def _ensure_cog_has_overviews(dst: Path, cog_options: dict):
    """Rebuild a COG in-place when the writer produced no overviews."""
    if _get_overview_count(dst) > 0:
        _assert_has_overviews(dst)
        return

    logger.warning("COG output has no overviews; rebuilding with forced ZSTD overviews")
    src = dst.with_suffix(".no_overviews.tif")
    repaired = dst.with_suffix(".with_overviews.tif")
    src.unlink(missing_ok=True)
    repaired.unlink(missing_ok=True)
    dst.replace(src)
    try:
        build_cog_with_overviews(src, repaired, cog_options)
        repaired.replace(dst)
        _assert_has_overviews(dst)
    finally:
        src.unlink(missing_ok=True)
        repaired.unlink(missing_ok=True)


def _build_zstd_overviews(path: Path, cog_options: dict):
    """Build internal ZSTD-compressed overviews on a temporary GeoTIFF."""
    logger.info("Building ZSTD-compressed overviews: %s", path)
    previous_options = {
        "COMPRESS_OVERVIEW": gdal.GetConfigOption("COMPRESS_OVERVIEW"),
        "ZLEVEL_OVERVIEW": gdal.GetConfigOption("ZLEVEL_OVERVIEW"),
        "BIGTIFF_OVERVIEW": gdal.GetConfigOption("BIGTIFF_OVERVIEW"),
        "GDAL_TIFF_OVR_BLOCKSIZE": gdal.GetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE"),
        "GDAL_NUM_THREADS": gdal.GetConfigOption("GDAL_NUM_THREADS"),
    }
    try:
        gdal.SetConfigOption("COMPRESS_OVERVIEW", cog_options.get("OVERVIEW_COMPRESS", "ZSTD"))
        gdal.SetConfigOption("ZLEVEL_OVERVIEW", cog_options.get("LEVEL", "1"))
        gdal.SetConfigOption("BIGTIFF_OVERVIEW", "YES")
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", cog_options.get("BLOCKSIZE", "512"))
        gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            raise ValueError(f"Failed to open temporary TIFF for overviews: {path}")
        result = ds.BuildOverviews(
            cog_options.get("OVERVIEW_RESAMPLING", "AVERAGE"),
            [2, 4, 8, 16, 32, 64, 128, 256],
        )
        ds = None
        if result != 0:
            raise ValueError(f"BuildOverviews failed: {path}")
    finally:
        for key, value in previous_options.items():
            gdal.SetConfigOption(key, value)


def _write_cog_da_original(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """Save a DataArray directly as a COG (with progress display)."""
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
                    # More detailed progress display
                    logger.info("Computing result chunks...")
                    # Progress display using tqdm
                    class TqdmCallback(Callback):
                        def __init__(self):
                            self.tqdm = None
                            
                        def _start(self, dsk):
                            self.tqdm = tqdm(total=len(dsk), desc='Computing', unit='tasks')
                            
                        def _posttask(self, key, result, dsk, _state, _worker_id):
                            self.tqdm.update(1)
                            
                        def _finish(self, dsk, _state, _failed):
                            self.tqdm.close()
                    
                    with TqdmCallback():
                        computed_data = data.compute()
                else:
                    computed_data = data.compute()
                
                # Wrap the computed data back into xarray
                computed_da = xr.DataArray(
                    computed_data,
                    dims=data.dims,
                    coords=data.coords,
                    attrs=data.attrs,
                    name=data.name
                )
                
                # Carry over CRS information
                if hasattr(data, 'rio') and data.rio.crs is not None:
                    computed_da.rio.write_crs(data.rio.crs, inplace=True)
                # Set NoData explicitly on the output (prevents a black border): 0 for integer output, NaN for float.
                _nd = output_nodata_for_dtype(computed_da.dtype)
                computed_da.rio.write_nodata(_nd, inplace=True)

                # Write the COG
                logger.info("Writing to COG...")
                computed_da.rio.to_raster(
                    dst,
                    driver="COG",
                    **cog_options,
                )
                _ensure_cog_has_overviews(dst, cog_options)
            
            size_mb = os.path.getsize(dst) / 2**20
            logger.info(f"[OK] COG written: {dst} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.warning(f"COG driver failed: {e}, falling back to gdal_translate")
            _fallback_cog_write(data, dst, cog_options)
    else:
        logger.warning(f"GDAL {major}.{minor} < 3.8, using fallback method")
        _fallback_cog_write(data, dst, cog_options)

def _write_cog_da_chunked_impl(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """Chunked write implementation for large datasets."""
    major, minor = check_gdal_version()
    use_cog_driver = major > 3 or (major == 3 and minor >= 8)
    dtype_str = str(data.dtype)
    cog_options = get_cog_options(dtype_str)
    
    # Decide memory release from actual free VRAM (real VRAM usage, not pool free_bytes)
    try:
        _free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        available_vram_gb = _free_bytes / (1024**3)
    except Exception:
        available_vram_gb = 10.0  # safe default
    if available_vram_gb < 10:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        try:
            client = get_client()
            client.run(lambda: gc.collect())
        except Exception:
            pass

    # Check free DRAM and enable prefetch (container-aware, not host RAM)
    available_ram_gb = container_memory_available_gb()

    if available_ram_gb > 20:  # when more than 20GB is free
        logger.info(f"Enabling chunk prefetching (available DRAM: {available_ram_gb:.1f}GB)")
        
        # Set Dask scheduler hints.  NOTE: keep blockwise fusion ON (the dask
        # default).  Disabling it (optimization.fuse.active=False) leaves every
        # intermediate as a separate task whose GPU result lingers until all
        # consumers finish; for deep graphs (e.g. visual_saliency / fractal_anomaly
        # combine 6 per-scale fields) this makes device memory grow monotonically
        # to tens of GB and exhaust the RMM pool.  Fusion frees per-block
        # intermediates immediately, bounding peak VRAM.
        prefetch_config = {
            "distributed.worker.memory.pause": 0.90,  # allow up to 90% memory usage
            "distributed.worker.memory.spill": 0.95,  # spill at 95%
        }
    else:
        logger.info(f"Prefetching disabled (available DRAM: {available_ram_gb:.1f}GB < 20GB)")
        prefetch_config = {}
    
    # Apply prefetch settings and run chunk processing
    with dask_config.set(prefetch_config):

        # Create a temporary directory
        tmp_parent = _select_chunk_temp_parent(int(data.nbytes))
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            # Process per chunk
            chunk_files = []

            # Fall back to normal processing when not a Dask array
            if not hasattr(data.data, 'to_delayed'):
                logger.info("Data is not chunked with Dask, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return

            # Validate chunk information
            if not hasattr(data, 'chunks') or data.chunks is None:
                logger.warning("No chunk information found, falling back to regular processing")
                _write_cog_da_original(data, dst, show_progress)
                return

            delayed_chunks = data.data.to_delayed()
            n_rows = int(delayed_chunks.shape[0])
            n_cols = int(delayed_chunks.shape[1])
            total_chunks = n_rows * n_cols

            y_dim, x_dim = data.dims[0], data.dims[1]
            src_crs = data.rio.crs if hasattr(data, 'rio') else None

            def _persist_chunk(idx: int, i: int, j: int, chunk_data) -> Path:
                """Write one computed chunk to a temporary GeoTIFF; return its path."""
                chunk_height, chunk_width = chunk_data.shape[0], chunk_data.shape[1]
                y_start = sum(data.chunks[0][:i])
                x_start = sum(data.chunks[1][:j])
                y_end = y_start + chunk_height
                x_end = x_start + chunk_width
                chunk_da = xr.DataArray(
                    chunk_data,
                    dims=data.dims,
                    coords={
                        y_dim: data.coords[y_dim].isel({y_dim: slice(y_start, y_end)}),
                        x_dim: data.coords[x_dim].isel({x_dim: slice(x_start, x_end)}),
                    },
                    attrs=data.attrs,
                )
                if src_crs is not None:
                    chunk_da.rio.write_crs(src_crs, inplace=True)
                chunk_file = Path(tmpdir) / f"chunk_{idx}.tif"
                chunk_da.rio.to_raster(
                    chunk_file,
                    driver="GTiff",
                    compress="ZSTD",
                    zstd_level=1,
                    predictor=int(cog_options.get("PREDICTOR", 1)),
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    BIGTIFF="YES",
                    num_threads="ALL_CPUS",
                )
                del chunk_da
                return chunk_file

            # Parallel streaming write:
            # The chunk graph is embarrassingly parallel, so multiple Dask-CUDA
            # workers (= GPUs) compute different chunks in parallel while the client
            # writes the finished ones. Fall back to serial when no distributed client is available.
            client = None
            try:
                client = get_client()
            except Exception:
                client = None

            if client is not None:
                try:
                    n_workers = max(1, len(client.scheduler_info().get("workers", {})))
                except Exception:
                    n_workers = 1
                max_inflight = max(2, n_workers * 2)
                from distributed import as_completed as _as_completed

                coords_flat = [(i, j) for i in range(n_rows) for j in range(n_cols)]
                task_iter = iter(enumerate(coords_flat))
                fut_meta = {}
                inflight = _as_completed()

                def _submit_next() -> bool:
                    try:
                        idx, (i, j) = next(task_iter)
                    except StopIteration:
                        return False
                    fut = client.compute(delayed_chunks[i, j])
                    fut_meta[fut] = (idx, i, j)
                    inflight.add(fut)
                    return True

                logger.info(
                    "Parallel chunk write: %d chunks, %d worker(s), up to %d in flight",
                    total_chunks, n_workers, max_inflight,
                )
                done = 0
                with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                    for _ in range(min(max_inflight, total_chunks)):
                        if not _submit_next():
                            break
                    for fut in inflight:
                        idx, i, j = fut_meta.pop(fut)
                        try:
                            chunk_data = fut.result()
                            chunk_files.append(_persist_chunk(idx, i, j, chunk_data))
                            del chunk_data
                        except Exception as e:
                            logger.error(f"Failed to process chunk {i},{j}: {e}")
                            raise
                        finally:
                            del fut
                        done += 1
                        if done % 10 == 0:
                            try:
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass
                        pbar.update(1)
                        pbar.set_postfix({"saved": f"{len(chunk_files)}"})
                        _submit_next()
            else:
                # Serial fallback (when no distributed client is available)
                idx = 0
                with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                    for i in range(n_rows):
                        for j in range(n_cols):
                            try:
                                chunk_data = delayed_chunks[i, j].compute()
                                chunk_files.append(_persist_chunk(idx, i, j, chunk_data))
                                del chunk_data
                            except Exception as e:
                                logger.error(f"Failed to process chunk {i},{j}: {e}")
                                raise
                            idx += 1
                            if idx % 10 == 0:
                                cp.get_default_memory_pool().free_all_blocks()
                            pbar.update(1)
                
            # Consolidate via VRT -> single intermediate GeoTIFF -> overviews -> COG
            if not chunk_files:
                raise ValueError("No chunks were successfully processed")

            logger.info(f"Creating VRT from {len(chunk_files)} chunks...")
            vrt_file = Path(tmpdir) / "merged.vrt"
            gdal.BuildVRT(str(vrt_file), [str(f) for f in chunk_files])

            # GDAL progress callback
            pbar = tqdm(total=100, desc="Consolidating", unit="%")
            def gdal_progress_callback(complete, _message, _cb_data):
                pbar.n = int(complete * 100)
                pbar.refresh()
                if complete >= 1.0:
                    pbar.close()
                return 1

            # Reduce disk footprint: consolidate the chunks into one intermediate GeoTIFF,
            # then free the chunks (= one full-resolution copy) and the VRT before the
            # heavy overview/COG stage. Otherwise three large copies "chunks + intermediate
            # + final COG" coexist on the same disk and the peak grows to ~3.66x
            # (translating the VRT directly to COG is the same, since the driver makes an internal temp file).
            # Deleting the chunks after consolidation keeps the peak at ~2.66x.
            merged_tif = Path(tmpdir) / "merged_tmp.tif"
            logger.info("Consolidating %d chunks into single GeoTIFF: %s",
                        len(chunk_files), merged_tif)
            consolidate_options = [
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COMPRESS=ZSTD",
                f"ZLEVEL={cog_options.get('LEVEL', '1')}",
                "BIGTIFF=YES",
                "NUM_THREADS=ALL_CPUS",
            ]
            if 'PREDICTOR' in cog_options:
                consolidate_options.append(f"PREDICTOR={cog_options['PREDICTOR']}")

            result = gdal.Translate(
                str(merged_tif),
                str(vrt_file),
                format="GTiff",
                creationOptions=consolidate_options,
                callback=gdal_progress_callback,
            )
            if result is None:
                raise ValueError("Chunk consolidation failed")
            result = None

            # Set NoData explicitly on the intermediate GeoTIFF so the (AVERAGE) overviews
            # exclude NoData and the final COG inherits the nodata tag (prevents a black border).
            # 0 for integer output, NaN for float.
            _mds = gdal.Open(str(merged_tif), gdal.GA_Update)
            if _mds is not None:
                try:
                    _mds.GetRasterBand(1).SetNoDataValue(
                        float(output_nodata_for_dtype(data.dtype))
                    )
                finally:
                    _mds = None

            # The full-resolution data is now in merged_tmp.tif, so delete the chunks and VRT
            # immediately to free disk before the heavy stage (overviews + COG).
            freed = 0
            for f in chunk_files:
                try:
                    freed += f.stat().st_size
                    f.unlink()
                except OSError:
                    pass
            try:
                vrt_file.unlink()
            except OSError:
                pass
            n_done = len(chunk_files)
            chunk_files.clear()
            logger.info("Freed %s of chunk staging before COG build",
                        _format_gib(freed))

            # Single GeoTIFF -> in-place overviews -> COG (FORCE_USE_EXISTING)
            logger.info("Converting consolidated GeoTIFF to COG format...")
            build_cog_with_overviews(merged_tif, dst, cog_options)

            # Free the intermediate GeoTIFF too (reserve room for the final output without waiting for TemporaryDirectory cleanup)
            try:
                merged_tif.unlink()
            except OSError:
                pass

            logger.info(f"Successfully created COG from {n_done} chunks")

def _dask_worker_memory_limit_gb() -> Optional[float]:
    """Return the smallest connected Dask worker memory limit in GB.

    This reflects the real container/cgroup cap (RunPod/Colab/k8s) that bounds
    any single-worker gather, unlike ``psutil`` which reports host RAM.  Falls
    back to ``dask.system.MEMORY_LIMIT`` (also cgroup-aware) and finally None.
    """
    try:
        client = get_client()
        limits = [
            w.get("memory_limit")
            for w in client.scheduler_info()["workers"].values()
        ]
        limits = [float(x) for x in limits if x]
        if limits:
            return min(limits) / (1024**3)
    except Exception:
        pass
    try:
        from dask.system import MEMORY_LIMIT  # cgroup-aware byte count
        if MEMORY_LIMIT:
            return float(MEMORY_LIMIT) / (1024**3)
    except Exception:
        pass
    return None


def write_cog_da_chunked(data: xr.DataArray, dst: Path, show_progress: bool = True):
    """Write COG (auto-selected by available memory)."""
    total_gb = data.nbytes / (1024**3)
    
    # Auto-determine the threshold from system memory.
    #
    # IMPORTANT: psutil reports the *host* RAM, not the container's cgroup limit.
    # On RunPod / Colab / k8s the Dask worker is capped far below the host total
    # (e.g. host 503GB but worker memory_limit 46.6GiB), and the direct-write
    # path below gathers the ENTIRE result onto a *single* worker via one
    # ``finalize`` task.  If we size the threshold from host RAM we wrongly pick
    # direct writing and the worker OOM-kills (signal 9) in a recompute loop.
    # So clamp the available figure to the real per-worker limit.
    mem_info = psutil.virtual_memory()
    total_ram_gb = mem_info.total / (1024**3)
    host_avail_gb = mem_info.available / (1024**3)

    worker_limit_gb = _dask_worker_memory_limit_gb()
    if worker_limit_gb is not None:
        available_ram_gb = min(host_avail_gb, worker_limit_gb)
    else:
        available_ram_gb = host_avail_gb

    # Detect the Google Colab environment
    is_colab = 'google.colab' in sys.modules
    
    # Set the safety factor
    if is_colab:
        # Colab is conservative (40% of available memory)
        safety_factor = 0.4
        # but clamped to a minimum of 20GB and a maximum of 60GB
        min_threshold = 20
        max_threshold = 60
    else:
        # Local environments are a bit more aggressive (60%)
        safety_factor = 0.6
        min_threshold = 30
        max_threshold = 100
    
    # Compute the threshold
    # Base it on currently available memory (more realistic)
    chunk_threshold = available_ram_gb * safety_factor
    
    # Clamp to range
    chunk_threshold = max(min_threshold, min(chunk_threshold, max_threshold))
    
    # The direct-write path peaks at roughly 2-3x the output size on ONE worker
    # (persisted intermediate + per-chunk asnumpy host copies + the contiguous
    # finalize concat), so never let the threshold exceed what a single worker
    # can actually hold for that peak.
    if worker_limit_gb is not None:
        single_worker_direct_cap = worker_limit_gb * 0.85 / 3.0
        chunk_threshold = min(chunk_threshold, single_worker_direct_cap)

    # Log output
    logger.info(f"System RAM: {total_ram_gb:.1f}GB total, {host_avail_gb:.1f}GB available")
    if worker_limit_gb is not None:
        logger.info(f"Dask worker memory limit: {worker_limit_gb:.1f}GB (cgroup-aware)")
    logger.info(f"Memory threshold: {chunk_threshold:.1f}GB (safety factor: {safety_factor*100:.0f}%)")
    
    # Also display GPU info for reference
    try:
        meminfo = cp.cuda.runtime.memGetInfo()
        vram_free_gb = meminfo[0] / (1024**3)
        vram_total_gb = meminfo[1] / (1024**3)
        logger.info(f"GPU VRAM: {vram_total_gb:.1f}GB total, {vram_free_gb:.1f}GB free")
    except Exception:
        pass
    
    # Compare data size against the threshold
    if total_gb > chunk_threshold:
        logger.info(f"Large dataset ({total_gb:.1f}GB) > threshold ({chunk_threshold:.1f}GB)")
        logger.info("Using chunked writing to avoid memory issues")
        _write_cog_da_chunked_impl(data, dst, show_progress)
    else:
        logger.info(f"Dataset ({total_gb:.1f}GB) <= threshold ({chunk_threshold:.1f}GB)")
        logger.info("Using direct writing for better performance")
        _write_cog_da_original(data, dst, show_progress)

def _fallback_cog_write(data: xr.DataArray, dst: Path, cog_options: dict):
    """Fallback: create the COG via a temporary file."""
    tmp = dst.with_suffix(".tmp.tif")
    try:
        # Exclude COG-specific options
        tiff_options = {
            k: v for k, v in cog_options.items()
            if k not in [
                'OVERVIEWS',
                'OVERVIEW_COMPRESS',
                'OVERVIEW_COUNT',
                'OVERVIEW_RESAMPLING',
            ]
        }
        
        # Compute with progress display, then write
        logger.info("Computing result...")
        with ProgressBar():
            computed_data = data.compute()
        
        # Wrap the computed data back into xarray
        computed_da = xr.DataArray(
            computed_data,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs,
            name=data.name
        )
        
        # Carry over CRS information
        if hasattr(data, 'rio') and data.rio.crs is not None:
            computed_da.rio.write_crs(data.rio.crs, inplace=True)
        # Set NoData explicitly (0 for integer output, NaN for float). The (AVERAGE) overviews
        # exclude NoData and the final COG inherits the nodata tag.
        computed_da.rio.write_nodata(
            output_nodata_for_dtype(computed_da.dtype), inplace=True,
        )

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
# 5. gdal_translate/gdaladdo fallback functions
###############################################################################

def build_cog_with_overviews(src: Path, dst: Path, cog_options: dict):
    """For older GDAL: build overviews on a temporary TIFF, then convert to COG."""
    # Container-aware CPU budget for GDAL threads: os.cpu_count() reports the
    # host (e.g. 64) while a cgroup CFS quota may cap us at ~7, so the host count
    # would oversubscribe the quota.
    num_cpus = container_cpu_count()

    _build_zstd_overviews(src, cog_options)
    _assert_has_overviews(src)
    logger.info("Converting overview-backed temporary TIFF to COG: %s", dst)

    translate_options = [
        f"COMPRESS={cog_options.get('COMPRESS', 'ZSTD')}",
        f"LEVEL={cog_options.get('LEVEL', '1')}",
        f"OVERVIEW_COMPRESS={cog_options.get('OVERVIEW_COMPRESS', 'ZSTD')}",
        f"BLOCKSIZE={cog_options.get('BLOCKSIZE', '512')}",
        "BIGTIFF=YES",
        f"NUM_THREADS={num_cpus}",
        "OVERVIEWS=FORCE_USE_EXISTING",
        f"OVERVIEW_RESAMPLING={cog_options.get('OVERVIEW_RESAMPLING', 'AVERAGE')}",
    ]
    
    # Add PREDICTOR only when present
    if 'PREDICTOR' in cog_options:
        translate_options.append(f"PREDICTOR={cog_options['PREDICTOR']}")

    result = gdal.Translate(
        str(dst),
        str(src),
        format="COG",
        creationOptions=translate_options,
    )
    if result is None:
        raise ValueError(f"COG translate failed: {dst}")
    result = None
    _assert_has_overviews(dst)

###############################################################################
# 6. Main pipeline
###############################################################################

def validate_inputs(src_cog: str):
    """Validate input parameters."""
    if not Path(src_cog).exists():
        raise FileNotFoundError(f"Input file not found: {src_cog}")


def _compute_rvi_global_stats_from_overview(
    src_cog: str,
    *,
    radii: List[int],
    weights: Optional[List[float]],
    pixel_size: float,
    sample_max: int = 2048,
) -> Optional[tuple]:
    """Estimate the RVI normalization scale from a decimated overview read.

    Striding the full-resolution Dask array forces every chunk (the entire
    dataset) to be read and copied to the GPU before any write progress is
    visible, which stalls on very large rasters.  A decimated rasterio read uses
    the COG overview pyramid and returns a representative full-extent sample at a
    tiny fraction of the cost -- mirroring the tile backend's strategy.
    """
    try:
        from ..algorithms._impl_rvi import compute_rvi_efficient_block
        from ..algorithms._normalization import rvi_stat_func
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("RVI overview stats helpers unavailable: %s", exc)
        return None

    if not radii:
        return None

    try:
        with rasterio.open(src_cog) as src:
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale))
            sample_h = max(128, int(src.height / scale))
            sample_ma = src.read(
                1,
                out_shape=(sample_h, sample_w),
                resampling=Resampling.nearest,
                out_dtype=np.float32,
                masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)
            nodata = src.nodata

        if nodata is not None and not np.isnan(float(nodata)):
            sample = np.where(
                np.isclose(sample, float(nodata), rtol=0.0, atol=1e-6),
                np.nan,
                sample,
            ).astype(np.float32, copy=False)

        sample_pixel_size = float(pixel_size) * float(scale)
        scaled_radii = [max(1, int(round(float(r) / scale))) for r in radii]

        sample_gpu = cp.asarray(sample, dtype=cp.float32)
        rvi_sample = compute_rvi_efficient_block(
            sample_gpu,
            radii=scaled_radii,
            weights=weights,
            pixel_size=sample_pixel_size,
        )
        stats = rvi_stat_func(rvi_sample)
        if not stats or not np.isfinite(float(stats[0])) or float(stats[0]) <= 1e-9:
            return None
        logger.info(
            "RVI global stats from overview: decimation=%.1fx, radii=%s -> %s, abs_p80=%.6f",
            scale,
            list(radii),
            scaled_radii,
            float(stats[0]),
        )
        return stats
    except Exception as exc:
        logger.warning(
            "Failed to compute RVI overview stats; falling back to window sampling: %s",
            exc,
        )
        return None


def _compute_rvi_overview_coarse_field(
    src_cog: str,
    *,
    large_radii: List[int],
    large_weights: List[float],
    sample_max: int = 2048,
):
    """Compute the large-radius RVI coarse field (Sum w*mean) from the COG overview.

    Reading the stored overview (decimated read) avoids materialising the huge
    full-resolution halo a large radius would otherwise require.  The returned
    CuPy field is sampled per-block as ``W_large*block - upsample(field)``.
    Returns ``None`` on any failure so the caller transparently falls back to the
    full-resolution radii path.
    """
    if not large_radii:
        return None
    try:
        from ..algorithms._impl_rvi import compute_rvi_large_coarse_field
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("RVI overview coarse-field helpers unavailable: %s", exc)
        return None
    try:
        with rasterio.open(src_cog) as src:
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            sample_w = max(128, int(src.width / scale))
            sample_h = max(128, int(src.height / scale))
            sample_ma = src.read(
                1, out_shape=(sample_h, sample_w), resampling=Resampling.average,
                out_dtype=np.float32, masked=True,
            )
            sample = sample_ma.filled(np.nan).astype(np.float32, copy=False)
            nodata = src.nodata
        if nodata is not None and not np.isnan(float(nodata)):
            sample = np.where(
                np.isclose(sample, float(nodata), rtol=0.0, atol=1e-6), np.nan, sample
            ).astype(np.float32, copy=False)
        coarse_dem = cp.asarray(sample, dtype=cp.float32)
        field = compute_rvi_large_coarse_field(
            coarse_dem, large_radii=large_radii, large_weights=large_weights,
            decimation=float(scale),
        )
        logger.info(
            "RVI large-radius overview field: decimation=%.1fx, large_radii=%s",
            scale, list(large_radii),
        )
        return field
    except Exception as exc:
        logger.warning("Failed to compute RVI overview coarse field: %s", exc)
        return None


def _compute_fractal_relief_stats(
    src_cog: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
) -> Optional[tuple]:
    """Global roughness (relief) p10/p75 for fractal_anomaly, from full-resolution
    stratified tiles.  ``relief_conf`` otherwise uses a per-block roughness
    percentile, which differs tile-to-tile and produces tile-boundary seams; a
    single global pair makes it consistent across the whole raster.  Computed at
    full resolution so the magnitude matches the per-block roughness."""
    try:
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        from ..algorithms._impl_fractal_anomaly import compute_roughness_multiscale
        from ..algorithms.dask_registry import ALGORITHMS
    except Exception as exc:
        logger.warning("fractal relief-stats helpers unavailable: %s", exc)
        return None
    try:
        defaults = ALGORITHMS["fractal_anomaly"].get_default_params() or {}
    except Exception:
        defaults = {}
    radii = (params or {}).get("radii") or defaults.get("radii") or [4, 8, 16, 32, 64]
    if len(radii) < 5:
        radii = [4, 8, 16, 32, 64]
    max_r = max(float(r) for r in radii)
    try:
        margin = int(min(2 * max_r + 16, max_tile // 4))
        tile = int(min(max_tile, max(2048, 4 * margin)))
        pooled = []
        with rasterio.open(src_cog) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            def _dn(a):
                a = a.astype(np.float32, copy=False)
                if nodata is not None and not np.isnan(float(nodata)):
                    a = np.where(np.isclose(a, float(nodata), atol=1e-6), np.nan, a)
                return a

            cov = max(1, max(W, H) // 512)
            ov = _dn(src.read(1, out_shape=(max(1, H // cov), max(1, W // cov)),
                              resampling=Resampling.nearest, out_dtype=np.float32,
                              masked=True).filled(np.nan))
            vmask = np.isfinite(ov)
            if not vmask.any():
                return None
            ys, xs = np.where(vmask)
            by0, by1 = int(ys.min()) * cov, min(H, (int(ys.max()) + 1) * cov)
            bx0, bx1 = int(xs.min()) * cov, min(W, (int(xs.max()) + 1) * cov)
            ch_, cw_ = max(1, (by1 - by0) // grid), max(1, (bx1 - bx0) // grid)
            for gy in range(grid):
                for gx in range(grid):
                    ccy = by0 + gy * ch_ + ch_ // 2
                    ccx = bx0 + gx * cw_ + cw_ // 2
                    wy0 = int(min(max(0, ccy - tile // 2), max(0, H - tile)))
                    wx0 = int(min(max(0, ccx - tile // 2), max(0, W - tile)))
                    tw, th = min(tile, W - wx0), min(tile, H - wy0)
                    a = _dn(src.read(1, window=Window(wx0, wy0, tw, th),
                                     out_dtype=np.float32, masked=True).filled(np.nan))
                    if float(np.isfinite(a).mean()) < min_valid_frac:
                        continue
                    g = cp.asarray(a)
                    sig = compute_roughness_multiscale(g, radii, window_mult=3, detrend=True)
                    rough = cp.mean(sig, axis=2)
                    m = int(min(margin, rough.shape[0] // 3, rough.shape[1] // 3))
                    if m > 0:
                        rough = rough[m:-m, m:-m]
                    vals = rough[~cp.isnan(rough)]
                    if vals.size:
                        pooled.append(cp.asnumpy(vals))
                    del g, sig, rough, vals
                    cp.get_default_memory_pool().free_all_blocks()
        if not pooled:
            return None
        allv = np.concatenate(pooled)
        p10 = float(np.percentile(allv, 10))
        p75 = float(np.percentile(allv, 75))
        if not (np.isfinite(p10) and np.isfinite(p75) and p75 > p10):
            return None
        logger.info("fractal_anomaly global relief: p10=%.4g p75=%.4g (from %d tiles)",
                    p10, p75, len(pooled))
        return (p10, p75)
    except Exception as exc:
        logger.warning("Failed to compute fractal relief stats: %s", exc)
        return None


def _compute_npr_grad_stats(
    src_cog: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
    small_radius_max: float = 600.0,
) -> Optional[dict]:
    """Per-radius GLOBAL gradient (base, range, mean) for npr_edges, from full-res
    stratified tiles.  npr's edge threshold is otherwise computed per block, which
    differs tile-to-tile and seams.  Only the small (full-res, multi-tile) radii
    need this; large radii run the whole coarsened grid as one block and are
    already global.  Returns {round(radius): (base, range, mean)}."""
    try:
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        from ..algorithms._impl_npr_edges import compute_npr_edges_spatial_block
        from ..algorithms._base import classify_resolution
    except Exception as exc:
        logger.warning("npr grad-stats helpers unavailable: %s", exc)
        return None
    radii = (params or {}).get("radii") or []
    small = [float(r) for r in radii if float(r) <= small_radius_max]
    if not small:
        return None
    pixel_size = float(params.get("pixel_size", 1.0))
    edge_sigma = float(params.get("edge_sigma", 1.0))
    tl = float(params.get("threshold_low", 0.1))
    th_ = float(params.get("threshold_high", 0.3))
    low_res = classify_resolution(pixel_size) in ("low", "very_low", "ultra_low")
    try:
        margin = int(min(2 * max(small) + 16, max_tile // 4))
        tile = int(min(max_tile, max(2048, 4 * margin)))
        tiles = []
        with rasterio.open(src_cog) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            def _dn(a):
                a = a.astype(np.float32, copy=False)
                if nodata is not None and not np.isnan(float(nodata)):
                    a = np.where(np.isclose(a, float(nodata), atol=1e-6), np.nan, a)
                return a

            cov = max(1, max(W, H) // 512)
            ov = _dn(src.read(1, out_shape=(max(1, H // cov), max(1, W // cov)),
                              resampling=Resampling.nearest, out_dtype=np.float32,
                              masked=True).filled(np.nan))
            vmask = np.isfinite(ov)
            if not vmask.any():
                return None
            ys, xs = np.where(vmask)
            by0, by1 = int(ys.min()) * cov, min(H, (int(ys.max()) + 1) * cov)
            bx0, bx1 = int(xs.min()) * cov, min(W, (int(xs.max()) + 1) * cov)
            ch_, cw_ = max(1, (by1 - by0) // grid), max(1, (bx1 - bx0) // grid)
            for gy in range(grid):
                for gx in range(grid):
                    ccy = by0 + gy * ch_ + ch_ // 2
                    ccx = bx0 + gx * cw_ + cw_ // 2
                    wy0 = int(min(max(0, ccy - tile // 2), max(0, H - tile)))
                    wx0 = int(min(max(0, ccx - tile // 2), max(0, W - tile)))
                    tw, th2 = min(tile, W - wx0), min(tile, H - wy0)
                    a = _dn(src.read(1, window=Window(wx0, wy0, tw, th2),
                                     out_dtype=np.float32, masked=True).filled(np.nan))
                    if float(np.isfinite(a).mean()) >= min_valid_frac:
                        tiles.append(cp.asarray(a))
        if not tiles:
            return None
        out = {}
        for r in small:
            pool = []
            for g in tiles:
                grad = compute_npr_edges_spatial_block(
                    g, edge_sigma=edge_sigma, threshold_low=tl, threshold_high=th_,
                    pixel_size=pixel_size, radius=r, _return_grad=True)
                m = int(min(int(2 * r + 16), grad.shape[0] // 3, grad.shape[1] // 3))
                if m > 0:
                    grad = grad[m:-m, m:-m]
                v = grad[~cp.isnan(grad)]
                if v.size:
                    pool.append(cp.asnumpy(v))
                del grad, v
            cp.get_default_memory_pool().free_all_blocks()
            if not pool:
                continue
            allv = np.concatenate(pool)
            mean = float(np.mean(allv))
            if low_res:
                base, rng = mean, float(np.std(allv)) * 1.5
            else:
                base = float(np.percentile(allv, 50))
                rng = float(np.percentile(allv, 90)) - base
            out[int(round(r))] = (base, rng, mean)
        for g in tiles:
            del g
        cp.get_default_memory_pool().free_all_blocks()
        if not out:
            return None
        logger.info("npr_edges global gradient threshold computed for radii %s",
                    sorted(out.keys()))
        return out
    except Exception as exc:
        logger.warning("Failed to compute npr grad stats: %s", exc)
        return None


# Normalized algorithms whose display range (-1..1 / 0..1) is derived from a
# robust p99 of the algorithm's own output.  The stat is computed at FULL
# RESOLUTION (so scale-sensitive algorithms keep correct magnitudes) on several
# tiles stratified across the whole valid extent (so heterogeneous scenes -- e.g.
# Mt Fuji's smooth flanks + singular summit -- are represented and singular
# outliers fall into the unclipped >1 tail rather than dominating the scale).
# Each entry: (module, raw_block_function, stat_function).  The raw block runs
# with the SAME parameters as the main pass (resolved via the registry defaults +
# the run's params, filtered to the block's signature).
_NORM_STAT_SPECS = {
    "rvi": ("_impl_rvi", "compute_rvi_efficient_block", "rvi_stat_func"),
    "fractal_anomaly": ("_impl_fractal_anomaly",
                        "compute_fractal_dimension_block", "fractal_stat_func"),
    "scale_space_surprise": ("_impl_experimental",
                            "compute_scale_space_surprise_block",
                            "scale_space_surprise_stat_func"),
    "visual_saliency": ("_impl_visual_saliency",
                        "compute_visual_saliency_block", "visual_saliency_stat_func"),
    "multiscale_terrain": ("_impl_multiscale_terrain",
                          "compute_multiscale_combined_raw", "multiscale_stat_func"),
    # Bounded [0,1] maps concentrated in a narrow high band: data-driven
    # [p1, p99] -> [0, 1] contrast stretch so the integer codes are not wasted.
    "ambient_occlusion": ("_impl_ambient_occlusion",
                         "compute_ambient_occlusion_block",
                         "robust_unsigned_stretch_stat_func"),
    "openness": ("_impl_openness", "compute_openness_vectorized",
                "robust_unsigned_stretch_stat_func"),
}


def _norm_stat_max_scale(merged: dict) -> float:
    """Largest pixel-scale among the algorithm's radii/scales/kernel parameters."""
    vals = []
    for key in ("radii", "scales"):
        v = merged.get(key)
        if isinstance(v, (list, tuple)) and v:
            vals.append(max(float(x) for x in v))
    ks = merged.get("kernel_size")
    if ks:
        vals.append(float(ks))
    return max(vals) if vals else 16.0


def _compute_norm_stats_tiled(
    src_cog: str,
    algorithm: str,
    params: dict,
    *,
    grid: int = 3,
    max_tile: int = 4096,
    min_valid_frac: float = 0.02,
) -> Optional[tuple]:
    """Robust full-resolution, full-extent normalization stats via stratified tiles.

    Reads several full-resolution windows tiled across the valid-data bounding box
    (located from a coarse overview, so off-center footprints are handled), runs
    the algorithm's raw block function on each with the main-pass parameters, pools
    the interior valid pixels, and returns the algorithm's robust ``(offset,
    p99-scale)`` statistics over the pool.  Full resolution keeps scale-sensitive
    magnitudes correct; pooling the whole extent dilutes singular outliers into the
    p99 tail.  Returns ``None`` on failure (caller falls back to window sampling).
    """
    spec = _NORM_STAT_SPECS.get(algorithm)
    if spec is None:
        return None
    try:
        import inspect
        from rasterio.windows import Window
        from rasterio.enums import Resampling
        from ..algorithms.dask_registry import ALGORITHMS
        mod = __import__(f"FujiShaderGPU.algorithms.{spec[0]}", fromlist=[spec[1]])
        block_func = getattr(mod, spec[1])
        stat_func = getattr(mod, spec[2])
        try:
            defaults = ALGORITHMS[algorithm].get_default_params() or {}
        except Exception:
            defaults = {}
        merged = {**defaults, **(params or {})}
    except Exception as exc:
        logger.warning("Tiled norm-stats helpers unavailable for %s: %s", algorithm, exc)
        return None

    try:
        accepted = set(inspect.signature(block_func).parameters)
        # Drop None-valued params so the block function falls back to its own
        # defaults (e.g. fractal_anomaly's radii default None -> would not iterate).
        kw = {k: merged[k] for k in list(merged) if k in accepted and merged[k] is not None}
        if "normalize" in accepted:
            kw["normalize"] = False
        max_scale = _norm_stat_max_scale(merged)
        margin = int(min(max_scale, max_tile // 4))
        tile = int(min(max_tile, max(2048, 4 * margin)))

        pooled = []
        with rasterio.open(src_cog) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            def _denodata(a):
                a = a.astype(np.float32, copy=False)
                if nodata is not None and not np.isnan(float(nodata)):
                    a = np.where(np.isclose(a, float(nodata), atol=1e-6), np.nan, a)
                return a

            # Coarse overview -> valid-data bounding box.
            cov = max(1, max(W, H) // 512)
            ov = _denodata(src.read(
                1, out_shape=(max(1, H // cov), max(1, W // cov)),
                resampling=Resampling.nearest, out_dtype=np.float32,
                masked=True).filled(np.nan))
            vmask = np.isfinite(ov)
            if not vmask.any():
                return None
            ys, xs = np.where(vmask)
            by0, by1 = int(ys.min()) * cov, min(H, (int(ys.max()) + 1) * cov)
            bx0, bx1 = int(xs.min()) * cov, min(W, (int(xs.max()) + 1) * cov)

            cell_h = max(1, (by1 - by0) // grid)
            cell_w = max(1, (bx1 - bx0) // grid)
            for gy in range(grid):
                for gx in range(grid):
                    ccy = by0 + gy * cell_h + cell_h // 2
                    ccx = bx0 + gx * cell_w + cell_w // 2
                    wy0 = int(min(max(0, ccy - tile // 2), max(0, H - tile)))
                    wx0 = int(min(max(0, ccx - tile // 2), max(0, W - tile)))
                    tw, th = min(tile, W - wx0), min(tile, H - wy0)
                    a = _denodata(src.read(
                        1, window=Window(wx0, wy0, tw, th),
                        out_dtype=np.float32, masked=True).filled(np.nan))
                    if float(np.isfinite(a).mean()) < min_valid_frac:
                        continue
                    g = cp.asarray(a)
                    raw = block_func(g, **kw)
                    m = int(min(margin, raw.shape[0] // 3, raw.shape[1] // 3))
                    if m > 0:
                        raw = raw[m:-m, m:-m]
                    vals = raw[~cp.isnan(raw)]
                    if vals.size:
                        pooled.append(cp.asnumpy(vals))
                    del g, raw, vals
                    cp.get_default_memory_pool().free_all_blocks()

        if not pooled:
            return None
        pooled_gpu = cp.asarray(np.concatenate(pooled))
        stats = stat_func(pooled_gpu)
        if not stats or not np.isfinite(float(stats[-1])) or float(stats[-1]) <= 1e-9:
            return None
        logger.info(
            "%s global stats from %d full-res tiles (tile=%d, margin=%d, %d px): %s",
            algorithm, len(pooled), tile, margin, int(pooled_gpu.size),
            tuple(round(float(s), 6) for s in stats),
        )
        return stats
    except Exception as exc:
        logger.warning("Failed tiled norm stats for %s: %s", algorithm, exc)
        return None


def _quantize_block_cp(block, *, a_coef: float, b_coef: float,
                       dn_min: int, dn_max: int, cp_dtype):
    """Linearly encode a float32 CuPy block to integer codes (NaN/NoData -> 0).

    ``DN = clip(round(a_coef*value + b_coef), dn_min, dn_max)``.  Mapping params
    (signed/unsigned, centring) come from ``output_encoding.quantize_params``.
    """
    dn = cp.rint(cp.float32(a_coef) * block + cp.float32(b_coef))
    dn = cp.clip(dn, cp.float32(dn_min), cp.float32(dn_max))
    dn = cp.where(cp.isnan(block), cp.float32(0.0), dn)
    return dn.astype(cp_dtype)


def _estimate_output_range(result_gpu: da.Array,
                           *, lo_pct: float = 1.0, hi_pct: float = 99.0,
                           max_samples: int = 4_000_000):
    """Robust [p1, p99] range from a strided sample (unbounded-output fallback)."""
    try:
        n = int(result_gpu.size)
        step = max(1, int((n / float(max_samples)) ** 0.5))
        sample = result_gpu[::step, ::step].compute()
        valid = sample[cp.isfinite(sample)]
        if valid.size == 0:
            return (0.0, 1.0)
        lo = float(cp.percentile(valid, lo_pct))
        hi = float(cp.percentile(valid, hi_pct))
        if not (hi > lo):
            hi = lo + 1.0
        return (lo, hi)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Output range estimation failed (%s); using [0, 1]", exc)
        return (0.0, 1.0)


def _apply_scale_offset(path: Path, scale: float, offset: float) -> None:
    """Best-effort: record GDAL band scale/offset so DN -> physical is recoverable."""
    try:
        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            return
        try:
            band = ds.GetRasterBand(1)
            band.SetScale(float(scale))
            band.SetOffset(float(offset))
        finally:
            ds = None
    except Exception as exc:  # pragma: no cover - metadata is non-critical
        logger.warning("Could not write scale/offset metadata: %s", exc)


def run_pipeline(
    src_cog: str,
    dst_cog: str,
    algorithm: str = "rvi",
    radii: Optional[List[int]] = None,
    agg: str = "mean",
    chunk: Optional[int] = None,
    show_progress: bool = True,
    auto_radii: bool = True,
    **algo_params
):
    """Improved main pipeline."""
    # Check the algorithm
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    
    algo = ALGORITHMS[algorithm]
    
    # Validate input
    validate_inputs(src_cog)

    # GDAL read optimization (set before cluster creation so spawned workers inherit it)
    _configure_gdal_read_performance()
    _log_overview_availability(src_cog)

    # Check memory status (container-aware: reports the cgroup cap, not host RAM)
    logger.info(f"System memory: {container_memory_total_gb():.1f} GB total, "
                f"{container_memory_available_gb():.1f} GB available")

    # Get the memory fraction (from algo_params, else default)
    memory_fraction = algo_params.pop('memory_fraction', None)
    # visual_saliency / fractal_anomaly build many per-scale fields plus heavy
    # combine intermediates; on large rasters their peak device memory exceeds the
    # conservative default and fragments the RMM pool.  The GPU is single-process
    # here, so give these two a larger share of it (the rest keep the default).
    if algorithm in ("visual_saliency", "fractal_anomaly"):
        _mf = 0.5 if memory_fraction is None else float(memory_fraction)
        if _mf < 0.78:
            memory_fraction = 0.78
            logger.info(
                "Combine-heavy algorithm '%s': raising GPU memory fraction %.2f -> 0.78",
                algorithm, _mf,
            )
    # NoData override: replace the given value with float NaN after load (not passed to the algorithm)
    nodata_override = algo_params.pop('nodata_override', None)
    # Output encoding: float32 (default) / int16 / uint8. Not passed to the algorithm.
    output_dtype = str(algo_params.pop('output_dtype', 'float32') or 'float32').lower()
    output_range = algo_params.pop('output_range', None)
    if output_dtype not in SUPPORTED_OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output_dtype={output_dtype!r}. "
            f"Choose from {SUPPORTED_OUTPUT_DTYPES}."
        )
    cluster, client = make_cluster(memory_fraction)  # direct call into dask_cluster.py
    
    try:
        # Automatic chunk-size determination
        if chunk is None:
            if is_zarr_path(src_cog):
                dem_probe = load_input_dataarray(src_cog, 1024)
                height = dem_probe.sizes[dem_probe.dims[-2]]
                width = dem_probe.sizes[dem_probe.dims[-1]]
                total_pixels = height * width
                total_gb = (total_pixels * 4) / (1024**3)
            else:
                with rasterio.open(src_cog) as src:
                    total_pixels = src.width * src.height
                    total_gb = (total_pixels * 4) / (1024**3)
                
            # VRAM-aware dynamic chunk sizing
            try:
                _meminfo = cp.cuda.runtime.memGetInfo()
                _vram_gb = _meminfo[1] / (1024**3)
            except Exception:
                _vram_gb = 16.0
            chunk = compute_dask_chunk(
                _vram_gb, data_gb=total_gb, algorithm=algorithm,
            )

            # ALGORITHM_COMPLEXITY (which drives the chunk size) is calibrated for
            # the LOCAL single-pass.  A multi-radius spatial run computes one
            # response per radius and holds them together for the weighted combine,
            # so the peak per-block VRAM is several times higher -- heavy blocks
            # (hillshade surface normals, atmospheric trig, etc.) then overflow the
            # RMM pool.  Shrink the chunk by ~1/sqrt(n_radii) so the per-block
            # footprint stays inside the pool regardless of the per-algorithm
            # complexity estimate.
            _n_radii = len(radii) if radii else 0
            if _n_radii > 1:
                _shrink = (2.0 / (1.0 + min(_n_radii, 6))) ** 0.5
                _shrunk = max(2048, (int(chunk * _shrink) // 256) * 256)
                if _shrunk < chunk:
                    logger.info(
                        "Multi-radius run (%d radii): shrinking chunk %d -> %d "
                        "to keep per-block VRAM within the RMM pool.",
                        _n_radii, chunk, _shrunk,
                    )
                    chunk = _shrunk

            logger.info(f"Dataset size: {total_gb:.1f} GB, using chunk size: {chunk}x{chunk}")
        
        # 6-1) Lazy-load the DEM (COG or Zarr)
        dem: xr.DataArray = load_input_dataarray(src_cog, chunk)

        # Manual NoData override: replace matching cells with NaN (lazy, chunk-preserving).
        if nodata_override is not None and np.isfinite(nodata_override):
            dem = dem.where(dem != np.float32(nodata_override))
            logger.info("Applied --nodata override: %s -> NaN", float(nodata_override))

        logger.info(f"DEM shape: {dem.shape}, dtype: {dem.dtype}, "
                   f"chunks: {dem.chunks}")

        # Get algorithm-specific default parameters
        default_params = algo.get_default_params()

        # Prepare parameters
        params = {
            **default_params,
            **algo_params,
            'show_progress': show_progress,
            'agg': agg
        }

        # NoData void filling is owned by the preprocessing command
        # (`python -m FujiShaderGPU.prepare`); the pipeline no longer fills holes.

        # 6-2) Convert to a CuPy array (improved: explicit metadata)
        gpu_arr: da.Array = dem.data.map_blocks(
            cp.asarray,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )

        # Inject anisotropic pixel scales (simple geographic DEM support).
        px_m_x, px_m_y, pixel_size_m, is_geo, lat_center = _detect_metric_scales_from_dataarray(dem)
        params['pixel_size'] = float(pixel_size_m)
        params.setdefault('pixel_scale_x', float(px_m_x))
        params.setdefault('pixel_scale_y', float(px_m_y))
        # Tile path parity: set is_geographic_dem and elevation_scale.
        # Note: in the Dask path pixel_scale_x/y are raw meter values so
        #   elevation_scale is NOT applied to the DEM array.  The field is
        #   provided only for algorithm-level flag consistency with the tile
        #   pipeline (where DEM is pre-scaled by elevation_scale).
        params.setdefault('is_geographic_dem', bool(is_geo))
        params.setdefault('elevation_scale',
                          float(1.0 / max(pixel_size_m, 1e-6)) if is_geo else 1.0)
        if is_geo:
            ratio = abs(px_m_y) / max(abs(px_m_x), 1e-9)
            logger.info(
                "Geographic DEM approximation enabled: "
                f"lat={lat_center:.3f}, dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m, dy/dx={ratio:.4f}"
            )
        else:
            logger.info(f"Projected pixel scales: dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m")
        
        # 6-2.5) Auto-determination (for the RVI algorithm)
        if algorithm == "rvi":
            pixel_size = float(params.get('pixel_size', 1.0))
            
            # Make the new efficient mode the default
            if radii is None and auto_radii:
                logger.info("Analyzing terrain for automatic radii determination...")
                
                # Terrain analysis
                terrain_stats = analyze_terrain_characteristics(
                    gpu_arr, sample_ratio=0.01, src_path=src_cog)
                terrain_stats['pixel_size'] = pixel_size
                
                # Determine optimal radii
                radii, weights = determine_optimal_radii(terrain_stats)
                
                logger.info("Terrain analysis results:")
                logger.info(f"  - Elevation range: {terrain_stats['elevation_range']:.1f} m")
                logger.info(f"  - Mean slope: {terrain_stats['mean_slope']:.3f}")
                logger.info(f"  - Auto-determined radii: {radii} pixels")
                logger.info(f"  - Weights: {[f'{w:.2f}' for w in weights]}")
                
                # Store into parameters
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = weights
                
            elif radii is not None:
                # Manually specified radius mode
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = algo_params.get('weights', None)
            
            else:
                raise ValueError("Either provide radii or enable auto_radii")

        # Many algorithms need the pixel size (for non-RVI cases)
        elif algorithm != "rvi" and ('pixel_size' not in params or params['pixel_size'] == 1.0):
            # Already injected above; keep this branch as a no-op for compatibility.
            params['pixel_size'] = float(params.get('pixel_size', 1.0))

        # CLI passes explicit radii through run_pipeline's top-level radii
        # parameter, so restore it for non-RVI spatial algorithms.
        if algorithm != "rvi" and radii is not None:
            params['radii'] = radii
            logger.info(f"Setting radii for {algorithm}: {radii}")

        # Normalized algorithms (RVI/fractal_anomaly/scale_space_surprise/
        # visual_saliency/multiscale_terrain): derive the display range
        # (-1..1 / 0..1) from a robust p99 of the algorithm's own FULL-RESOLUTION
        # output, pooled over tiles stratified across the whole valid extent.
        # Full resolution keeps scale-sensitive magnitudes correct (radii/kernel
        # are not decimated); pooling the whole extent is robust to off-center
        # data and dilutes singular outliers (e.g. a volcano summit) into the
        # unclipped >1 tail rather than letting them set the scale.
        if (
            algorithm in _NORM_STAT_SPECS
            and "global_stats" not in params
            and not is_zarr_path(src_cog)
        ):
            _norm_stats = _compute_norm_stats_tiled(src_cog, algorithm, params)
            if _norm_stats is not None:
                params["global_stats"] = _norm_stats

        # fractal_anomaly's relief_conf uses a per-block roughness percentile that
        # varies tile-to-tile (visible tile-boundary seams).  Inject a global
        # roughness p10/p75 so it is consistent across the whole raster.
        if (
            algorithm == "fractal_anomaly"
            and params.get("relief_p10") is None
            and params.get("relief_p75") is None
            and not is_zarr_path(src_cog)
        ):
            _relief = _compute_fractal_relief_stats(src_cog, params)
            if _relief is not None:
                params["relief_p10"], params["relief_p75"] = _relief

        # npr_edges thresholds edges against a per-block gradient distribution
        # (tile-boundary seams).  Inject a global per-radius gradient threshold so
        # the small (full-res) radii are consistent across tiles.
        if (
            algorithm == "npr_edges"
            and "_npr_grad_stats" not in params
            and not is_zarr_path(src_cog)
        ):
            _ngs = _compute_npr_grad_stats(src_cog, params)
            if _ngs:
                params["_npr_grad_stats"] = _ngs

        # fractal_anomaly / visual_saliency hybrid coarse path (RVI-style): when the
        # user gives explicit large --radii, precompute their per-scale response
        # fields (roughness / smooth) from the COG overview.  The algorithm then
        # computes small radii at full resolution and samples these large fields with
        # no per-chunk halo -- accurate large radii (no MAX_DEPTH halo truncation,
        # which both blurred large scales and, for fractal, drove tile-boundary
        # seams) and bounded VRAM on huge streaming rasters.
        _HYBRID_PFX = {
            "fractal_anomaly": "_fractal",
            "visual_saliency": "_vs",
            "scale_space_surprise": "_sss",
        }
        if (
            algorithm in _HYBRID_PFX
            and params.get("radii")
            and f"{_HYBRID_PFX[algorithm]}_large_fields" not in params
            and not is_zarr_path(src_cog)
        ):
            try:
                from ..algorithms._nan_utils import compute_overview_scale_fields
                _radii = [float(r) for r in (params.get("radii") or [])]
                _pfx = _HYBRID_PFX[algorithm]
                if algorithm == "fractal_anomaly":
                    from ..algorithms._impl_fractal_anomaly import (
                        fractal_large_scale_predicate as _pred,
                        _fractal_roughness_block as _bfn,
                    )
                    if len(_radii) < 5:
                        _radii = [4.0, 8.0, 16.0, 32.0, 64.0]
                elif algorithm == "visual_saliency":
                    from ..algorithms._impl_visual_saliency import (
                        vs_large_scale_predicate as _pred,
                        _vs_smooth_block as _bfn,
                    )
                    _radii = [max(0.5, float(s)) for s in _radii]
                    if len(_radii) < 4:
                        _radii = [2.0, 4.0, 8.0, 16.0]
                else:  # scale_space_surprise
                    from ..algorithms._impl_experimental import (
                        sss_large_scale_predicate as _pred,
                        _sss_smooth_block as _bfn,
                    )
                    _radii = sorted(s for s in _radii if s > 0)
                _large = [r for r in _radii if _pred(r)]
                if _large:
                    _fields, _decim = compute_overview_scale_fields(
                        src_cog, large_radii=_large, block_fn=_bfn)
                    if _fields:
                        params[f"{_pfx}_large_fields"] = _fields
                        params[f"{_pfx}_full_shape"] = tuple(int(s) for s in gpu_arr.shape)
                        logger.info(
                            "%s hybrid overview path: large_radii=%s, decimation=%.1fx "
                            "-> per-chunk halo bounded (small radii only)",
                            algorithm, [int(round(r)) for r in _large], _decim)
            except Exception as exc:
                logger.warning(
                    "%s hybrid overview path unavailable; using single-block path: %s",
                    algorithm, exc)

        # RVI large-radius-from-overview fast path: split radii at a chunk-aware
        # threshold; compute the large-radius contribution from the COG overview
        # (no huge per-chunk halo) and let the algorithm add it to the small-radius
        # full-resolution RVI.  Any failure falls back to the full-resolution path.
        if (
            algorithm == "rvi"
            and not is_zarr_path(src_cog)
            and "_rvi_coarse_field" not in params
        ):
            try:
                from ..algorithms._impl_rvi import (
                    rvi_default_large_radius_threshold,
                    split_radii_by_threshold,
                )
                _full_radii = list(params.get("radii") or [])
                if _full_radii:
                    _threshold = rvi_default_large_radius_threshold(int(chunk))
                    _sr, _sw, _lr, _lw = split_radii_by_threshold(
                        _full_radii, params.get("weights"), _threshold
                    )
                    if _lr:
                        _field = _compute_rvi_overview_coarse_field(
                            src_cog, large_radii=_lr, large_weights=_lw,
                        )
                        if _field is not None:
                            params["_rvi_coarse_field"] = _field
                            params["_rvi_small_radii"] = _sr
                            params["_rvi_small_weights"] = _sw
                            params["_rvi_w_large"] = float(sum(_lw))
                            params["_rvi_full_shape"] = tuple(int(s) for s in gpu_arr.shape)
                            params["_rvi_field_offset"] = (0, 0)
                            logger.info(
                                "RVI overview large-radius path: small=%s, large=%s "
                                "(threshold=%dpx) -> per-chunk halo from %d to %d px",
                                _sr, _lr, _threshold,
                                max(_full_radii) + 16,
                                (max(_sr) + 16) if _sr else 16,
                            )
            except Exception as exc:
                logger.warning(
                    "RVI overview large-radius path unavailable; using full-resolution radii: %s",
                    exc,
                )

        # 6-3) Run the algorithm (within run_pipeline)
        # Redact bulky internal arrays (e.g. the RVI coarse field) from logs/metadata.
        _log_params = {
            k: v for k, v in params.items()
            if not (k.startswith("_rvi_coarse_field")
                    or k in ("_fractal_large_fields", "_vs_large_fields",
                             "_sss_large_fields"))
        }
        logger.info(f"Computing {algorithm} with parameters: {_log_params}")

        # Apply the algorithm (lazy evaluation)
        result_gpu: da.Array = algo.process(gpu_arr, **params)

        # Re-apply the input NoData footprint: keep NaN exactly where the source
        # DEM is NoData.  Algorithms are NaN-aware internally, but multiscale /
        # large-radius paths fill voids (cliff-free) for stability; this final
        # mask guarantees the exterior stays NoData with no boundary halo.
        # (Skip when the algorithm changed the shape, e.g. agg='stack'.)
        if result_gpu.shape == gpu_arr.shape:
            result_gpu = da.where(
                da.isnan(gpu_arr), cp.float32(cp.nan), result_gpu
            )

        # Drop internal helper arrays so they never reach COG metadata (str(params)).
        for _k in (
            "_rvi_coarse_field", "_rvi_small_radii", "_rvi_small_weights",
            "_rvi_w_large", "_rvi_full_shape", "_rvi_field_offset",
            "_fractal_large_fields", "_fractal_full_shape",
            "_vs_large_fields", "_vs_full_shape",
            "_sss_large_fields", "_sss_full_shape",
        ):
            params.pop(_k, None)

        # 6-3.5) Quantize the output dtype (float32 passes through). NaN (NoData) -> 0,
        # valid values are linearly stretched from the algorithm's native range to [1, levels].
        out_np_dtype = "float32"
        out_scale_offset = None
        if output_dtype in ("int16", "uint8") and result_gpu.shape == gpu_arr.shape:
            value_range = resolve_output_range(
                algorithm, params=params, override=output_range,
            )
            if value_range is None:
                logger.info(
                    "No fixed output range for %s (unit=%s); estimating from data percentiles",
                    algorithm, params.get("unit", ""),
                )
                value_range = _estimate_output_range(result_gpu)
            lo, hi = float(value_range[0]), float(value_range[1])
            qp = quantize_params(lo, hi, output_dtype)
            cp_dtype = cp.int16 if output_dtype == "int16" else cp.uint8
            out_np_dtype = output_dtype
            # Visualization integer outputs are written as plain DN rasters
            # (uint8: 0..255, int16: raw codes) with NoData=0 and NO GDAL
            # scale/offset metadata.  Embedding scale/offset makes QGIS auto-
            # unscale the band and present it as Float32 over the physical range
            # (e.g. RVI -1.5..1.5), which looks "quantized float with NoData=0"
            # and is confusing for a display product.  The DN<->value mapping is
            # still logged and recorded in the COG 'parameters'/value_range attrs
            # for anyone who needs to recover physical units.
            out_scale_offset = None
            logger.info(
                "Output dtype=%s (%s): range [%.6g, %.6g] -> DN [%d, %d], NoData=0 "
                "(DN<->value mapping: value = %.6g*DN + %.6g; not embedded as scale/offset)",
                output_dtype, "signed" if qp["signed"] else "unsigned",
                lo, hi, qp["dn_min"], qp["dn_max"], qp["scale"], qp["offset"],
            )
            result_gpu = result_gpu.map_blocks(
                _quantize_block_cp,
                dtype=np.dtype(out_np_dtype),
                meta=cp.empty((0, 0), dtype=cp_dtype),
                a_coef=qp["a_coef"], b_coef=qp["b_coef"],
                dn_min=qp["dn_min"], dn_max=qp["dn_max"], cp_dtype=cp_dtype,
            )
        elif output_dtype in ("int16", "uint8"):
            logger.warning(
                "Output dtype=%s requested but result shape changed (e.g. agg=stack); "
                "writing float32 instead.", output_dtype,
            )

        # 6-4) GPU->CPU transfer back (improved: explicit dtype)
        result_cpu = result_gpu.map_blocks(
            cp.asnumpy,
            dtype=out_np_dtype,
            meta=np.empty((0, 0), dtype=out_np_dtype),
        )

        # Run the computation with progress display
        # ---------- Compute trigger after the GPU->CPU conversion ----------
        # When it exceeds 20 GB, skip persist and let
        # write_cog_da_chunked() stream-compute it.
        total_gb = result_cpu.nbytes / (1024**3)
        if total_gb <= 20:
            if show_progress:
                logger.info("Persisting small dataset for faster workflow")
                result_cpu = client.persist(result_cpu, optimize_graph=True)
                progress(result_cpu, interval='1s')
            else:
                result_cpu = result_cpu.persist()
        else:
            logger.info(f"Large dataset ({total_gb:.1f} GB) - skip persist; "
                        "chunked writer will stream-compute each tile")
    
        # 6-5) xarray wrap (improved: simplified coordinate construction)
        # 6-5) xarray wrap (coordinate construction)
        dims = dem.dims
        coords = dem.coords
        
        # Set the appropriate value range per algorithm
        if algorithm in ['hillshade']:
            # Hillshade is now also float32 (0..1), aligned with tile backend.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1",
                "data_type": "float32"
            }
        elif algorithm in ['slope']:
            # Slope depends on the unit (degree, percent, radian)
            unit = params.get('unit', 'degree')
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "unit": unit,
                "data_type": "float32"
            }
        elif algorithm in ['rvi', 'fractal_anomaly']:
            # Signed terrain anomaly outputs map p80(abs(value)) to +/-1,
            # with overflow preserved for strong extrema.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "-1.5 to +1.5",
                "normal_range": "-1 to +1",
                "normal_percentile": "80",
                "data_type": "float32"
            }
        elif algorithm in [
            'atmospheric_scattering',
            'multiscale_terrain',
            'curvature',
            'visual_saliency',
            'ambient_occlusion',
            'openness',
            'scale_space_surprise',
            'multi_light_uncertainty',
        ]:
            # Unsigned analysis outputs map p80(value) to +1,
            # with overflow preserved for strong extrema.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1.5",
                "normal_range": "0 to 1",
                "normal_percentile": "80",
                "data_type": "float32"
            }
        else:
            # Display/stylized outputs keep their native display range.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "0 to 1",
                "data_type": "float32"
            }
            
        # Reflect the actual on-disk dtype (the per-algorithm blocks above hard-code
        # data_type=float32, which is misleading for quantized int16/uint8 output and
        # shows up verbatim in QGIS metadata). value_range stays as the physical
        # interpretation / DN<->value documentation.
        attrs["data_type"] = out_np_dtype

        result_da = xr.DataArray(
            result_cpu,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=algorithm.upper(),
        )
        
        # Carry over CRS info from the source DEM (for COG input)
        if hasattr(dem, 'rio') and dem.rio.crs is not None:
            result_da.rio.write_crs(dem.rio.crs, inplace=True)
        
        # 6-6) Output (COG or Zarr)
        dst_path = Path(dst_cog)
        if is_zarr_path(str(dst_path)):
            logger.info("Writing output as Zarr: %s", dst_path)
            write_zarr_output(result_da, dst_path, show_progress=show_progress)
        else:
            write_cog_da_chunked(result_da, dst_path, show_progress=show_progress)
            # Record DN->physical recovery (integer outputs only); non-critical.
            if out_scale_offset is not None:
                _apply_scale_offset(dst_path, out_scale_offset[0], out_scale_offset[1])
        logger.info("Pipeline completed successfully!")
        gc.collect()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Explicitly clear the CuPy memory pool
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        # More thorough cleanup
        try:
            # Force-clear worker memory
            def clear_worker_memory():
                import gc
                import cupy as cp
                # Raise worker/nanny log levels before teardown so the benign
                # shutdown-race "Failed to communicate with scheduler during
                # heartbeat -> CommClosedError: Stream is closed" traceback is
                # not emitted at ERROR after the pipeline already succeeded.
                import logging as _logging
                for _name in ("distributed.worker", "distributed.nanny",
                              "distributed.core", "distributed.comm"):
                    _logging.getLogger(_name).setLevel(_logging.CRITICAL)
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                gc.collect()
                return True
            client.run(clear_worker_memory)
            # Close the client first and wait until it fully shuts down
            client.close(timeout=10)
            client.shutdown()  # ensure all workers and the scheduler terminate
        except Exception as e:
            logger.debug(f"Client shutdown warning (can be ignored): {e}")
        
        try:
            # Close the cluster (ignore exceptions since it may already be closed)
            cluster.close(timeout=10)
        except Exception as e:
            logger.debug(f"Cluster close warning (can be ignored): {e}")
        
        # Wait for the Dask worker processes to terminate for sure
        time.sleep(3)
        
        # Final GC
        gc.collect()

