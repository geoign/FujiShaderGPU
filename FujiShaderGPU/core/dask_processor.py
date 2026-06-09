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

from ..algorithms.common.spatial_mode import RADII_DRIVEN_ALGOS, MULTISCALE_REQUIRED_ALGOS
from ..algorithms._norm_stats import inject_global_stats
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

    # Shared, container-aware GDAL I/O tuning (single source of truth with the
    # tile backend). force=False -> respect any user-set GDAL env on the read path.
    from ..config.gdal_config import apply_gdal_io_config
    apply_gdal_io_config(cache_mb, force=False)
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
        # ALIGNED_LEVELS keeps overview block grids aligned with full-res tiles
        # (matches the tile backend; better COG locality for partial reads).
        "ALIGNED_LEVELS": "4",
        "BIGTIFF": "YES",
        # Container/cgroup-aware: GDAL's "ALL_CPUS" resolves to the host core
        # count and ignores the CFS quota, oversubscribing throttled containers
        # (matches the tile backend's _gdal_num_threads()).
        "NUM_THREADS": str(container_cpu_count()),
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

            src_crs = data.rio.crs if hasattr(data, 'rio') else None

            # ---- Direct single-file windowed write (no per-chunk files / VRT) ----
            # Each computed chunk is written straight into ONE pre-created tiled
            # BigTIFF via a windowed WriteArray, with cluster compute overlapped
            # against the disk write through a dedicated writer thread.  This drops
            # the old per-chunk-TIFF + VRT + consolidation-Translate stage (a full
            # extra read+write of the whole raster) and lets the GPU keep computing
            # while finished tiles are compressed/written -- a large win on huge
            # rasters where that staging dominated wall-clock and disk I/O.
            from osgeo import gdal_array
            import queue as _queue
            import threading as _threading

            H = int(sum(data.chunks[0]))
            W = int(sum(data.chunks[1]))
            gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(data.dtype).type)
            nodata_val = float(output_nodata_for_dtype(data.dtype))
            row_off = [int(sum(data.chunks[0][:i])) for i in range(n_rows)]
            col_off = [int(sum(data.chunks[1][:j])) for j in range(n_cols)]

            merged_tif = Path(tmpdir) / "merged_tmp.tif"
            create_opts = [
                "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
                "COMPRESS=ZSTD", f"ZLEVEL={cog_options.get('LEVEL', '1')}",
                "BIGTIFF=YES", f"NUM_THREADS={container_cpu_count()}",
            ]
            if 'PREDICTOR' in cog_options:
                create_opts.append(f"PREDICTOR={cog_options['PREDICTOR']}")
            out_ds = gdal.GetDriverByName("GTiff").Create(
                str(merged_tif), W, H, 1, gdal_dtype, options=create_opts,
            )
            if out_ds is None:
                raise ValueError("Failed to create intermediate GeoTIFF for chunked write")
            try:
                out_ds.SetGeoTransform(data.rio.transform().to_gdal())
            except Exception as exc:
                logger.warning("Could not set geotransform on chunked output: %s", exc)
            try:
                if src_crs is not None:
                    out_ds.SetProjection(src_crs.to_wkt())
            except Exception as exc:
                logger.warning("Could not set projection on chunked output: %s", exc)
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(nodata_val)

            # Single writer thread (GDAL dataset writes must be serialized); the
            # bounded queue back-pressures compute so host RAM stays low.
            write_q = _queue.Queue(maxsize=8)
            write_err = {}

            def _writer():
                while True:
                    item = write_q.get()
                    try:
                        if item is None:
                            break
                        arr, xoff, yoff = item
                        out_band.WriteArray(np.ascontiguousarray(arr), xoff, yoff)
                    except Exception as exc:  # surface in the main thread
                        write_err.setdefault("e", exc)
                    finally:
                        write_q.task_done()

            wt = _threading.Thread(target=_writer, name="cog-writer", daemon=True)
            wt.start()

            client = None
            try:
                client = get_client()
            except Exception:
                client = None

            try:
                if client is not None:
                    try:
                        n_workers = max(1, len(client.scheduler_info().get("workers", {})))
                    except Exception:
                        n_workers = 1
                    # A few extra in-flight chunks keep the GPU worker(s) and the
                    # writer thread busy at the same time (compute || write overlap).
                    max_inflight = max(4, n_workers * 3)
                    from distributed import as_completed as _as_completed
                    coords_flat = [(i, j) for i in range(n_rows) for j in range(n_cols)]
                    task_iter = iter(coords_flat)
                    fut_meta = {}
                    inflight = _as_completed()

                    def _submit_next() -> bool:
                        try:
                            i, j = next(task_iter)
                        except StopIteration:
                            return False
                        fut = client.compute(delayed_chunks[i, j])
                        fut_meta[fut] = (i, j)
                        inflight.add(fut)
                        return True

                    logger.info(
                        "Direct chunk write: %d chunks -> single BigTIFF, %d worker(s), up to %d in flight",
                        total_chunks, n_workers, max_inflight,
                    )
                    done = 0
                    with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                        for _ in range(min(max_inflight, total_chunks)):
                            if not _submit_next():
                                break
                        for fut in inflight:
                            i, j = fut_meta.pop(fut)
                            try:
                                arr = fut.result()
                            finally:
                                del fut
                            if write_err:
                                raise write_err["e"]
                            write_q.put((arr, col_off[j], row_off[i]))
                            del arr
                            done += 1
                            if done % 10 == 0:
                                try:
                                    cp.get_default_memory_pool().free_all_blocks()
                                except Exception:
                                    pass
                            pbar.update(1)
                            _submit_next()
                else:
                    # Serial fallback (no distributed client)
                    with tqdm(total=total_chunks, desc="Writing chunks", unit="chunk") as pbar:
                        for i in range(n_rows):
                            for j in range(n_cols):
                                arr = delayed_chunks[i, j].compute()
                                if write_err:
                                    raise write_err["e"]
                                write_q.put((arr, col_off[j], row_off[i]))
                                del arr
                                pbar.update(1)
                # Drain and stop the writer.
                write_q.put(None)
                write_q.join()
                wt.join(timeout=120)
                if write_err:
                    raise write_err["e"]
                out_band.FlushCache()
                out_ds.FlushCache()
            finally:
                out_band = None
                out_ds = None

            # merged_tmp.tif now holds the full result in one tiled BigTIFF. Build
            # the COG (overviews + COG layout) in a single CreateCopy via the COG
            # driver; fall back to the gdaladdo+translate path on older GDAL.
            logger.info("Building COG (overviews + layout) from single intermediate: %s", dst)
            if not _build_cog_via_cog_driver(merged_tif, dst, cog_options):
                build_cog_with_overviews(merged_tif, dst, cog_options)
            try:
                merged_tif.unlink()
            except OSError:
                pass
            logger.info("Successfully created COG (%d chunks, direct single-file write)", total_chunks)

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

def _build_cog_via_cog_driver(src: Path, dst: Path, cog_options: dict) -> bool:
    """Build the final COG (overviews + COG layout) in ONE CreateCopy via GDAL's
    COG driver.  Returns True on success, False if the driver is unavailable or
    fails (the caller then falls back to ``build_cog_with_overviews``).  This
    avoids the extra in-place ``gdaladdo`` + full-file re-``Translate`` of the
    fallback path -- one fewer full read+write of the whole raster."""
    try:
        if gdal.GetDriverByName("COG") is None:
            return False
    except Exception:
        return False
    num_cpus = container_cpu_count()
    opts = [
        f"COMPRESS={cog_options.get('COMPRESS', 'ZSTD')}",
        f"LEVEL={cog_options.get('LEVEL', '1')}",
        f"OVERVIEW_COMPRESS={cog_options.get('OVERVIEW_COMPRESS', 'ZSTD')}",
        f"BLOCKSIZE={cog_options.get('BLOCKSIZE', '512')}",
        f"OVERVIEW_RESAMPLING={cog_options.get('OVERVIEW_RESAMPLING', 'AVERAGE')}",
        f"OVERVIEW_COUNT={cog_options.get('OVERVIEW_COUNT', '8')}",
        f"ALIGNED_LEVELS={cog_options.get('ALIGNED_LEVELS', '4')}",
        "BIGTIFF=YES",
        f"NUM_THREADS={num_cpus}",
    ]
    if 'PREDICTOR' in cog_options:
        opts.append(f"PREDICTOR={cog_options['PREDICTOR']}")
    try:
        pbar = tqdm(total=100, desc="Building COG", unit="%")

        def _cb(complete, _msg, _data):
            pbar.n = int(complete * 100)
            pbar.refresh()
            if complete >= 1.0:
                pbar.close()
            return 1

        res = gdal.Translate(
            str(dst), str(src), format="COG", creationOptions=opts, callback=_cb,
        )
        if res is None:
            return False
        res = None
        return True
    except Exception as exc:
        logger.warning("COG driver build failed (%s); falling back to gdaladdo+translate", exc)
        return False


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


def _compute_topousm_fast_overview_coarse_field(
    src_cog: str,
    *,
    large_radii: List[int],
    large_weights: List[float],
    sample_max: int = 2048,
):
    """Compute the large-radius TopoUSM Fast coarse field (Sum w*mean) from the COG overview.

    Reading the stored overview (decimated read) avoids materialising the huge
    full-resolution halo a large radius would otherwise require.  The returned
    CuPy field is sampled per-block as ``W_large*block - upsample(field)``.
    Returns ``None`` on any failure so the caller transparently falls back to the
    full-resolution radii path.
    """
    if not large_radii:
        return None
    try:
        from ..algorithms._impl_topousm_fast import compute_topousm_fast_large_coarse_field
        from rasterio.enums import Resampling
    except Exception as exc:
        logger.warning("TopoUSM Fast overview coarse-field helpers unavailable: %s", exc)
        return None
    try:
        with rasterio.open(src_cog) as src:
            scale = max(src.width / sample_max, src.height / sample_max, 1.0)
            # Both axes from the SAME scale (no per-axis floor) so the actual
            # decimation stays isotropic on elongated rasters.
            sample_w = max(1, int(round(src.width / scale)))
            sample_h = max(1, int(round(src.height / scale)))
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
        field = compute_topousm_fast_large_coarse_field(
            coarse_dem, large_radii=large_radii, large_weights=large_weights,
            decimation=float(scale),
        )
        logger.info(
            "TopoUSM Fast large-radius overview field: decimation=%.1fx, large_radii=%s",
            scale, list(large_radii),
        )
        return field
    except Exception as exc:
        logger.warning("Failed to compute TopoUSM Fast overview coarse field: %s", exc)
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
                           max_window: int = 4096):
    """Robust [p1, p99] range from a bounded central window (unbounded-output fallback).

    A strided sample (``arr[::step, ::step]``) looks cheap but forces every
    chunk of the whole result graph to compute once just for this estimate --
    effectively doubling the run.  A contiguous central window only computes the
    chunks it overlaps."""
    try:
        h, w = (int(s) for s in result_gpu.shape[:2])
        win = max(256, min(h, w, int(max_window)))
        y0 = max(0, (h - win) // 2)
        x0 = max(0, (w - win) // 2)
        sample = result_gpu[y0:y0 + win, x0:x0 + win].compute()
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


def run_pipeline(
    src_cog: str,
    dst_cog: str,
    algorithm: str = "topousm_fast",
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
        if nodata_override is not None:
            if np.isnan(nodata_override):
                # NaN cells already propagate as NoData through every NaN-aware
                # kernel; nothing to replace.
                logger.info("--nodata nan: NaN is already treated as NoData; no replacement needed")
            elif np.isfinite(nodata_override):
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
        # pixel_scale_x/y are REAL signed meters per pixel; the DEM array is
        # never rescaled -- the same convention as the tile backend, so the
        # shared (raw-elevation) normalization stats apply to both.
        px_m_x, px_m_y, pixel_size_m, is_geo, lat_center = _detect_metric_scales_from_dataarray(dem)
        params['pixel_size'] = float(pixel_size_m)
        params.setdefault('pixel_scale_x', float(px_m_x))
        params.setdefault('pixel_scale_y', float(px_m_y))
        params.setdefault('is_geographic_dem', bool(is_geo))
        if is_geo:
            ratio = abs(px_m_y) / max(abs(px_m_x), 1e-9)
            logger.info(
                "Geographic DEM approximation enabled: "
                f"lat={lat_center:.3f}, dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m, dy/dx={ratio:.4f}"
            )
        else:
            logger.info(f"Projected pixel scales: dx={abs(px_m_x):.3f}m, dy={abs(px_m_y):.3f}m")
        
        # 6-2.5) Resolve radii/weights from the full Dask-array shape and inject
        # into params (see process_dem_tiles for the shared rule):
        #   * --mode local  -> radii=[1], weights=[1.0]; explicit radii ignored.
        #   * --mode spatial -> geometric radii truncated by the DEM short side.
        #   * fractal_anomaly / scale_space_surprise / visual_saliency need >=2
        #     scales, so --mode local falls back to the spatial default + warning.
        from ..algorithms.common.spatial_mode import (
            auto_spatial_profile, LOCAL_RADII, LOCAL_WEIGHTS,
        )
        _dem_short_side = min(int(gpu_arr.shape[-2]), int(gpu_arr.shape[-1]))
        _user_mode = str(params.get("mode", "spatial")).lower()
        _is_local = _user_mode == "local"
        if _is_local and algorithm in MULTISCALE_REQUIRED_ALGOS:
            logger.warning(
                "%s requires multiple scales; --mode local is not supported -- "
                "using the spatial default instead.", algorithm,
            )
            params['mode'] = 'spatial'
            _user_mode = 'spatial'
            _is_local = False

        if algorithm == "topousm_fast":
            if _is_local:
                if radii is not None:
                    logger.warning(
                        "--mode local ignores explicit radii; forcing radii=%s.", LOCAL_RADII)
                radii, weights = list(LOCAL_RADII), list(LOCAL_WEIGHTS)
                logger.info("Local mode: radii=%s, weights=%s", radii, weights)
                params['mode'] = 'radius'
                params['radii'] = radii
                params['weights'] = weights

            elif radii is None and auto_radii:
                radii, weights = auto_spatial_profile(_dem_short_side)
                logger.info(
                    "Auto spatial radii (short_side=%d px): radii=%s, weights=%s",
                    _dem_short_side, radii, [round(w, 3) for w in weights],
                )
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

        else:
            # Many algorithms need the pixel size (for non-TopoUSM Fast cases)
            if 'pixel_size' not in params or params['pixel_size'] == 1.0:
                params['pixel_size'] = float(params.get('pixel_size', 1.0))

            if _is_local:
                if radii is not None or params.get('scales'):
                    logger.warning(
                        "--mode local ignores explicit radii/scales; forcing radii=%s.", LOCAL_RADII)
                params['radii'] = list(LOCAL_RADII)
                params['weights'] = list(LOCAL_WEIGHTS)
                logger.info("Local mode: radii=%s, weights=%s", LOCAL_RADII, LOCAL_WEIGHTS)
            elif radii is not None:
                # CLI passes explicit radii through run_pipeline's top-level radii.
                params['radii'] = radii
                logger.info(f"Setting radii for {algorithm}: {radii}")
            elif _user_mode == "spatial" and algorithm in RADII_DRIVEN_ALGOS:
                _auto_r, _auto_w = auto_spatial_profile(_dem_short_side)
                params['radii'] = _auto_r
                if params.get('weights') is None:
                    params['weights'] = _auto_w
                logger.info(
                    "Auto spatial radii (%s, short_side=%d px): radii=%s",
                    algorithm, _dem_short_side, _auto_r,
                )

        # Compute + inject every per-algorithm global normalization statistic
        # (fractal relief -> robust display range -> npr gradient -> specular
        # roughness, in that order) via the shared backend-neutral helper, so the
        # dask and tile pipelines cannot drift on these.  All steps are full-res,
        # global (seam-free) and mode-independent.
        inject_global_stats(src_cog, algorithm, params, is_zarr=is_zarr_path(src_cog))

        # Unified coarse source: read ONE decimated overview of the DEM and share it
        # across every algorithm's large-radius path (the coarse-overview combine is
        # the seam-free, memory-bounded TopoUSM Fast method).  All multiscale algorithms then
        # derive their large radii from this single cheap read instead of a
        # full-resolution da.coarsen pass each.  Injected for any spatial run with
        # explicit radii; algorithms that do not use the coarse path ignore it.
        _overview_dem = None
        _overview_decim = None
        if (
            params.get("radii")
            and str(params.get("mode", "")).lower() == "spatial"
            and "_overview_coarse_dem" not in params
            and not is_zarr_path(src_cog)
        ):
            try:
                from ..algorithms._nan_utils import read_overview_coarse_dem
                _overview_dem, _overview_decim = read_overview_coarse_dem(src_cog)
                if _overview_dem is not None:
                    params["_overview_coarse_dem"] = _overview_dem
                    params["_overview_decimation"] = _overview_decim
                    logger.info(
                        "Unified overview coarse source: %dx%d (decimation=%.1fx)",
                        int(_overview_dem.shape[1]), int(_overview_dem.shape[0]),
                        float(_overview_decim))
            except Exception as exc:
                logger.warning("Unified overview coarse source unavailable: %s", exc)

        # fractal_anomaly / visual_saliency hybrid coarse path (TopoUSM Fast-style): when the
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
                        src_cog, large_radii=_large, block_fn=_bfn,
                        coarse_dem=_overview_dem, decimation=_overview_decim)
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

        # TopoUSM Fast large-radius-from-overview fast path: split radii at a chunk-aware
        # threshold; compute the large-radius contribution from the COG overview
        # (no huge per-chunk halo) and let the algorithm add it to the small-radius
        # full-resolution TopoUSM Fast.  Any failure falls back to the full-resolution path.
        if (
            algorithm == "topousm_fast"
            and not is_zarr_path(src_cog)
            and "_topousm_fast_coarse_field" not in params
        ):
            try:
                from ..algorithms._impl_topousm_fast import (
                    topousm_fast_default_large_radius_threshold,
                    split_radii_by_threshold,
                )
                _full_radii = list(params.get("radii") or [])
                if _full_radii:
                    _threshold = topousm_fast_default_large_radius_threshold(int(chunk))
                    _sr, _sw, _lr, _lw = split_radii_by_threshold(
                        _full_radii, params.get("weights"), _threshold
                    )
                    if _lr:
                        _field = _compute_topousm_fast_overview_coarse_field(
                            src_cog, large_radii=_lr, large_weights=_lw,
                        )
                        if _field is not None:
                            params["_topousm_fast_coarse_field"] = _field
                            params["_topousm_fast_small_radii"] = _sr
                            params["_topousm_fast_small_weights"] = _sw
                            params["_topousm_fast_w_large"] = float(sum(_lw))
                            params["_topousm_fast_full_shape"] = tuple(int(s) for s in gpu_arr.shape)
                            params["_topousm_fast_field_offset"] = (0, 0)
                            logger.info(
                                "TopoUSM Fast overview large-radius path: small=%s, large=%s "
                                "(threshold=%dpx) -> per-chunk halo from %d to %d px",
                                _sr, _lr, _threshold,
                                max(_full_radii) + 16,
                                (max(_sr) + 16) if _sr else 16,
                            )
            except Exception as exc:
                logger.warning(
                    "TopoUSM Fast overview large-radius path unavailable; using full-resolution radii: %s",
                    exc,
                )

        # 6-3) Run the algorithm (within run_pipeline)
        # Redact bulky internal arrays (e.g. the TopoUSM Fast coarse field) from logs/metadata.
        _log_params = {
            k: v for k, v in params.items()
            if not (k.startswith("_topousm_fast_coarse_field")
                    or k in ("_fractal_large_fields", "_vs_large_fields",
                             "_sss_large_fields", "_overview_coarse_dem"))
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
            "_topousm_fast_coarse_field", "_topousm_fast_small_radii", "_topousm_fast_small_weights",
            "_topousm_fast_w_large", "_topousm_fast_full_shape", "_topousm_fast_field_offset",
            "_fractal_large_fields", "_fractal_full_shape",
            "_vs_large_fields", "_vs_full_shape",
            "_sss_large_fields", "_sss_full_shape",
            "_overview_coarse_dem", "_overview_decimation",
        ):
            params.pop(_k, None)

        # 6-3.5) Quantize the output dtype (float32 passes through). NaN (NoData) -> 0,
        # valid values are linearly stretched from the algorithm's native range to [1, levels].
        out_np_dtype = "float32"
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
            # unscale the band and present it as Float32 over the physical range,
            # which looks "quantized float with NoData=0" and is confusing for a
            # display product.  The DN<->value mapping is still logged and
            # recorded in the COG 'parameters'/value_range attrs for anyone who
            # needs to recover physical units.  Same policy as the tile backend.
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
        elif algorithm in ['topousm_fast', 'fractal_anomaly']:
            # Signed terrain anomaly outputs map the robust p99(|value|) to +/-1;
            # the tail beyond +/-1 passes through unclipped.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "~-1 to +1 (p99-normalized, tail unclipped)",
                "normal_range": "-1 to +1",
                "normal_percentile": "99",
                "data_type": "float32"
            }
        elif algorithm == 'blur':
            # Raw smoothed elevation -- same units as the input DEM.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "elevation (input units)",
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
            # Unsigned analysis outputs map the robust p99 to +1; normalized
            # outputs keep an unclipped tail past 1.
            attrs = {
                **dem.attrs,
                "algorithm": algorithm,
                "parameters": str(params),
                "processing": f"Dask-CUDA {algorithm.upper()}",
                "value_range": "~0 to 1 (p99-normalized, tail unclipped)",
                "normal_range": "0 to 1",
                "normal_percentile": "99",
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

