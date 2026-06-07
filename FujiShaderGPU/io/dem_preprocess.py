"""DEM preprocessing: convert any GDAL raster to a FujiShaderGPU-ready COG.

This module turns an arbitrary GDAL-readable raster into a Cloud Optimized
GeoTIFF (ZSTD + internal overviews, float32) that the FujiShaderGPU pipeline
expects, optionally filling NoData voids during the conversion.

NoData fill is intentionally a **low-frequency** operation: the value inside a
void is best estimated by a smooth surface derived from the surrounding terrain.
We therefore fill on a coarse overview grid (a few thousand pixels) -- where
edge-connectivity and interpolation are effectively free -- and bilinearly
upsample that surface to fill the full-resolution voids while streaming the
output.  This keeps the cost almost independent of the full raster size.

Fill modes
----------
- ``none``      : no filling; NoData is preserved.
- ``enclosed``  : fill only interior voids (NoData *not* connected to the raster
                  border).  Border-connected NoData (e.g. ocean / dataset
                  exterior) is kept as NoData.  **Default.**
- ``all``       : fill every NoData cell, including border-connected regions, and
                  emit a fully dense raster with no NoData (useful for 3D model
                  generation where NaN/NoData is not allowed).  Note that filling
                  large exterior regions is a coarse extrapolation.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as rio_window_transform
from osgeo import gdal

from ..utils.cpu import container_cpu_count
from ..utils.memory import container_memory_available_gb
from ..utils.nodata_handler import _edge_connected_mask
from ..utils.paths import resolve_tmp_dir, safe_abspath, safe_unlink

logger = logging.getLogger(__name__)

FILL_MODES = ("none", "enclosed", "all")

# Strips cut per worker for the parallel fill pass.  >1 so the process pool can
# load-balance uneven strips (ocean compresses fast, terrain is heavy) instead
# of stranding fast workers while a few heavy strips finish the tail.
STRIPS_PER_WORKER = 6


# ---------------------------------------------------------------------------
# Coarse-grid fill (the expensive work happens here, on a small array)
# ---------------------------------------------------------------------------
def _fill_coarse_surface(coarse: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a fully-valued smooth surface over a small coarse grid.

    Valid cells are preserved exactly; invalid cells receive a valid-weighted
    Gaussian estimate (smooth), falling back to the nearest valid value where the
    Gaussian support does not reach (e.g. the middle of a very large void).
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    out = coarse.astype(np.float32, copy=True)
    if valid.all():
        return out
    if not valid.any():
        # No reference data at all; nothing meaningful to fill with.
        return np.zeros_like(out, dtype=np.float32)

    # Nearest valid value for every cell (guarantees a finite value everywhere).
    nearest_idx = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    nearest = out[tuple(nearest_idx)]

    # Valid-weighted Gaussian smoothing removes nearest-neighbour (Voronoi) seams
    # in the filled region while keeping valid cells exact.
    sigma = float(max(1.0, min(coarse.shape) / 64.0))
    weights = valid.astype(np.float32)
    sv = gaussian_filter(np.where(valid, coarse, 0.0).astype(np.float32), sigma, mode="nearest")
    sw = gaussian_filter(weights, sigma, mode="nearest")
    smooth = np.where(sw > 1e-6, sv / np.maximum(sw, 1e-6), nearest)

    out = np.where(valid, coarse, smooth).astype(np.float32)
    out = np.where(np.isfinite(out), out, nearest).astype(np.float32)
    return out


def _detect_border_nodata(
    sample: np.ndarray,
    valid: np.ndarray,
    *,
    min_border_fraction: float = 0.5,
    min_total_fraction: float = 0.02,
) -> Optional[float]:
    """Guess an *undeclared* NoData value from a dominant constant border.

    Many DEMs ship with a wide constant frame (sea / dataset exterior) whose
    NoData tag was lost during conversion.  When a single finite value occupies
    a large fraction of the raster's outer ring -- and a non-trivial share of the
    whole grid -- it is almost certainly that forgotten sentinel rather than real
    terrain.  Returns the value, or ``None`` when no clear constant frame exists.

    ``sample`` must be a NEAREST-resampled coarse grid (no averaging) so exterior
    cells keep the *exact* sentinel value; ``valid`` excludes cells already masked
    by the declared NoData (so an already-tagged sentinel is not re-reported).
    """
    h, w = sample.shape
    if h < 8 or w < 8:
        return None
    ring = np.concatenate([sample[0, :], sample[-1, :], sample[1:-1, 0], sample[1:-1, -1]])
    ring_valid = np.concatenate([valid[0, :], valid[-1, :], valid[1:-1, 0], valid[1:-1, -1]])
    ring = ring[ring_valid & np.isfinite(ring)]
    if ring.size == 0:
        return None
    vals, counts = np.unique(ring, return_counts=True)
    i = int(np.argmax(counts))
    cand = float(vals[i])
    if counts[i] / ring.size < float(min_border_fraction):
        return None
    # Require the value to cover a non-trivial share of the whole grid, so a thin
    # genuine coastal strip on one edge is not mistaken for a NoData frame.
    if float(np.mean(sample == np.float32(cand))) < float(min_total_fraction):
        return None
    return cand


def _coarse_shape(width: int, height: int, coarse_max: int) -> tuple[int, int]:
    scale = max(width / coarse_max, height / coarse_max, 1.0)
    cw = max(1, int(round(width / scale)))
    ch = max(1, int(round(height / scale)))
    return ch, cw


def _band_height(width: int, target_pixels: int = 16_000_000) -> int:
    """Rows per streaming band.

    Bounded by a pixel budget rather than bytes: filling builds float64
    coordinate grids (2x) for ``map_coordinates`` on top of the float32 data, so
    a full-width band of ``target_pixels`` keeps transient memory at roughly a
    few hundred MB regardless of raster width.
    """
    rows = int(target_pixels // max(1, width))
    return int(max(64, min(4096, rows)))


# ---------------------------------------------------------------------------
# COG writer
# ---------------------------------------------------------------------------
def _cog_creation_options(
    *,
    block_size: int,
    overview_count: int,
    zstd_level: int,
    num_threads: str,
) -> list[str]:
    return [
        "COMPRESS=ZSTD",
        f"LEVEL={zstd_level}",
        "PREDICTOR=3",  # float32
        f"BLOCKSIZE={block_size}",
        "OVERVIEWS=IGNORE_EXISTING",
        "OVERVIEW_RESAMPLING=AVERAGE",
        f"OVERVIEW_COUNT={overview_count}",
        "OVERVIEW_COMPRESS=ZSTD",
        "BIGTIFF=YES",
        f"NUM_THREADS={num_threads}",
    ]


def _validate_cog_overviews(dst_cog: Path) -> None:
    ds = gdal.Open(str(dst_cog), gdal.GA_ReadOnly)
    ov = ds.GetRasterBand(1).GetOverviewCount() if ds is not None else 0
    ds = None
    if ov <= 0:
        raise RuntimeError(f"COG output unexpectedly has no overviews: {dst_cog}")
    logger.info("COG overview levels: %d", ov)


def _translate_to_cog(
    src_tiff: Path,
    dst_cog: Path,
    *,
    block_size: int,
    overview_count: int,
    zstd_level: int,
    num_threads: str,
) -> None:
    """Convert a (temporary) GeoTIFF into a COG with internal ZSTD overviews.

    The source temp GeoTIFF already carries the intended NoData (NaN for
    none/enclosed, unset for all), so it is inherited here -- avoiding any
    float-NaN formatting issues with the translate options.
    """
    if gdal.GetDriverByName("COG") is None:
        raise RuntimeError(
            "GDAL COG driver is unavailable (GDAL >= 3.1 required). "
            "Update GDAL to produce FujiShaderGPU-compatible COGs."
        )
    creation_options = _cog_creation_options(
        block_size=block_size,
        overview_count=overview_count,
        zstd_level=zstd_level,
        num_threads=num_threads,
    )
    logger.info("Writing COG (ZSTD + AVERAGE overviews x%d): %s", overview_count, dst_cog)
    result = gdal.Translate(
        str(dst_cog),
        str(src_tiff),
        format="COG",
        creationOptions=creation_options,
    )
    if result is None:
        raise RuntimeError(f"COG translate failed: {dst_cog}")
    result = None

    _validate_cog_overviews(dst_cog)


def _translate_source_to_cog_fast(
    src_path: Path,
    dst_cog: Path,
    *,
    block_size: int,
    overview_count: int,
    zstd_level: int,
    num_threads: str,
    src_nodata: Optional[float],
    dst_nodata: Optional[float],
) -> None:
    """Translate source band 1 directly to COG when Python-side fill is a no-op."""
    if gdal.GetDriverByName("COG") is None:
        raise RuntimeError(
            "GDAL COG driver is unavailable (GDAL >= 3.1 required). "
            "Update GDAL to produce FujiShaderGPU-compatible COGs."
        )

    creation_options = _cog_creation_options(
        block_size=block_size,
        overview_count=overview_count,
        zstd_level=zstd_level,
        num_threads=num_threads,
    )
    dst_cog.parent.mkdir(parents=True, exist_ok=True)

    translate_src = str(src_path)
    vrt_path: Optional[str] = None
    if src_nodata is not None and dst_nodata is not None:
        # BuildVRT has srcNodata support while TranslateOptions does not.  The
        # VRT keeps the fast path inside GDAL while preserving the NaN NoData
        # contract used by the streaming path.
        vrt_path = f"/vsimem/fujishader_prepare_{os.getpid()}_{uuid.uuid4().hex}.vrt"
        vrt = gdal.BuildVRT(
            vrt_path,
            [str(src_path)],
            bandList=[1],
            srcNodata=float(src_nodata),
            VRTNodata=float(dst_nodata),
        )
        if vrt is None:
            raise RuntimeError(f"VRT build failed for fast COG translate: {src_path}")
        vrt = None
        translate_src = vrt_path

    try:
        kwargs = {
            "format": "COG",
            "outputType": gdal.GDT_Float32,
            "bandList": [1],
            "creationOptions": creation_options,
        }
        if dst_nodata is not None:
            kwargs["noData"] = float(dst_nodata)
        logger.info(
            "Writing COG directly from source (ZSTD + AVERAGE overviews x%d): %s",
            overview_count,
            dst_cog,
        )
        result = gdal.Translate(str(dst_cog), translate_src, **kwargs)
        if result is None:
            raise RuntimeError(f"Direct COG translate failed: {dst_cog}")
        result = None
    finally:
        if vrt_path is not None:
            gdal.Unlink(vrt_path)

    _validate_cog_overviews(dst_cog)


def _dedupe_finite_nodata_values(
    src_nodata: Optional[float],
    extra_nodata: list[float],
) -> list[float]:
    values: list[float] = []
    if src_nodata is not None and np.isfinite(src_nodata):
        values.append(float(src_nodata))
    for value in extra_nodata:
        if not np.isfinite(value):
            continue
        if not any(np.isclose(value, existing) for existing in values):
            values.append(float(value))
    return values


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preprocess_dem_to_cog(
    input_path: str,
    output_path: str,
    *,
    fill_mode: str = "enclosed",
    coarse_max: int = 2048,
    block_size: int = 512,
    overview_count: int = 8,
    zstd_level: int = 1,
    num_threads: str = "ALL_CPUS",
    overwrite: bool = False,
    nodata_override: Optional[float] = None,
    detect_nodata: bool = True,
    nodata_border_fraction: float = 0.5,
    max_workers: Optional[int] = None,
) -> None:
    """Convert ``input_path`` (any GDAL raster) into a FujiShaderGPU-ready COG.

    The output is a single-band float32 COG (ZSTD, ``block_size`` tiles, internal
    AVERAGE overviews).  CRS and pixel grid are preserved; **no reprojection** is
    performed.  Band 1 is used.  See the module docstring for ``fill_mode``.

    NoData handling (all converted to float NaN *before* filling):
    - the raster's declared NoData (read via masked I/O);
    - ``nodata_override`` -- an explicit sentinel to treat as NoData even when the
      raster declares none (or declares a different one);
    - when ``detect_nodata`` is true (default), a *dominant constant border* is
      auto-detected and treated as an undeclared NoData sentinel.  Disable with
      ``detect_nodata=False``; ``nodata_border_fraction`` (default 0.5) is the
      minimum share of the raster's outer ring the value must occupy.
    """
    fill_mode = str(fill_mode).lower()
    if fill_mode not in FILL_MODES:
        raise ValueError(f"Unknown fill_mode={fill_mode!r}. Choose from {FILL_MODES}.")

    in_path = Path(input_path)
    out_path = Path(output_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input raster not found: {in_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path} (use overwrite=True / --force)."
        )

    with rasterio.open(in_path) as src:
        if src.count < 1:
            raise ValueError("Input raster has no bands.")
        if src.count > 1:
            logger.warning("Input has %d bands; only band 1 is used (DEM).", src.count)
        width, height = int(src.width), int(src.height)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        logger.info(
            "Input: %dx%d, dtype=%s, nodata=%s, CRS=%s",
            width, height, src.dtypes[0], src_nodata, src_crs,
        )

        # ---- 1) Build the coarse fill surface (only when filling) -----------
        ch, cw = _coarse_shape(width, height, coarse_max)
        coarse_ma = src.read(
            1, out_shape=(ch, cw), resampling=Resampling.average,
            out_dtype=np.float32, masked=True,
        )
        coarse = np.ma.getdata(coarse_ma).astype(np.float32, copy=False)
        cmask = np.ma.getmaskarray(coarse_ma)

        # ---- Extra NoData sentinels -> NaN (override + auto-detected) --------
        # The declared NoData is already masked by the masked read above.  Here
        # we additionally honour an explicit override and, by default, sniff out
        # an *undeclared* constant border (a NoData frame whose tag was lost in
        # conversion) so it is not treated as real terrain by the fill / pipeline.
        extra_nodata: list[float] = []
        if nodata_override is not None and np.isfinite(nodata_override):
            extra_nodata.append(float(nodata_override))
            logger.info("NoData override: %s -> NaN", float(nodata_override))
        if detect_nodata:
            nn_ma = src.read(
                1, out_shape=(ch, cw), resampling=Resampling.nearest,
                out_dtype=np.float32, masked=True,
            )
            nn = np.ma.getdata(nn_ma).astype(np.float32, copy=False)
            nn_valid = (~np.ma.getmaskarray(nn_ma)) & np.isfinite(nn)
            detected = _detect_border_nodata(
                nn, nn_valid, min_border_fraction=nodata_border_fraction,
            )
            if detected is not None:
                already = any(np.isclose(detected, v) for v in extra_nodata)
                declared = src_nodata is not None and np.isclose(detected, float(src_nodata))
                if not already and not declared:
                    extra_nodata.append(float(detected))
                    logger.warning(
                        "Auto-detected undeclared NoData from constant border: %s -> NaN "
                        "(disable with --no-detect-nodata / detect_nodata=False)",
                        float(detected),
                    )

        # Fold the sentinels into the coarse mask so the fill surface, edge
        # connectivity (enclosed), and hole detection treat them as NoData.
        for v in extra_nodata:
            cmask = cmask | (coarse == np.float32(v))
        cvalid = (~cmask) & np.isfinite(coarse)
        has_holes = bool((~cvalid).any())

        do_fill = fill_mode != "none" and has_holes
        surface = None
        exterior_coarse = None
        if do_fill:
            surface = _fill_coarse_surface(coarse, cvalid)
            if fill_mode == "enclosed":
                # NoData connected to the (global) coarse border = exterior; keep it.
                exterior_coarse = _edge_connected_mask(~cvalid)
                n_ext = int(exterior_coarse.sum())
                n_void = int((~cvalid).sum())
                logger.info(
                    "Fill=enclosed: coarse voids=%d, exterior(kept)=%d, filled=%d",
                    n_void, n_ext, n_void - n_ext,
                )
            else:  # all
                logger.info("Fill=all: every NoData cell is filled (output will be dense).")
        elif fill_mode != "none":
            logger.info("No NoData detected; fill is a no-op.")

        # Output NoData policy: dense for 'all', NaN sentinel otherwise.
        out_nodata: Optional[float] = None if fill_mode == "all" else float("nan")

        # If filling is disabled or the raster has no holes, keep the whole
        # conversion inside GDAL.  This avoids the full-resolution Python
        # read/modify/write staging pass and the temporary GeoTIFF.
        nodata_values = _dedupe_finite_nodata_values(src_nodata, extra_nodata)
        if not do_fill:
            fast_src_nodata: Optional[float] = None
            fast_dst_nodata = out_nodata
            can_fast_translate = True
            if out_nodata is None:
                # 'all' mode promises a dense output with no NoData metadata.
                # Without a fill pass, that is only equivalent when there is no
                # declared or inferred NoData to clear.
                can_fast_translate = (
                    not nodata_values
                    and not (src_nodata is not None and not np.isfinite(src_nodata))
                )
            elif len(nodata_values) > 1:
                # GDAL's direct Translate path can set one source NoData value
                # via a VRT.  Multiple finite sentinels still need the existing
                # streaming normalisation loop.
                can_fast_translate = False
            elif nodata_values:
                fast_src_nodata = nodata_values[0]

            if can_fast_translate:
                if fill_mode == "none":
                    logger.info("Fill=none: using direct COG translate fast path.")
                else:
                    logger.info("Fill is a no-op; using direct COG translate fast path.")
                _translate_source_to_cog_fast(
                    in_path,
                    out_path,
                    block_size=block_size,
                    overview_count=overview_count,
                    zstd_level=zstd_level,
                    num_threads=num_threads,
                    src_nodata=fast_src_nodata,
                    dst_nodata=fast_dst_nodata,
                )
                size_mb = os.path.getsize(out_path) / (1024 * 1024)
                logger.info(
                    "[OK] COG written: %s (%.1f MB, fill=%s)",
                    out_path,
                    size_mb,
                    fill_mode,
                )
                return

            logger.info(
                "No fill required, but NoData normalisation needs the streaming path."
            )

        # ---- 2) Stream full-resolution staging, then convert to COG ---------
        # Honor FUJISHADER_TMP_DIR / CPL_TMPDIR / TMPDIR (a large persistent
        # volume on RunPod/Colab) before defaulting next to the output, so the
        # full-resolution staging file(s) do not land on a small disk.
        tmp_parent, tmp_origin = resolve_tmp_dir(safe_abspath(out_path).parent)
        if tmp_origin:
            logger.info("Staging temporary file(s) in %s (from $%s)", tmp_parent, tmp_origin)
        band_rows = _band_height(width)

        base_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": src_crs,
            "transform": src_transform,
            "tiled": True,
            "blockxsize": block_size,
            "blockysize": block_size,
            "compress": "ZSTD",
            "zlevel": zstd_level,
            "predictor": 3,
            "BIGTIFF": "YES",
        }
        if out_nodata is not None:
            base_profile["nodata"] = out_nodata

        common = dict(
            width=width, height=height, band_rows=band_rows, ch=ch, cw=cw,
            do_fill=do_fill, surface=surface, exterior_coarse=exterior_coarse,
            extra_nodata=extra_nodata, fill_mode=fill_mode, out_nodata=out_nodata,
            out_path=out_path, block_size=block_size,
            overview_count=overview_count, zstd_level=zstd_level,
        )

        # The per-band fill is embarrassingly parallel across horizontal strips.
        # The serial Python loop is single-threaded and CPU-bound -- it leaves a
        # multi-core box almost entirely idle -- so for large rasters we fan the
        # strips out to worker processes (each writes its own GeoTIFF), mosaic
        # them with a VRT, and run a single multi-threaded COG translate.
        #
        # Size the pool to the *container's* CPU budget, not os.cpu_count():
        # under a CFS quota (RunPod/Colab/k8s) the host may show 64 cores while
        # the container is throttled to ~7.  Spawning 32 workers there just
        # oversubscribes the quota -- dozens of runnable processes thrash a
        # handful of effective cores and run slower than a right-sized pool.
        n_bands_total = math.ceil(height / band_rows)
        cpu_budget = container_cpu_count()
        if max_workers is None:
            n_workers = min(cpu_budget, n_bands_total)
        else:
            n_workers = max(1, min(int(max_workers), n_bands_total))
        logger.info(
            "Container CPU budget: %d cores (host reports %d); fill workers: %d",
            cpu_budget, os.cpu_count() or 1, n_workers,
        )

        if n_workers <= 1:
            _stream_fill_serial(src, tmp_parent, base_profile, num_threads, **common)
        else:
            _stream_fill_parallel(
                in_path, tmp_parent, base_profile, num_threads,
                n_workers=n_workers, n_bands_total=n_bands_total, **common,
            )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("[OK] COG written: %s (%.1f MB, fill=%s)", out_path, size_mb, fill_mode)


def _sample_coarse(
    surface: np.ndarray,
    exterior_coarse: Optional[np.ndarray],
    row_off: int,
    band_h: int,
    width: int,
    height: int,
    ch: int,
    cw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample the coarse surface (and exterior mask) for a row band.

    Coordinates are computed in the *global* coarse grid so that adjacent bands
    sample identical positions -> the upsampled fill is seamless.
    """
    from scipy.ndimage import map_coordinates

    rr = (row_off + np.arange(band_h, dtype=np.float64) + 0.5) * (ch / height) - 0.5
    cc = (np.arange(width, dtype=np.float64) + 0.5) * (cw / width) - 0.5
    grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
    coords = np.stack([grid_r, grid_c], axis=0)

    fill_vals = map_coordinates(
        surface, coords, order=1, mode="nearest", output=np.float32
    )
    if exterior_coarse is None:
        ext = np.zeros((band_h, width), dtype=bool)
    else:
        ext = map_coordinates(
            exterior_coarse.astype(np.float32), coords, order=1, mode="nearest"
        ) > 0.5
    return fill_vals, ext


def _apply_band_fill(
    band_ma,
    *,
    row_off: int,
    bh: int,
    width: int,
    height: int,
    ch: int,
    cw: int,
    do_fill: bool,
    surface,
    exterior_coarse,
    extra_nodata: list,
    fill_mode: str,
    out_nodata: Optional[float],
) -> np.ndarray:
    """NoData-normalise (and optionally fill) one full-width row band.

    The single source of truth for the per-band maths so the serial and parallel
    streaming paths produce identical pixels.  ``row_off`` is the band's offset in
    the *global* grid, so the coarse fill is sampled at the same positions in both
    paths and across strip boundaries (seamless).
    """
    arr = np.ma.getdata(band_ma).astype(np.float32, copy=True)
    wmask = np.ma.getmaskarray(band_ma) | ~np.isfinite(arr)
    for v in extra_nodata:
        wmask = wmask | (arr == np.float32(v))
    # Normalise every NoData cell to NaN up front, so finite sentinels (e.g.
    # -9999) are handled identically to declared NoData regardless of fill mode
    # (the fill step below then overwrites only the cells it is meant to fill).
    if wmask.any():
        arr = np.where(wmask, np.float32(np.nan), arr).astype(np.float32)

    if do_fill and wmask.any():
        # Sample the global coarse surface at this band's pixel centres
        # (bilinear) -- seamless across bands.
        fill_vals, ext = _sample_coarse(
            surface, exterior_coarse, row_off, bh, width, height, ch, cw,
        )
        if fill_mode == "enclosed":
            fill_here = wmask & ~ext
        else:  # all
            fill_here = wmask
        arr = np.where(fill_here, fill_vals, arr).astype(np.float32)

    if out_nodata is None:
        # 'all' mode must be fully dense; replace any residual NaN.
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    else:
        # Remaining masked cells become the NaN nodata sentinel.
        arr = np.where(np.isfinite(arr), arr, np.float32(np.nan))
    return arr


def _stream_fill_serial(
    src,
    tmp_parent: Path,
    base_profile: dict,
    num_threads: str,
    *,
    width: int,
    height: int,
    band_rows: int,
    ch: int,
    cw: int,
    do_fill: bool,
    surface,
    exterior_coarse,
    extra_nodata: list,
    fill_mode: str,
    out_nodata: Optional[float],
    out_path: Path,
    block_size: int,
    overview_count: int,
    zstd_level: int,
) -> None:
    """Single-process streaming: one temp GeoTIFF, then translate to COG."""
    profile = dict(base_profile)
    profile["num_threads"] = num_threads
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp.tif", dir=str(tmp_parent))
    os.close(fd)
    tmp_tiff = Path(tmp_name)
    try:
        with rasterio.open(tmp_tiff, "w", **profile) as dst:
            for row in range(0, height, band_rows):
                bh = min(band_rows, height - row)
                band_ma = src.read(
                    1, window=Window(0, row, width, bh),
                    out_dtype=np.float32, masked=True,
                )
                arr = _apply_band_fill(
                    band_ma, row_off=row, bh=bh, width=width, height=height,
                    ch=ch, cw=cw, do_fill=do_fill, surface=surface,
                    exterior_coarse=exterior_coarse, extra_nodata=extra_nodata,
                    fill_mode=fill_mode, out_nodata=out_nodata,
                )
                dst.write(arr, 1, window=Window(0, row, width, bh))
        _translate_to_cog(
            tmp_tiff, out_path, block_size=block_size,
            overview_count=overview_count, zstd_level=zstd_level,
            num_threads=num_threads,
        )
    finally:
        safe_unlink(tmp_tiff)


def _process_strip(task: tuple) -> str:
    """Worker process: fill one horizontal strip into its own GeoTIFF.

    Runs in a separate process, so it opens an independent input handle and
    writes a standalone strip; the parent mosaics the strips via a VRT.  Returns
    the strip path.
    """
    (in_path, strip_path, strip_start, strip_h, width, height, band_rows,
     ch, cw, surface, exterior_coarse, extra_nodata, do_fill, fill_mode,
     out_nodata, strip_profile, gdal_cachemax_mb) = task

    profile = dict(strip_profile)
    profile["height"] = strip_h
    profile["transform"] = rio_window_transform(
        Window(0, strip_start, width, strip_h), strip_profile["transform"],
    )
    # Bound this worker's GDAL block cache (process-wide; SetCacheMax takes
    # effect immediately, unlike the GDAL_CACHEMAX config option which is only
    # read when the cache is first created).  Keeps N workers within the cgroup.
    gdal.SetCacheMax(int(gdal_cachemax_mb) * 1024 * 1024)
    with rasterio.open(in_path) as src, \
            rasterio.open(strip_path, "w", **profile) as dst:
        for r in range(strip_start, strip_start + strip_h, band_rows):
            bh = min(band_rows, strip_start + strip_h - r)
            band_ma = src.read(
                1, window=Window(0, r, width, bh),
                out_dtype=np.float32, masked=True,
            )
            arr = _apply_band_fill(
                band_ma, row_off=r, bh=bh, width=width, height=height,
                ch=ch, cw=cw, do_fill=do_fill, surface=surface,
                exterior_coarse=exterior_coarse, extra_nodata=extra_nodata,
                fill_mode=fill_mode, out_nodata=out_nodata,
            )
            dst.write(arr, 1, window=Window(0, r - strip_start, width, bh))
    return strip_path


def _stream_fill_parallel(
    in_path: Path,
    tmp_parent: Path,
    base_profile: dict,
    num_threads: str,
    *,
    n_workers: int,
    n_bands_total: int,
    width: int,
    height: int,
    band_rows: int,
    ch: int,
    cw: int,
    do_fill: bool,
    surface,
    exterior_coarse,
    extra_nodata: list,
    fill_mode: str,
    out_nodata: Optional[float],
    out_path: Path,
    block_size: int,
    overview_count: int,
    zstd_level: int,
) -> None:
    """Multi-process streaming: per-strip GeoTIFFs -> VRT mosaic -> one COG translate."""
    # Cut MORE strips than workers so the pool load-balances dynamically.  Work
    # per row is very uneven -- ocean/NoData strips compress fast and finish
    # early, terrain strips are 3x heavier -- so equal-sized strips per worker
    # leave fast workers idle while a few heavy ones grind out the tail (only
    # ~2/7 cores busy near the end).  Many small strips let an idle worker steal
    # the next queued strip, keeping every core busy until the very end.
    n_strips = min(n_bands_total, max(n_workers, n_workers * STRIPS_PER_WORKER))
    bands_per_strip = math.ceil(n_bands_total / n_strips)
    # Strip writes are single-threaded (process-level parallelism already
    # saturates the cores); ALL_CPUS is reserved for the final COG translate.
    strip_profile = dict(base_profile)
    strip_profile["num_threads"] = "1"
    # Per-worker GDAL block cache, bounded by ~40% of container-available RAM so
    # the pool stays within the cgroup cap (default GDAL cache would be sized
    # from host RAM and N workers could together exceed the container limit).
    cache_budget_mb = int(container_memory_available_gb() * 1024 * 0.4)
    gdal_cachemax_mb = max(512, min(2048, cache_budget_mb // max(1, n_workers)))

    tasks = []
    strip_paths: list[Path] = []
    si = 0
    for strip_start in range(0, height, bands_per_strip * band_rows):
        strip_h = min(bands_per_strip * band_rows, height - strip_start)
        fd, sp = tempfile.mkstemp(suffix=f".strip{si:04d}.tif", dir=str(tmp_parent))
        os.close(fd)
        strip_paths.append(Path(sp))
        tasks.append((
            str(in_path), sp, strip_start, strip_h, width, height, band_rows,
            ch, cw, surface, exterior_coarse, extra_nodata, do_fill, fill_mode,
            out_nodata, strip_profile, gdal_cachemax_mb,
        ))
        si += 1

    vrt_path: Optional[Path] = None
    try:
        logger.info(
            "Parallel fill: %d strips (~%d rows each) across %d worker processes "
            "(GDAL cache %dMB/worker)",
            len(tasks), bands_per_strip * band_rows, n_workers, gdal_cachemax_mb,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_strip, t) for t in tasks]
            for fut in as_completed(futures):
                fut.result()  # re-raise any worker exception

        vrt_path = Path(
            tempfile.mkstemp(suffix=".mosaic.vrt", dir=str(tmp_parent))[1]
        )
        vrt = gdal.BuildVRT(str(vrt_path), [str(p) for p in strip_paths])
        if vrt is None:
            raise RuntimeError("BuildVRT failed to mosaic fill strips.")
        vrt = None  # flush/close

        _translate_to_cog(
            vrt_path, out_path, block_size=block_size,
            overview_count=overview_count, zstd_level=zstd_level,
            num_threads=num_threads,
        )
    finally:
        for p in strip_paths:
            safe_unlink(p)
        if vrt_path is not None:
            safe_unlink(vrt_path)
