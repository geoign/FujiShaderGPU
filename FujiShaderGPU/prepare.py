"""
FujiShaderGPU/prepare.py

Preprocessing CLI: convert any GDAL-readable raster into a FujiShaderGPU-ready
Cloud Optimized GeoTIFF (ZSTD + internal overviews, single-band float32), with
optional NoData void filling.

Usage
-----
    python -m FujiShaderGPU.prepare input.(tif|img|vrt|...) output_cog.tif
    python -m FujiShaderGPU.prepare input.tif out.tif --fill-mode enclosed
    python -m FujiShaderGPU.prepare input.tif out.tif --fill-mode all --force

The FujiShaderGPU main pipeline assumes its input is a COG **with overviews**;
running inputs through this command first guarantees fast spatial/multi-scale
processing and removes NoData filling from the per-tile/per-chunk hot path.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .io.dem_preprocess import FILL_MODES, preprocess_dem_to_cog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m FujiShaderGPU.prepare",
        description=(
            "Convert any GDAL raster into a FujiShaderGPU-compatible COG (overview+ZSTD, float32), "
            "optionally filling NoData voids -- a preprocessing command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NoData fill modes (--fill-mode):
  none      do not fill (keep NoData)
  enclosed  fill only interior voids, not the border-connected exterior (NoData touching the raster edge = sea/outside) [default]
  all       fill every NoData incl. the exterior and remove NoData entirely (dense raster; for 3D models, etc.)

Examples:
  python -m FujiShaderGPU.prepare input.tif output_cog.tif
  python -m FujiShaderGPU.prepare input.img output_cog.tif --fill-mode all --force
""",
    )
    parser.add_argument("input", help="Input raster (any format GDAL can read)")
    parser.add_argument("output", help="Output COG (.tif)")
    parser.add_argument(
        "--fill-mode",
        choices=FILL_MODES,
        default="enclosed",
        help="NoData fill mode (default: enclosed)",
    )
    parser.add_argument(
        "--coarse-max",
        type=int,
        default=2048,
        help="Longest side of the coarse fill grid (px); larger is finer but slower (default: 2048)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="COG tile/block size (default: 512)",
    )
    parser.add_argument(
        "--overview-count",
        type=int,
        default=8,
        help="Number of overview levels to generate (default: 8)",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=1,
        help="ZSTD compression level (default: 1)",
    )
    parser.add_argument(
        "--num-threads",
        default="ALL_CPUS",
        help="Number of GDAL parallel threads (default: ALL_CPUS)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Worker processes for the parallel fill/streaming pass (default: auto = "
            "min(container CPU budget, band count); the container CPU budget is the "
            "cgroup quota, not the host core count). Use 1 to force the serial path."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=str,
        default="auto",
        help=(
            "NoData handling (default: auto). 'auto' infers NoData: (1) the "
            "declared metadata NoData, then any undeclared (2) sentinel/extreme "
            "value dominating the grid (0, -9999, int min/max, ...) or (3) value "
            "dominating the border; (4) none if nothing matches. A number forces "
            "that value; 'nan'; 'none' disables inference (treat all as valid)."
        ),
    )
    parser.add_argument(
        "--no-detect-nodata",
        dest="detect_nodata",
        action="store_false",
        help="Disable the auto inference even with --nodata auto (alias for --nodata none)",
    )
    parser.set_defaults(detect_nodata=True)
    parser.add_argument(
        "--nodata-border-fraction",
        type=float,
        default=0.5,
        help="Auto rule 3: minimum fraction of the outer ring the value must occupy (default: 0.5)",
    )
    parser.add_argument(
        "--nodata-sentinel-fraction",
        type=float,
        default=0.05,
        help="Auto rule 2: minimum fraction of the whole grid a sentinel/extreme must occupy (default: 0.05)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output even if it exists",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )
    return parser


def _interpret_nodata(raw_value, detect_default: bool):
    """Map the ``--nodata`` string to ``(nodata_override, detect_nodata)``.

    - ``auto`` (default): infer (rules 1-4); ``detect_nodata`` stays on.
    - ``none`` / ``off``: no inference, no override (treat all as valid).
    - ``nan``: treat NaN as NoData (already the float sentinel); no inference.
    - a number: force that value as NoData; no inference.
    ``detect_default`` is ``False`` when ``--no-detect-nodata`` was passed, which
    forces inference off regardless.
    """
    text = "auto" if raw_value is None else str(raw_value).strip().lower()
    if text in {"auto", ""}:
        return None, bool(detect_default)
    if text in {"none", "off"}:
        return None, False
    if text in {"nan", "+nan", "-nan"}:
        return float("nan"), False
    try:
        return float(raw_value), False
    except ValueError as exc:
        raise ValueError(
            f"Invalid --nodata value: {raw_value!r} (use a number, 'auto', 'none', or 'nan')"
        ) from exc


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    nodata_override, detect_nodata = _interpret_nodata(args.nodata, args.detect_nodata)
    logger.info(
        "=== DEM preprocessing (COG-ification + fill: %s | nodata: %s) ===",
        args.fill_mode,
        "auto" if (detect_nodata and nodata_override is None) else
        ("none" if (not detect_nodata and nodata_override is None) else nodata_override),
    )
    try:
        preprocess_dem_to_cog(
            args.input,
            args.output,
            fill_mode=args.fill_mode,
            coarse_max=args.coarse_max,
            block_size=args.block_size,
            overview_count=args.overview_count,
            zstd_level=args.zstd_level,
            num_threads=args.num_threads,
            overwrite=args.force,
            nodata_override=nodata_override,
            detect_nodata=detect_nodata,
            nodata_border_fraction=args.nodata_border_fraction,
            nodata_sentinel_fraction=args.nodata_sentinel_fraction,
            max_workers=args.workers,
        )
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
