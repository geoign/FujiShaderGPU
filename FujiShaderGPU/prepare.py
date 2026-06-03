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
        "--nodata",
        type=str,
        default=None,
        help="Explicit NoData value (e.g. -9999, 0, nan); replaced with NaN before filling",
    )
    parser.add_argument(
        "--no-detect-nodata",
        dest="detect_nodata",
        action="store_false",
        help="Disable auto-detection of undeclared NoData from a constant border (default: enabled)",
    )
    parser.set_defaults(detect_nodata=True)
    parser.add_argument(
        "--nodata-border-fraction",
        type=float,
        default=0.5,
        help="Auto-detection: minimum fraction of the outer ring the value must occupy (default: 0.5)",
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


def _parse_nodata(raw_value):
    """Parse a --nodata string into a float (or None). 'nan' is accepted but is a
    no-op for filling (already NaN); finite values are converted to NaN."""
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid --nodata value: {raw_value}") from exc


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("=== DEM preprocessing (COG-ification + fill: %s) ===", args.fill_mode)
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
            nodata_override=_parse_nodata(args.nodata),
            detect_nodata=args.detect_nodata,
            nodata_border_fraction=args.nodata_border_fraction,
        )
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
