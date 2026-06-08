"""
FujiShaderGPU/cli/windows_cli.py
CLI for Windows - tile-based processing implementation.

All algorithm/output/spatial arguments are defined once in ``cli/args.py`` and
added by the shared base class; this module only declares the tile/COG-specific
options and wires the parsed args into ``process_dem_tiles``.
"""
import os
from typing import List
import argparse

from .base import BaseCLI
from .args import (
    TILE_ARGS,
    add_arguments,
    build_algo_params,
    parse_nodata_override,
    parse_output_range,
)
from ..core.tile_processor import DEFAULT_ALGORITHMS


class WindowsCLI(BaseCLI):
    """CLI implementation for Windows (tile-based processing)."""

    def get_description(self) -> str:
        return "FujiShaderGPU - fast terrain analysis tool (Windows/tile-based processing)"

    def get_epilog(self) -> str:
        return """
Examples:
  # RVI (Ridge-Valley Index)
  fujishadergpu input.tif output.tif

  # Hillshade
  fujishadergpu input.tif output.tif --algorithm hillshade

  # Spatial RVI (manual radii)
  fujishadergpu input.tif output.tif --algorithm rvi --mode spatial --radii 4,16,64 --weights 0.5,0.3,0.2

  # Specify the GPU type
  fujishadergpu input.tif output.tif --gpu-type rtx4070

  # COG generation only (from existing tiles)
  fujishadergpu dummy.tif output.tif --cog-only --tmp-dir existing_tiles

Note: Windows and Linux share the same algorithm names and options.
      The main difference is the backend (Windows: tile processing / Linux: Dask-CUDA).
"""

    def get_supported_algorithms(self) -> List[str]:
        """Algorithms supported on Windows."""
        return list(DEFAULT_ALGORITHMS.keys())

    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """Add tile/COG backend options (shared algorithm args live in base)."""
        add_arguments(parser, TILE_ARGS)

    def _validate_platform_args(self, args: argparse.Namespace):
        # Combine the global-stats enable/disable flags (shared with Linux).
        if hasattr(args, "use_global_stats") and hasattr(args, "no_global_stats"):
            args.use_global_stats = args.use_global_stats and not args.no_global_stats

        # COG-only mode reads existing tiles instead of an input raster.
        if getattr(args, "cog_only", False):
            if not os.path.exists(args.tmp_dir):
                self.parser.error(f"--cog-only mode requires a tile directory: {args.tmp_dir}")
            args._skip_input_check = True

        if args.algorithm == "rvi" and not getattr(args, "no_auto_scale", False):
            self.logger.info("When radii are omitted, scales are auto-determined via terrain analysis")
        if args.cog_backend == "external" and not args.gdal_bin_dir:
            self.logger.warning(
                "With --cog-backend external, explicitly specifying --gdal-bin-dir is recommended"
            )

    def execute(self, args: argparse.Namespace):
        """Run tile-based processing."""
        from ..config.system_config import check_gdal_environment
        check_gdal_environment()

        params = self.get_common_params(args)
        params.update({
            "tile_size": args.tile_size,
            "padding": args.padding,
            "max_workers": args.max_workers,
            "nodata_threshold": args.nodata_threshold,
            "gpu_type": args.gpu_type,
            "multiscale_mode": not args.single_scale,
            "auto_scale_analysis": not args.no_auto_scale,
            "cog_only": args.cog_only,
            "cog_backend": args.cog_backend,
            "gdal_bin_dir": args.gdal_bin_dir,
        })

        if args.cog_only:
            self.logger.info("=== COG-only mode ===")
            self.logger.info(f"Tile directory: {args.tmp_dir}")
            self.logger.info(f"Output: {args.output}")
        else:
            self.logger.info(f"Input: {args.input}")
            self.logger.info(f"Output: {args.output}")
            self.logger.info(f"Algorithm: {args.algorithm}")
            self.logger.info(f"Mode: {'multiscale' if params['multiscale_mode'] else 'single-scale'}")
            self.logger.info(f"Spatial mode: {args.mode}")

        try:
            from ..core.tile_processor import process_dem_tiles

            if args.cog_only:
                params["input_path"] = params["input_path"] if os.path.exists(params["input_path"]) else "dummy_input.tif"

            algo_params = build_algo_params(args)

            process_dem_tiles(
                input_cog_path=params["input_path"],
                output_cog_path=params["output_path"],
                tmp_tile_dir=params["tmp_dir"],
                algorithm=params["algorithm"],
                tile_size=params["tile_size"],
                padding=params["padding"],
                sigma=10.0,
                max_workers=params["max_workers"],
                nodata_threshold=params["nodata_threshold"],
                nodata_override=parse_nodata_override(args.nodata),
                output_dtype=getattr(args, "output_dtype", "float32"),
                output_range=parse_output_range(getattr(args, "output_range", None)),
                gpu_type=params["gpu_type"],
                multiscale_mode=params["multiscale_mode"],
                pixel_size=params["pixel_size"],
                auto_scale_analysis=params["auto_scale_analysis"],
                cog_only=params["cog_only"],
                cog_backend=params["cog_backend"],
                gdal_bin_dir=params["gdal_bin_dir"],
                **algo_params,
            )

            self.logger.info("Done")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise
