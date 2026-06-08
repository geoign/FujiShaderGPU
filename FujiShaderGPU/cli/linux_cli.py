"""
FujiShaderGPU/cli/linux_cli.py
CLI for Linux - Dask-CUDA processing implementation.

All algorithm/output/spatial arguments are defined once in ``cli/args.py`` and
added by the shared base class; this module only declares the Dask-specific
options and wires the parsed args into ``run_pipeline``.
"""
from typing import List
import argparse
import os

import GPUtil
import numpy as np
import rasterio

from .base import BaseCLI
from .args import (
    DASK_ARGS,
    add_arguments,
    build_algo_params,
    parse_nodata_override,
    parse_output_range,
)
from ..algorithms.dask_registry import ALGORITHMS as DASK_ALGORITHMS


class LinuxCLI(BaseCLI):
    """CLI implementation for Linux (Dask-CUDA processing)."""

    def get_description(self) -> str:
        return """FujiShaderGPU - ultra-fast terrain analysis tool (Linux/Dask-CUDA)

Runs various terrain analyses on huge DEMs (200,000x200,000px COG) and
writes them out as Cloud-Optimized GeoTIFF."""

    def get_epilog(self) -> str:
        algos = ", ".join(self.get_supported_algorithms())
        return f"""
    Examples:
    # RVI: analyze terrain and auto-determine radii (recommended, fast)
    fujishadergpu input.tif output.tif

    # RVI: specify radii manually (new method, fast)
    fujishadergpu input.tif output.tif --radii 4,16,64,256

    # Other algorithms
    fujishadergpu input.tif output.tif --algo hillshade

    # Specify a large chunk size
    fujishadergpu input.tif output.tif --algo rvi --chunk 4096

    All available algorithms:
    {algos}
    """

    def get_supported_algorithms(self) -> List[str]:
        """Algorithms supported on Linux (all of them)."""
        return list(DASK_ALGORITHMS.keys())

    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """Add Dask-CUDA backend options (shared algorithm args live in base)."""
        add_arguments(parser, DASK_ARGS)

    def _validate_platform_args(self, args: argparse.Namespace):
        # Combine the auto-radii enable/disable flags.
        if hasattr(args, "auto_radii") and hasattr(args, "no_auto_radii"):
            args.auto_radii = args.auto_radii and not args.no_auto_radii

        # Combine the global-stats enable/disable flags.
        if hasattr(args, "use_global_stats") and hasattr(args, "no_global_stats"):
            args.use_global_stats = args.use_global_stats and not args.no_global_stats

        # RVI needs either explicit radii or auto-determination.
        if args.algorithm == "rvi":
            if not getattr(args, "radii_list", None) and not getattr(args, "auto_radii", True):
                self.parser.error("Specify radii or enable --auto-radii")

    def _resolve_pixel_size(self, args: argparse.Namespace, input_path: str):
        """Auto-detect pixel size in meters when the user did not specify it."""
        if getattr(args, "pixel_size", None) is not None:
            self.logger.info(f"User-specified pixel size: {args.pixel_size}m")
            return

        with rasterio.open(input_path) as src:
            if src.crs and src.crs.is_geographic:
                # Geographic CRS (lat/lon): convert degrees to meters at center latitude.
                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                lat_rad = np.radians(center_lat)
                meters_per_degree_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + \
                    1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
                meters_per_degree_lon = 111412.84 * np.cos(lat_rad) - \
                    93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
                pixel_size_x_deg = abs(src.transform[0])
                pixel_size_y_deg = abs(src.transform[4])
                pixel_size_x_m = pixel_size_x_deg * meters_per_degree_lon
                pixel_size_y_m = pixel_size_y_deg * meters_per_degree_lat
                args.pixel_size = (pixel_size_x_m + pixel_size_y_m) / 2
                self.logger.info(f"Geographic CRS detected: center latitude {center_lat:.2f}°")
                self.logger.info(f"Pixel size: {pixel_size_x_deg:.6f}° x {pixel_size_y_deg:.6f}°")
                self.logger.info(f"In meters: {args.pixel_size:.2f}m")
            else:
                # Projected CRS (already metric).
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                if src.crs:
                    units = src.crs.linear_units
                    if units and units.lower() not in ("metre", "meter"):
                        self.logger.warning(f"CRS unit is '{units}'; treating it as meters.")
                args.pixel_size = (pixel_size_x + pixel_size_y) / 2
                self.logger.info(f"Projected CRS: pixel size {args.pixel_size:.2f}m")

    def execute(self, args: argparse.Namespace):
        """Run Dask-CUDA processing."""
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"] = "0.70"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__SPILL"] = "0.75"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"] = "0.85"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"] = "0.95"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "30s"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
        os.environ["DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT"] = "60s"

        # Dynamic RMM settings from auto_tune.
        from FujiShaderGPU.config.auto_tune import compute_rmm_pool_gb
        gpus = GPUtil.getGPUs()
        gpu_memory_gb = gpus[0].memoryTotal / 1024 if gpus else 16  # conservative default
        _rmm_gb = compute_rmm_pool_gb(gpu_memory_gb)
        os.environ["RMM_ALLOCATOR"] = "pool"
        os.environ["RMM_POOL_SIZE"] = f"{_rmm_gb}GB"
        os.environ["RMM_MAXIMUM_POOL_SIZE"] = f"{int(_rmm_gb * 1.1)}GB"

        params = self.get_common_params(args)
        self._resolve_pixel_size(args, params["input_path"])

        self.logger.info("=== Dask-CUDA terrain analysis ===")
        self.logger.info(f"Input: {args.input}")
        self.logger.info(f"Output: {args.output}")
        self.logger.info(f"Algorithm: {args.algorithm}")

        algo_params = build_algo_params(args)
        # RVI auto-determines radii unless explicit --radii were given.
        auto_radii = getattr(args, "radii_list", None) is None and getattr(args, "auto_radii", True)

        try:
            from ..core.dask_processor import run_pipeline

            run_pipeline(
                src_cog=params["input_path"],
                dst_cog=params["output_path"],
                algorithm=args.algorithm,
                chunk=getattr(args, "chunk", None),
                show_progress=params["show_progress"],
                auto_radii=auto_radii,
                memory_fraction=getattr(args, "memory_fraction", None),
                nodata_override=parse_nodata_override(getattr(args, "nodata", None)),
                output_dtype=getattr(args, "output_dtype", "float32"),
                output_range=parse_output_range(getattr(args, "output_range", None)),
                **algo_params,
            )
        except Exception as e:
            self.logger.error(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
            raise
