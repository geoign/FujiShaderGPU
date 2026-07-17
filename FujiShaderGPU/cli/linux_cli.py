"""
FujiShaderGPU/cli/linux_cli.py
CLI for Linux - Dask-CUDA processing implementation.

All algorithm/output/spatial arguments are defined once in ``cli/args.py`` and
added by the shared base class; this module only declares the Dask-specific
options and wires the parsed args into ``run_pipeline``.
"""
from typing import List
import argparse

import rasterio

from .base import BaseCLI
from .args import (
    DASK_ARGS,
    add_arguments,
    build_algo_params,
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
    # TopoUSM Fast: analyze terrain and auto-determine radii (recommended, fast)
    fujishadergpu input.tif output.tif

    # TopoUSM Fast: specify radii manually (new method, fast)
    fujishadergpu input.tif output.tif --radii 4,16,64,256

    # Other algorithms
    fujishadergpu input.tif output.tif --algo hillshade

    # Specify a large chunk size
    fujishadergpu input.tif output.tif --algo topousm_fast --chunk 4096

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
        # TopoUSM Fast needs either explicit radii or auto-determination.
        if args.algorithm == "topousm_fast":
            if not getattr(args, "radii_list", None) and not getattr(args, "auto_radii", True):
                self.parser.error("Specify radii or enable --auto-radii")

    def _resolve_pixel_size(self, args: argparse.Namespace, input_path: str):
        """Auto-detect pixel size in meters when the user did not specify it."""
        if getattr(args, "pixel_size", None) is not None:
            self.logger.info(f"User-specified pixel size: {args.pixel_size}m")
            return

        # Zarr stores are not openable via rasterio/GDAL the way the Dask pipeline
        # reads them (xarray/rioxarray); probing with rasterio.open() here would
        # raise. run_pipeline detects the pixel size from the xarray metadata, so
        # leave args.pixel_size as None (auto-detect downstream).
        from ..core.dask_io import is_zarr_path
        if is_zarr_path(input_path):
            self.logger.info(
                "Zarr input: pixel size will be auto-detected from xarray/rioxarray "
                "metadata in run_pipeline."
            )
            return

        # args.pixel_size stays None in every auto-detect branch: run_pipeline
        # treats an explicit pixel_size as an isotropic override (is_geo=False,
        # both axes forced equal), so auto-setting the mean here silently
        # discarded the per-axis anisotropy -- ~10%/axis at mid latitudes on
        # geographic DEMs.  run_pipeline's own metadata detection keeps the
        # signed per-axis metre scales; this probe only logs what it will find.
        with rasterio.open(input_path) as src:
            if src.crs and src.crs.is_geographic:
                from ..io.raster_info import meters_per_degree

                bounds = src.bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                m_lon, m_lat = meters_per_degree(center_lat)
                pixel_size_x_deg = abs(src.transform[0])
                pixel_size_y_deg = abs(src.transform[4])
                self.logger.info(f"Geographic CRS detected: center latitude {center_lat:.2f} deg")
                self.logger.info(
                    f"Pixel size: {pixel_size_x_deg:.6f} deg x {pixel_size_y_deg:.6f} deg "
                    f"(~{pixel_size_x_deg * m_lon:.2f}m x {pixel_size_y_deg * m_lat:.2f}m); "
                    "anisotropic scales are auto-detected in run_pipeline"
                )
            else:
                # Projected CRS (already metric).
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                if src.crs:
                    units = src.crs.linear_units
                    if units and units.lower() not in ("metre", "meter"):
                        self.logger.warning(f"CRS unit is '{units}'; treating it as meters.")
                self.logger.info(
                    f"Projected CRS: pixel size {pixel_size_x:.2f}m x {pixel_size_y:.2f}m "
                    "(auto-detected in run_pipeline)"
                )

    def execute(self, args: argparse.Namespace):
        """Run Dask-CUDA processing."""
        params = self.get_common_params(args)
        self._resolve_pixel_size(args, params["input_path"])

        self.logger.info("=== Dask-CUDA terrain analysis ===")
        self.logger.info(f"Input: {args.input}")
        self.logger.info(f"Output: {args.output}")
        self.logger.info(f"Algorithm: {args.algorithm}")

        algo_params = build_algo_params(args)
        # TopoUSM Fast auto-determines radii unless explicit --radii were given.
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
                nodata_override=args.nodata_override,
                output_dtype=getattr(args, "output_dtype", "float32"),
                output_range=args.output_range_value,
                pixel_size=getattr(args, "pixel_size", None),
                **algo_params,
            )
        except Exception as e:
            self.logger.error(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
            raise
