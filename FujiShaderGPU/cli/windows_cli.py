"""
FujiShaderGPU/cli/windows_cli.py
CLI for Windows - tile-based processing implementation.
"""
import os
from typing import List
import argparse
from .base import BaseCLI
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

Note: Windows and Linux now share the same algorithm names.
      The main difference is the backend implementation (Windows: tile processing / Linux: Dask-CUDA).
"""
    
    def get_supported_algorithms(self) -> List[str]:
        """Algorithms supported on Windows."""
        return list(DEFAULT_ALGORITHMS.keys())
    
    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """Add Windows-specific arguments."""
        # Tile-processing options
        parser.add_argument(
            "--tile-size",
            type=int,
            help="Tile size (auto-detected when omitted)"
        )
        
        parser.add_argument(
            "--padding",
            type=int,
            help="Tile-boundary padding (auto-computed when omitted)"
        )
        
        parser.add_argument(
            "--max-workers",
            type=int,
            help="Number of parallel workers (auto-detected when omitted)"
        )
        
        parser.add_argument(
            "--nodata-threshold",
            type=float,
            default=1.0,
            help="NoData skip threshold (default: 1.0)"
        )

        parser.add_argument(
            "--nodata",
            type=str,
            default=None,
            help="Explicit NoData value (e.g. 0, -9999, nan)"
        )

        parser.add_argument(
            "--output-dtype",
            choices=["float32", "int16", "uint8"],
            default="float32",
            help="Output data type. int16/uint8 quantize for visualization (NoData=0) and shrink the COG (default: float32)"
        )

        parser.add_argument(
            "--output-range",
            type=str,
            default=None,
            help="Explicit quantization range 'lo,hi' (e.g. 0,90); defaults to the algorithm's native range"
        )
        
        # GPU options
        parser.add_argument(
            "--gpu-type",
            choices=["rtx4070", "t4", "l4", "a100", "auto"],
            default="auto",
            help="GPU type (default: auto)"
        )
        
        # Mode options
        parser.add_argument(
            "--single-scale",
            action="store_true",
            help="Force single-scale mode"
        )
        
        parser.add_argument(
            "--no-auto-scale",
            action="store_true",
            help="Disable automatic scale analysis"
        )

        parser.add_argument(
            "--mode",
            choices=["local", "spatial"],
            default="local",
            help="Compute mode: local (neighborhood) / spatial (radius integration). With spatial and no radii, YAML presets are used"
        )

        parser.add_argument(
            "--radii",
            type=str,
            help="Explicit spatial radii (px), e.g. 4,16,64; when omitted, YAML is auto-selected by pixel size"
        )

        parser.add_argument(
            "--weights",
            type=str,
            help="Spatial weights (e.g. 0.5,0.3,0.2); when omitted, YAML/equal weights are applied"
        )

        # Add algorithm-specific parameters
        parser.add_argument(
            "--azimuth",
            type=float,
            default=315.0,
            help="Sun azimuth (degrees, default: 315, used by Hillshade)"
        )
        
        parser.add_argument(
            "--altitude",
            type=float,
            default=45.0,
            help="Sun altitude (degrees, default: 45, used by Hillshade)"
        )

        parser.add_argument(
            "--z-factor",
            type=float,
            default=1.0,
            help="Hillshade vertical exaggeration (default: 1.0)"
        )
        
        parser.add_argument(
            "--color-mode",
            choices=["warm", "cool", "grayscale"],
            default="warm",
            help="Color mode (default: warm, used by Hillshade)"
        )
        
        parser.add_argument(
            "--cog-only",
            action="store_true",
            help="Only generate a COG from existing tiles"
        )
        parser.add_argument(
            "--cog-backend",
            choices=["internal", "external", "auto"],
            default="internal",
            help="COG generation backend (default: internal)"
        )
        parser.add_argument(
            "--gdal-bin-dir",
            type=str,
            default=None,
            help="External GDAL bin directory (e.g. C:\\Program Files\\GDAL)"
        )

        # Experimental algorithms
        parser.add_argument(
            "--surprise-scales",
            type=str,
            help="Scale-Space Surprise scales, comma-separated (e.g. 1,2,4,8,16)"
        )
        parser.add_argument(
            "--surprise-enhancement",
            type=float,
            default=2.0,
            help="Scale-Space Surprise enhancement factor (default: 2.0)"
        )
        parser.add_argument(
            "--ml-azimuths",
            type=str,
            help="Multi-light azimuths, comma-separated (e.g. 315,45,135,225)"
        )
        parser.add_argument(
            "--uncertainty-weight",
            type=float,
            default=0.7,
            help="Multi-light uncertainty weight (default: 0.7)"
        )
    
    def _validate_platform_args(self, args: argparse.Namespace):
        """Validate Windows-specific arguments."""
        # Special handling for COG-only mode
        if hasattr(args, 'cog_only') and args.cog_only:
            if not os.path.exists(args.tmp_dir):
                self.parser.error(f"--cog-only mode requires a tile directory: {args.tmp_dir}")
            # Flag to skip the input-file check
            args._skip_input_check = True
        
        # Algorithm-specific validation
        if args.algorithm == "rvi" and not hasattr(args, 'no_auto_scale'):
            self.logger.info("When radii are omitted, scales are auto-determined via terrain analysis")
        if args.cog_backend == "external" and not args.gdal_bin_dir:
            self.logger.warning(
                "With --cog-backend external, explicitly specifying --gdal-bin-dir is recommended"
            )
    
    def execute(self, args: argparse.Namespace):
        """Run tile-based processing."""
        # Check configuration
        from ..config.system_config import check_gdal_environment
        check_gdal_environment()
        
        # Prepare parameters
        params = self.get_common_params(args)
        
        # Add Windows-specific parameters
        params.update({
            'tile_size': args.tile_size,
            'padding': args.padding,
            'max_workers': args.max_workers,
            'nodata_threshold': args.nodata_threshold,
            'gpu_type': args.gpu_type,
            'multiscale_mode': not args.single_scale,
            'auto_scale_analysis': not args.no_auto_scale,
            'cog_only': args.cog_only,
            'cog_backend': args.cog_backend,
            'gdal_bin_dir': args.gdal_bin_dir,
        })
        
        # Log output
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
        
        # Run processing
        try:
            from ..core.tile_processor import process_dem_tiles
            
            # In COG-only mode, adjust the input path
            if args.cog_only:
                params['input_path'] = params['input_path'] if os.path.exists(params['input_path']) else "dummy_input.tif"

            radii_list = None
            weights_list = None
            if getattr(args, "radii", None):
                try:
                    radii_list = [int(v.strip()) for v in args.radii.split(",") if v.strip()]
                except ValueError:
                    self.parser.error("Invalid --radii format. Use comma-separated integers: 4,16,64")
            if getattr(args, "weights", None):
                try:
                    weights_list = [float(v.strip()) for v in args.weights.split(",") if v.strip()]
                except ValueError:
                    self.parser.error("Invalid --weights format. Use comma-separated numbers: 0.5,0.3,0.2")
            
            # Prepare algorithm-specific parameters
            algo_params = {}
            if args.algorithm == "hillshade":
                algo_params.update({
                    'azimuth': args.azimuth,
                    'altitude': args.altitude,
                    'color_mode': args.color_mode,
                    'z_factor': args.z_factor,
                })
            elif args.algorithm == "scale_space_surprise":
                if args.surprise_scales:
                    algo_params['scales'] = [float(s.strip()) for s in args.surprise_scales.split(",")]
                algo_params['enhancement'] = args.surprise_enhancement
            elif args.algorithm == "multi_light_uncertainty":
                if args.ml_azimuths:
                    algo_params['azimuths'] = [float(a.strip()) for a in args.ml_azimuths.split(",")]
                algo_params['altitude'] = args.altitude
                algo_params['uncertainty_weight'] = args.uncertainty_weight

            algo_params['mode'] = args.mode
            if radii_list:
                algo_params['radii'] = radii_list
            if weights_list:
                algo_params['weights'] = weights_list
            
            process_dem_tiles(
                input_cog_path=params['input_path'],
                output_cog_path=params['output_path'],
                tmp_tile_dir=params['tmp_dir'],
                algorithm=params['algorithm'],  # add the algorithm
                tile_size=params['tile_size'],
                padding=params['padding'],
                sigma=10.0,
                max_workers=params['max_workers'],
                nodata_threshold=params['nodata_threshold'],
                nodata_override=self._parse_nodata_override(args.nodata),
                output_dtype=getattr(args, 'output_dtype', 'float32'),
                output_range=self._parse_output_range(getattr(args, 'output_range', None)),
                gpu_type=params['gpu_type'],
                multiscale_mode=params['multiscale_mode'],
                pixel_size=params['pixel_size'],
                auto_scale_analysis=params['auto_scale_analysis'],
                cog_only=params['cog_only'],
                cog_backend=params['cog_backend'],
                gdal_bin_dir=params['gdal_bin_dir'],
                **algo_params  # expand algorithm-specific parameters
            )
            
            self.logger.info("Done")
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise

    @staticmethod
    def _parse_nodata_override(raw_value):
        if raw_value is None:
            return None
        text = str(raw_value).strip().lower()
        if text in {"nan", "+nan", "-nan"}:
            return float("nan")
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid --nodata value: {raw_value}") from exc

    @staticmethod
    def _parse_output_range(raw_value):
        """Parse --output-range 'lo,hi' into a (lo, hi) tuple (or None)."""
        if raw_value is None:
            return None
        try:
            lo_s, hi_s = str(raw_value).split(",")
            lo, hi = float(lo_s), float(hi_s)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid --output-range {raw_value!r}; expected 'lo,hi' (e.g. 0,90)."
            ) from exc
        if not (hi > lo):
            raise ValueError(f"--output-range requires hi > lo, got {raw_value!r}.")
        return (lo, hi)
