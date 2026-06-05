"""
FujiShaderGPU/cli/linux_cli.py
CLI for Linux - Dask-CUDA processing implementation.
"""
from typing import List, Optional
import argparse
import os

import GPUtil
import numpy as np
import rasterio
from .base import BaseCLI
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
        """Add Linux-specific arguments."""
        # Dask processing options
        parser.add_argument(
            "--chunk",
            type=int,
            help="Chunk width (px); auto-determined when omitted"
        )
        
        parser.add_argument(
            "--memory-fraction",
            type=float,
            default=0.4,  # changed to a more conservative default
            help="GPU memory fraction (default: 0.4)"  # help text also updated
        )

        parser.add_argument(
            "--nodata",
            type=str,
            default=None,
            help="Explicit NoData value (e.g. -9999, 0, nan); replaced with float NaN before processing"
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

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        # Multiscale processing
        parser.add_argument(
            "--agg",
            choices=["mean", "min", "max", "sum", "stack"],
            default="mean",
            help="Aggregation method across scales (default: mean)"
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

        parser.add_argument(
            "--auto-radii",
            action="store_true",
            default=True,
            help="Auto-determine radii via terrain analysis (RVI only, default: True)"
        )
        
        parser.add_argument(
            "--no-auto-radii",
            action="store_true",
            help="Disable automatic radii determination (RVI only)"
        )
        
        # Hillshade-specific
        parser.add_argument(
            "--azimuth",
            type=float,
            default=315,
            help="Sun azimuth (degrees, default: 315)"
        )
        
        parser.add_argument(
            "--altitude",
            type=float,
            default=45,
            help="Sun altitude (degrees, default: 45)"
        )
        
        parser.add_argument(
            "--z-factor",
            type=float,
            default=1.0,
            help="Vertical exaggeration (default: 1.0)"
        )
        
        parser.add_argument(
            "--multiscale",
            action="store_true",
            help="Run multiscale Hillshade"
        )

        # Slope-specific
        parser.add_argument(
            "--unit",
            choices=["degree", "percent", "radians"],
            default="degree",
            help="Slope unit (default: degree)"
        )
        
        # Curvature-specific
        parser.add_argument(
            "--curvature-type",
            choices=["mean", "gaussian", "planform", "profile"],
            default="mean",
            help="Curvature type (default: mean)"
        )
        
        # Common to Openness/Ambient Occlusion
        parser.add_argument(
            "--radius",
            type=int,
            default=10,
            help="Analysis radius (pixels, default: 10)"
        )
        
        # LRM-specific
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=25,
            help="Kernel size for trend removal (default: 25)"
        )
        
        # Openness-specific
        parser.add_argument(
            "--openness-type",
            choices=["positive", "negative"],
            default="positive",
            help="Openness type (default: positive)"
        )
        
        parser.add_argument(
            "--num-directions",
            type=int,
            default=16,
            help="Number of search directions (default: 16)"
        )
        
        parser.add_argument(
            "--max-distance",
            type=int,
            default=50,
            help="Maximum search distance (pixels, default: 50)"
        )

        # Specular-specific
        parser.add_argument(
            "--roughness-scale",
            type=float,
            default=20.0,
            help="Scale for roughness computation (default: 20.0)"
        )

        parser.add_argument(
            "--shininess",
            type=float,
            default=10.0,
            help="Gloss strength (default: 10.0)"
        )

        parser.add_argument(
            "--light-azimuth",
            type=float,
            default=315,
            help="Light azimuth (degrees, default: 315)"
        )

        parser.add_argument(
            "--light-altitude",
            type=float,
            default=45,
            help="Light altitude (degrees, default: 45)"
        )

        # Atmospheric Scattering-specific
        parser.add_argument(
            "--scattering-strength",
            type=float,
            default=0.5,
            help="Atmospheric scattering strength (default: 0.5)"
        )

        # Multiscale Terrain-specific
        parser.add_argument(
            "--scales",
            type=str,
            help="Multiscale terrain scales, comma-separated (e.g. 1,10,50,100)"
        )

        parser.add_argument(
            "--mst-weights",  # avoid colliding with weights
            type=str,
            help="Multiscale terrain weights, comma-separated (e.g. 0.4,0.3,0.2,0.1)"
        )

        # Visual Saliency-specific
        parser.add_argument(
            "--vs-scales",  # avoid colliding with scales
            type=str,
            help="Visual saliency scales, comma-separated (e.g. 2,4,8,16)"
        )

        parser.add_argument(
            "--use-global-stats",
            action="store_true",
            default=True,
            help="Use global statistics (default: True)"
        )

        parser.add_argument(
            "--no-global-stats",
            action="store_true",
            help="Disable global statistics"
        )

        parser.add_argument(
            "--downsample-factor",
            type=int,
            default=20,
            help="Downsample factor (default: 20)"
        )

        # NPR Edges-specific
        parser.add_argument(
            "--edge-sigma",
            type=float,
            default=1.0,
            help="Edge-detection blur strength (default: 1.0)"
        )

        parser.add_argument(
            "--threshold-low",
            type=float,
            default=0.2,
            help="Edge-detection lower threshold (default: 0.2)"
        )

        parser.add_argument(
            "--threshold-high",
            type=float,
            default=0.5,
            help="Edge-detection upper threshold (default: 0.5)"
        )

        # Ambient Occlusion-specific (only num-samples added)
        parser.add_argument(
            "--num-samples",
            type=int,
            default=16,
            help="Number of samples for AO (default: 16)"
        )

        # Fractal Anomaly-specific (newly added)
        parser.add_argument(
            "--fractal-radii",
            type=str,
            help="Fractal anomaly radii, comma-separated (e.g. 2,4,8,16,32)"
        )

        parser.add_argument(
            "--auto-fractal-radii",
            action="store_true",
            default=True,
            help="Auto-determine fractal radii from resolution (default: True)"
        )

        parser.add_argument(
            "--no-auto-fractal-radii",
            action="store_true",
            help="Disable automatic fractal radii determination"
        )
        
        # Generic intensity parameter
        parser.add_argument(
            "--intensity",
            type=float,
            default=1.0,
            help="Effect intensity (default: 1.0, used by several algorithms)"
        )

        # Scale-Space Surprise
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

        # Multi-light Uncertainty
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

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse arguments (overrides the base class)."""
        parsed_args = super().parse_args(args)
        
        # Parse radii (with attribute-existence check)
        if hasattr(parsed_args, 'radii') and parsed_args.radii:
            try:
                parsed_args.radii_list = [int(r.strip()) for r in parsed_args.radii.split(",")]
            except ValueError:
                self.parser.error("Invalid radii format. Provide comma-separated integers, e.g. 4,16,64,256")
        else:
            parsed_args.radii_list = None
        
        # Parse weights (with attribute-existence check)
        if hasattr(parsed_args, 'weights') and parsed_args.weights:
            try:
                parsed_args.weights_list = [float(w.strip()) for w in parsed_args.weights.split(",")]
            except ValueError:
                self.parser.error("Invalid weights format. Provide comma-separated numbers, e.g. 0.4,0.3,0.2,0.1")
        else:
            parsed_args.weights_list = None
        
        # Parse scales (for multiscale_terrain)
        if hasattr(parsed_args, 'scales') and parsed_args.scales:
            try:
                parsed_args.scales_list = [float(s.strip()) for s in parsed_args.scales.split(",")]
            except ValueError:
                self.parser.error("Invalid scales format. Provide comma-separated numbers, e.g. 1,10,50,100")
        else:
            parsed_args.scales_list = None

        # Parse mst_weights (for multiscale_terrain)
        if hasattr(parsed_args, 'mst_weights') and parsed_args.mst_weights:
            try:
                parsed_args.mst_weights_list = [float(w.strip()) for w in parsed_args.mst_weights.split(",")]
            except ValueError:
                self.parser.error("Invalid mst-weights format. Provide comma-separated numbers, e.g. 0.4,0.3,0.2,0.1")
        else:
            parsed_args.mst_weights_list = None

        # Parse vs_scales (for visual_saliency)
        if hasattr(parsed_args, 'vs_scales') and parsed_args.vs_scales:
            try:
                parsed_args.vs_scales_list = [float(s.strip()) for s in parsed_args.vs_scales.split(",")]
            except ValueError:
                self.parser.error("Invalid vs-scales format. Provide comma-separated numbers, e.g. 2,4,8,16")
        else:
            parsed_args.vs_scales_list = None

        # Parse fractal_radii (for fractal_anomaly)
        if hasattr(parsed_args, 'fractal_radii') and parsed_args.fractal_radii:
            try:
                parsed_args.fractal_radii_list = [int(r.strip()) for r in parsed_args.fractal_radii.split(",")]
            except ValueError:
                self.parser.error("Invalid fractal-radii format. Provide comma-separated integers, e.g. 2,4,8,16,32")
        else:
            parsed_args.fractal_radii_list = None

        if hasattr(parsed_args, 'surprise_scales') and parsed_args.surprise_scales:
            try:
                parsed_args.surprise_scales_list = [float(s.strip()) for s in parsed_args.surprise_scales.split(",")]
            except ValueError:
                self.parser.error("Invalid surprise-scales format. Provide comma-separated numbers, e.g. 1,2,4,8,16")
        else:
            parsed_args.surprise_scales_list = None

        if hasattr(parsed_args, 'ml_azimuths') and parsed_args.ml_azimuths:
            try:
                parsed_args.ml_azimuths_list = [float(a.strip()) for a in parsed_args.ml_azimuths.split(",")]
            except ValueError:
                self.parser.error("Invalid ml-azimuths format. Provide comma-separated numbers, e.g. 315,45,135,225")
        else:
            parsed_args.ml_azimuths_list = None

        return parsed_args
    
    def _validate_platform_args(self, args: argparse.Namespace):
        # Handle the auto-radii flag
        if hasattr(args, 'auto_radii') and hasattr(args, 'no_auto_radii'):
            args.auto_radii = args.auto_radii and not args.no_auto_radii
        
        # Handle the use_global_stats flag
        if hasattr(args, 'use_global_stats') and hasattr(args, 'no_global_stats'):
            args.use_global_stats = args.use_global_stats and not args.no_global_stats

        # Handle the auto_fractal_radii flag
        if hasattr(args, 'auto_fractal_radii') and hasattr(args, 'no_auto_fractal_radii'):
            args.auto_fractal_radii = args.auto_fractal_radii and not args.no_auto_fractal_radii
        
        # Error when RVI has no radii and auto-determination is also disabled
        if args.algorithm == "rvi":
            if not getattr(args, 'radii', None) and not getattr(args, 'auto_radii', True):
                self.parser.error("Specify radii or enable --auto-radii")

    @staticmethod
    def _parse_nodata_override(raw_value):
        """Parse --nodata into a float (or None). 'nan' is accepted (no-op)."""
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

    def execute(self, args: argparse.Namespace):
        """Run Dask-CUDA processing."""
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"]="0.70"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__SPILL"]="0.75"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"]="0.85"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"]="0.95"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"]="30s"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"]="60s"
        os.environ["DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT"]="60s"

        # Dynamic RMM settings from auto_tune
        from FujiShaderGPU.config.auto_tune import compute_rmm_pool_gb
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory_gb = gpus[0].memoryTotal / 1024
        else:
            gpu_memory_gb = 16  # conservative default
        _rmm_gb = compute_rmm_pool_gb(gpu_memory_gb)
        os.environ["RMM_ALLOCATOR"] = "pool"
        os.environ["RMM_POOL_SIZE"] = f"{_rmm_gb}GB"
        os.environ["RMM_MAXIMUM_POOL_SIZE"] = f"{int(_rmm_gb * 1.1)}GB"

        # Prepare parameters
        params = self.get_common_params(args)

        # Auto-detect pixel_size (when not specified)
        if not hasattr(args, 'pixel_size') or args.pixel_size is None:
            with rasterio.open(params['input_path']) as src:
                # Check the CRS
                if src.crs and src.crs.is_geographic:
                    # Geographic CRS (lat/lon) case
                    # Get the data's center latitude
                    bounds = src.bounds
                    center_lat = (bounds.bottom + bounds.top) / 2
                    
                    # Distance per degree of latitude (meters)
                    # Approximation using Earth's radius
                    lat_rad = np.radians(center_lat)
                    meters_per_degree_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + \
                                        1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
                    meters_per_degree_lon = 111412.84 * np.cos(lat_rad) - \
                                        93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
                    
                    # Convert pixel size from degrees to meters
                    pixel_size_x_deg = abs(src.transform[0])
                    pixel_size_y_deg = abs(src.transform[4])
                    
                    pixel_size_x_m = pixel_size_x_deg * meters_per_degree_lon
                    pixel_size_y_m = pixel_size_y_deg * meters_per_degree_lat
                    
                    # Use the mean (usually nearly identical)
                    args.pixel_size = (pixel_size_x_m + pixel_size_y_m) / 2
                    
                    self.logger.info(f"Geographic CRS detected: center latitude {center_lat:.2f}°")
                    self.logger.info(f"Pixel size: {pixel_size_x_deg:.6f}° x {pixel_size_y_deg:.6f}°")
                    self.logger.info(f"In meters: {args.pixel_size:.2f}m")
                    
                else:
                    # Projected CRS case (already in meters, etc.)
                    pixel_size_x = abs(src.transform[0])
                    pixel_size_y = abs(src.transform[4])
                    
                    # Check the unit (when a CRS is present)
                    if src.crs:
                        units = src.crs.linear_units
                        if units and units.lower() != 'metre' and units.lower() != 'meter':
                            self.logger.warning(f"CRS unit is '{units}'; treating it as meters.")
                    
                    args.pixel_size = (pixel_size_x + pixel_size_y) / 2
                    self.logger.info(f"Projected CRS: pixel size {args.pixel_size:.2f}m")
        else:
            # When the user specified it explicitly
            self.logger.info(f"User-specified pixel size: {args.pixel_size}m")

        # Log output
        self.logger.info("=== Dask-CUDA terrain analysis ===")
        self.logger.info(f"Input: {args.input}")
        self.logger.info(f"Output: {args.output}")
        self.logger.info(f"Algorithm: {args.algorithm}")
        
        # Set default values
        if not hasattr(args, 'auto_radii'):
            args.auto_radii = True
        if not hasattr(args, 'radii'):
            args.radii = None
        if not hasattr(args, 'radii_list'):
            args.radii_list = getattr(args, 'radii_list', None)
        if not hasattr(args, 'weights'):
            args.weights = None
        if not hasattr(args, 'weights_list'):
            args.weights_list = getattr(args, 'weights_list', None)

        # Prepare algorithm-specific parameters
        algo_params = {}
        
        # ... existing algorithm-specific parameter handling ...
        # Common parameters
        if hasattr(args, 'intensity'):
            algo_params['intensity'] = args.intensity
        if hasattr(args, 'pixel_size'):
            algo_params['pixel_size'] = args.pixel_size
        if hasattr(args, 'verbose'):
            algo_params['verbose'] = args.verbose

        # Hillshade-specific
        if args.algorithm == 'hillshade':
            if hasattr(args, 'azimuth'):
                algo_params['azimuth'] = args.azimuth
            if hasattr(args, 'altitude'):
                algo_params['altitude'] = args.altitude
            if hasattr(args, 'z_factor'):
                algo_params['z_factor'] = args.z_factor
            if hasattr(args, 'multiscale'):
                algo_params['multiscale'] = args.multiscale
            if hasattr(args, 'radii_list') and args.radii_list:
                algo_params['radii'] = args.radii_list
            if hasattr(args, 'weights_list') and args.weights_list:
                algo_params['weights'] = args.weights_list

        # Slope-specific
        elif args.algorithm == 'slope':
            if hasattr(args, 'unit'):
                algo_params['unit'] = args.unit

        # Curvature-specific
        elif args.algorithm == 'curvature':
            if hasattr(args, 'curvature_type'):
                algo_params['curvature_type'] = args.curvature_type

        # LRM-specific
        elif args.algorithm == 'lrm':
            if hasattr(args, 'kernel_size'):
                algo_params['kernel_size'] = args.kernel_size

        # Openness-specific
        elif args.algorithm == 'openness':
            if hasattr(args, 'radius'):
                algo_params['radius'] = args.radius
            if hasattr(args, 'openness_type'):
                algo_params['openness_type'] = args.openness_type
            if hasattr(args, 'num_directions'):
                algo_params['num_directions'] = args.num_directions
            if hasattr(args, 'max_distance'):
                algo_params['max_distance'] = args.max_distance

        # Ambient Occlusion-specific
        elif args.algorithm == 'ambient_occlusion':
            if hasattr(args, 'num_samples'):
                algo_params['num_samples'] = args.num_samples
            if hasattr(args, 'radius'):
                algo_params['radius'] = args.radius
        
        # Specular-specific
        elif args.algorithm == 'specular':
            if hasattr(args, 'roughness_scale'):
                algo_params['roughness_scale'] = args.roughness_scale
            if hasattr(args, 'shininess'):
                algo_params['shininess'] = args.shininess
            if hasattr(args, 'light_azimuth'):
                algo_params['light_azimuth'] = args.light_azimuth
            if hasattr(args, 'light_altitude'):
                algo_params['light_altitude'] = args.light_altitude

        # Atmospheric Scattering-specific
        elif args.algorithm == 'atmospheric_scattering':
            if hasattr(args, 'scattering_strength'):
                algo_params['scattering_strength'] = args.scattering_strength

        # Multiscale Terrain-specific
        elif args.algorithm == 'multiscale_terrain':
            if hasattr(args, 'scales_list') and args.scales_list:
                algo_params['scales'] = args.scales_list
            if hasattr(args, 'mst_weights_list') and args.mst_weights_list:
                algo_params['weights'] = args.mst_weights_list

        # Visual Saliency-specific
        elif args.algorithm == 'visual_saliency':
            if hasattr(args, 'vs_scales_list') and args.vs_scales_list:
                algo_params['scales'] = args.vs_scales_list
            if hasattr(args, 'use_global_stats'):
                # Handle no_global_stats
                use_global = args.use_global_stats and not getattr(args, 'no_global_stats', False)
                algo_params['use_global_stats'] = use_global
            if hasattr(args, 'downsample_factor'):
                algo_params['downsample_factor'] = args.downsample_factor

        # NPR Edges-specific
        elif args.algorithm == 'npr_edges':
            if hasattr(args, 'edge_sigma'):
                algo_params['edge_sigma'] = args.edge_sigma
            if hasattr(args, 'threshold_low'):
                algo_params['threshold_low'] = args.threshold_low
            if hasattr(args, 'threshold_high'):
                algo_params['threshold_high'] = args.threshold_high

        # Fractal Anomaly-specific
        elif args.algorithm == 'fractal_anomaly':
            if hasattr(args, 'fractal_radii_list') and args.fractal_radii_list:
                algo_params['radii'] = args.fractal_radii_list
        elif args.algorithm == 'scale_space_surprise':
            if hasattr(args, 'surprise_scales_list') and args.surprise_scales_list:
                algo_params['scales'] = args.surprise_scales_list
            if hasattr(args, 'surprise_enhancement'):
                algo_params['enhancement'] = args.surprise_enhancement
        elif args.algorithm == 'multi_light_uncertainty':
            if hasattr(args, 'ml_azimuths_list') and args.ml_azimuths_list:
                algo_params['azimuths'] = args.ml_azimuths_list
            if hasattr(args, 'altitude'):
                algo_params['altitude'] = args.altitude
            if hasattr(args, 'z_factor'):
                algo_params['z_factor'] = args.z_factor
            if hasattr(args, 'uncertainty_weight'):
                algo_params['uncertainty_weight'] = args.uncertainty_weight

        spatial_mode_algorithms = {
            "rvi",
            "hillshade",
            "slope",
            "specular",
            "atmospheric_scattering",
            "curvature",
            "ambient_occlusion",
            "openness",
            "multi_light_uncertainty",
            # multiscale_terrain uses the unified --radii as its gaussian scales
            # (a radius is a spatial scale here).  Explicit --radii overrides its
            # own --scales/--mst-weights, which still work when --radii is omitted.
            "multiscale_terrain",
        }
        if args.algorithm in spatial_mode_algorithms:
            algo_params['mode'] = getattr(args, 'mode', 'local')
            if args.algorithm != "rvi" and getattr(args, 'radii_list', None):
                algo_params['radii'] = args.radii_list
            if args.algorithm != "rvi" and getattr(args, 'weights_list', None):
                algo_params['weights'] = args.weights_list

        # Run processing
        try:
            from ..core.dask_processor import run_pipeline
            
            # Safe handling of RVI parameters
            if args.algorithm == "rvi":
                rvi_params = {
                    'radii': args.radii_list,  # only when manually specified
                    'weights': args.weights_list,
                    'auto_radii': args.radii_list is None,  # auto-determine when radii are not specified
                }
            else:
                # Non-RVI algorithms
                rvi_params = {}

            run_pipeline(
                src_cog=params['input_path'],
                dst_cog=params['output_path'],
                algorithm=args.algorithm,
                agg=getattr(args, 'agg', 'mean'),
                chunk=getattr(args, 'chunk', None),
                show_progress=params['show_progress'],
                memory_fraction=getattr(args, 'memory_fraction', None),
                nodata_override=self._parse_nodata_override(getattr(args, 'nodata', None)),
                output_dtype=getattr(args, 'output_dtype', 'float32'),
                output_range=self._parse_output_range(getattr(args, 'output_range', None)),
                **rvi_params,
                **algo_params
            )
            
        except Exception as e:
            self.logger.error(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
            raise
        
