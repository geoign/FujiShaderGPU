"""
FujiShaderGPU/cli/args.py

Declarative CLI argument schema shared by the Linux (Dask-CUDA) and Windows
(tile) front-ends.

Every algorithm-facing option is defined exactly **once** here, so both platforms
expose an identical command surface.  Each platform CLI only adds its own
backend-specific options (Dask cluster knobs vs. tile/COG knobs).  The same
helpers parse comma-separated list values and assemble the per-algorithm
parameter dict for both backends, so there is a single source of truth for the
CLI <-> algorithm mapping.

Argument specs are ``(flags, kwargs)`` pairs passed straight to
``argparse.ArgumentParser.add_argument``.
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

ArgSpec = Tuple[Tuple[str, ...], dict]


# ---------------------------------------------------------------------------
# Shared arguments (identical on every platform)
# ---------------------------------------------------------------------------
# Output encoding, NoData, and the unified spatial controls.
PIPELINE_ARGS: List[ArgSpec] = [
    (("--nodata",), dict(
        type=str, default=None,
        help="Explicit NoData value (e.g. -9999, 0, nan); replaced with float NaN before processing")),
    (("--output-dtype",), dict(
        choices=["float32", "int16", "uint8"], default="float32",
        help="Output data type. int16/uint8 quantize for visualization (NoData=0) and shrink the COG (default: float32)")),
    (("--output-range",), dict(
        type=str, default=None,
        help="Explicit quantization range 'lo,hi' (e.g. 0,90); defaults to the algorithm's native range")),
    (("--mode",), dict(
        choices=["local", "spatial"], default="spatial",
        help="Compute mode: spatial (multi-radius integration, default; auto radii from the DEM "
             "short side when --radii is omitted) / local (single-pixel neighborhood = radii=[1], "
             "weights=[1]; explicit --radii ignored). fractal_anomaly/scale_space_surprise/"
             "visual_saliency need multiple scales, so local falls back to spatial with a warning")),
    (("--radii",), dict(
        type=str,
        help="Explicit spatial radii (px), e.g. 4,16,64; when omitted, YAML is auto-selected by pixel size")),
    (("--weights",), dict(
        type=str,
        help="Spatial weights (e.g. 0.5,0.3,0.2); when omitted, YAML/equal weights are applied")),
    (("--agg",), dict(
        choices=["mean", "min", "max", "sum", "stack"], default="mean",
        help="Aggregation method across scales (default: mean)")),
]

# Per-algorithm tuning knobs.
ALGORITHM_ARGS: List[ArgSpec] = [
    # Hillshade / multi-light
    (("--azimuth",), dict(type=float, default=315.0, help="Sun azimuth (degrees, default: 315)")),
    (("--altitude",), dict(type=float, default=45.0, help="Sun altitude (degrees, default: 45)")),
    (("--z-factor",), dict(type=float, default=1.0, help="Vertical exaggeration (default: 1.0)")),
    (("--multiscale",), dict(action="store_true", help="Run multiscale Hillshade")),
    # Slope
    (("--unit",), dict(choices=["degree", "percent", "radians"], default="degree", help="Slope unit (default: degree)")),
    # Curvature
    (("--curvature-type",), dict(choices=["mean", "gaussian", "planform", "profile"], default="mean", help="Curvature type (default: mean)")),
    # Openness / Ambient Occlusion (shared analysis radius)
    (("--radius",), dict(type=int, default=10, help="Analysis radius (pixels, default: 10; Openness/Ambient Occlusion)")),
    (("--openness-type",), dict(choices=["positive", "negative"], default="positive", help="Openness type (default: positive)")),
    (("--num-directions",), dict(type=int, default=16, help="Number of search directions (default: 16)")),
    (("--max-distance",), dict(type=int, default=50, help="Maximum search distance (pixels, default: 50)")),
    (("--num-samples",), dict(type=int, default=16, help="Number of samples for Ambient Occlusion (default: 16)")),
    # Specular
    (("--roughness-scale",), dict(type=float, default=20.0, help="Scale for roughness computation (default: 20.0)")),
    (("--shininess",), dict(type=float, default=10.0, help="Gloss strength (default: 10.0)")),
    (("--light-azimuth",), dict(type=float, default=315.0, help="Light azimuth (degrees, default: 315)")),
    (("--light-altitude",), dict(type=float, default=45.0, help="Light altitude (degrees, default: 45)")),
    # Atmospheric scattering
    (("--scattering-strength",), dict(type=float, default=0.5, help="Atmospheric scattering strength (default: 0.5)")),
    # Multiscale terrain
    (("--scales",), dict(type=str, help="Multiscale terrain scales, comma-separated (e.g. 1,10,50,100)")),
    (("--mst-weights",), dict(type=str, help="Multiscale terrain weights, comma-separated (e.g. 0.4,0.3,0.2,0.1)")),
    # Blur (raw smoothed-elevation output)
    (("--blur-radius",), dict(type=float, default=16.0, help="Blur Gaussian sigma in pixels (default: 16.0; --radii first value overrides it)")),
    # Visual saliency
    (("--vs-scales",), dict(type=str, help="Visual saliency scales, comma-separated (e.g. 2,4,8,16)")),
    (("--global-stats",), dict(
        action=argparse.BooleanOptionalAction, default=True, dest="use_global_stats",
        help="Use global statistics for algorithms that support them (default: enabled)")),
    (("--downsample-factor",), dict(type=int, default=20, help="Downsample factor (default: 20)")),
    # NPR edges
    (("--edge-sigma",), dict(type=float, default=1.0, help="Edge-detection blur strength (default: 1.0)")),
    (("--threshold-low",), dict(type=float, default=0.2, help="Edge-detection lower threshold (default: 0.2)")),
    (("--threshold-high",), dict(type=float, default=0.5, help="Edge-detection upper threshold (default: 0.5)")),
    # Fractal anomaly
    (("--fractal-radii",), dict(type=str, help="Fractal anomaly radii, comma-separated (e.g. 2,4,8,16,32)")),
    # Scale-space surprise
    (("--surprise-scales",), dict(type=str, help="Scale-Space Surprise scales, comma-separated (e.g. 1,2,4,8,16)")),
    (("--surprise-enhancement",), dict(type=float, default=2.0, help="Scale-Space Surprise enhancement factor (default: 2.0)")),
    # Multi-light uncertainty
    (("--ml-azimuths",), dict(type=str, help="Multi-light azimuths, comma-separated (e.g. 315,45,135,225)")),
    (("--uncertainty-weight",), dict(type=float, default=0.7, help="Multi-light uncertainty weight (default: 0.7)")),
    # Generic
    (("--intensity",), dict(type=float, default=1.0, help="Effect intensity (default: 1.0, used by several algorithms)")),
]

SHARED_ARGS: List[ArgSpec] = PIPELINE_ARGS + ALGORITHM_ARGS


# ---------------------------------------------------------------------------
# Platform-specific arguments
# ---------------------------------------------------------------------------
DASK_ARGS: List[ArgSpec] = [
    (("--chunk",), dict(type=int, help="Chunk width (px); auto-determined when omitted")),
    (("--memory-fraction",), dict(
        type=float, default=None,
        help="GPU device-memory fraction. Default: auto (VRAM-aware, ~0.60-0.85 via auto_tune). Pass an explicit value to override.")),
    (("--verbose",), dict(action="store_true", help="Enable verbose logging")),
    (("--auto-radii",), dict(
        action=argparse.BooleanOptionalAction, default=True,
        help="Auto-determine radii via terrain analysis (TopoUSM Fast only, default: enabled)")),
]

TILE_ARGS: List[ArgSpec] = [
    (("--tile-size",), dict(type=int, help="Tile size (auto-detected when omitted)")),
    (("--padding",), dict(type=int, help="Tile-boundary padding (auto-computed when omitted)")),
    (("--max-workers",), dict(type=int, help="Number of parallel workers (auto-detected when omitted)")),
    (("--nodata-threshold",), dict(type=float, default=1.0, help="NoData skip threshold (default: 1.0)")),
    (("--single-scale",), dict(action="store_true", help="Force single-scale mode")),
    (("--cog-only",), dict(action="store_true", help="Only generate a COG from existing tiles")),
    (("--cog-backend",), dict(choices=["internal", "external", "auto"], default="internal", help="COG generation backend (default: internal)")),
    (("--gdal-bin-dir",), dict(type=str, default=None, help="External GDAL bin directory (e.g. C:\\Program Files\\GDAL)")),
]


# ---------------------------------------------------------------------------
# Comma-separated list fields: arg dest -> (parsed-list attribute, element type)
# ---------------------------------------------------------------------------
LIST_FIELDS = {
    "radii": ("radii_list", int),
    "weights": ("weights_list", float),
    "scales": ("scales_list", float),
    "mst_weights": ("mst_weights_list", float),
    "vs_scales": ("vs_scales_list", float),
    "fractal_radii": ("fractal_radii_list", int),
    "surprise_scales": ("surprise_scales_list", float),
    "ml_azimuths": ("ml_azimuths_list", float),
}


def add_arguments(parser: argparse.ArgumentParser, specs: List[ArgSpec]) -> None:
    """Add a list of ``(flags, kwargs)`` argument specs to ``parser``."""
    for flags, kwargs in specs:
        parser.add_argument(*flags, **kwargs)


def parse_list_fields(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Populate the ``*_list`` attributes from each comma-separated string field.

    A missing/empty field yields ``None``.  Invalid content calls ``parser.error``
    with a uniform message.  Runs for both platforms so list parsing is identical.
    """
    for dest, (attr, elem_type) in LIST_FIELDS.items():
        raw = getattr(args, dest, None)
        if raw:
            try:
                setattr(args, attr, [elem_type(v.strip()) for v in str(raw).split(",") if v.strip()])
            except (ValueError, TypeError):
                kind = "integers" if elem_type is int else "numbers"
                parser.error(f"Invalid --{dest.replace('_', '-')} format. Provide comma-separated {kind}.")
        else:
            setattr(args, attr, None)


def parse_nodata_override(raw_value) -> Optional[float]:
    """Parse ``--nodata`` into a float (or None). 'nan' is accepted."""
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid --nodata value: {raw_value}") from exc


def parse_output_range(raw_value) -> Optional[Tuple[float, float]]:
    """Parse ``--output-range`` 'lo,hi' into a (lo, hi) tuple (or None)."""
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


def build_algo_params(args: argparse.Namespace) -> dict:
    """Assemble the per-algorithm parameter dict from parsed args.

    Single source of truth for the CLI -> algorithm mapping, shared by both
    backends.  Returns only *algorithm-level* params (mode/radii/weights/agg plus
    each algorithm's own knobs); pipeline-level options (pixel_size, nodata,
    output dtype/range, chunk, memory fraction) are handled by each ``execute``.

    Spatial radii/weights are emitted for *every* algorithm (including TopoUSM Fast): the
    Dask backend routes them through ``run_pipeline``'s ``radii`` parameter and the
    tile backend reads them straight from the algorithm params, so one rule serves
    both.  Platform-divergent extras (e.g. ``verbose`` on Dask) are included only
    when present, via ``hasattr``.
    """
    algorithm = args.algorithm
    p: dict = {}

    # Universal controls.
    p["mode"] = getattr(args, "mode", "spatial")
    p["agg"] = getattr(args, "agg", "mean")
    p["intensity"] = getattr(args, "intensity", 1.0)
    if getattr(args, "radii_list", None):
        p["radii"] = args.radii_list
    if getattr(args, "weights_list", None):
        p["weights"] = args.weights_list
    if hasattr(args, "verbose"):
        p["verbose"] = args.verbose

    if algorithm == "hillshade":
        p["azimuth"] = args.azimuth
        p["altitude"] = args.altitude
        p["z_factor"] = args.z_factor
        p["multiscale"] = args.multiscale

    elif algorithm == "slope":
        p["unit"] = args.unit

    elif algorithm == "curvature":
        p["curvature_type"] = args.curvature_type

    elif algorithm == "openness":
        p["radius"] = args.radius
        p["openness_type"] = args.openness_type
        p["num_directions"] = args.num_directions
        p["max_distance"] = args.max_distance

    elif algorithm == "ambient_occlusion":
        p["num_samples"] = args.num_samples
        p["radius"] = args.radius

    elif algorithm == "specular":
        p["roughness_scale"] = args.roughness_scale
        p["shininess"] = args.shininess
        p["light_azimuth"] = args.light_azimuth
        p["light_altitude"] = args.light_altitude

    elif algorithm == "atmospheric_scattering":
        p["scattering_strength"] = args.scattering_strength

    elif algorithm == "multiscale_terrain":
        if getattr(args, "scales_list", None):
            p["scales"] = args.scales_list
        if getattr(args, "mst_weights_list", None):
            p["weights"] = args.mst_weights_list

    elif algorithm == "blur":
        # Single Gaussian sigma; the algorithm's _resolve_radius prefers the
        # unified --radii first value over this when radii are supplied.
        p["radius"] = args.blur_radius

    elif algorithm == "visual_saliency":
        if getattr(args, "vs_scales_list", None):
            p["scales"] = args.vs_scales_list
        p["use_global_stats"] = bool(getattr(args, "use_global_stats", True))
        p["downsample_factor"] = args.downsample_factor

    elif algorithm == "npr_edges":
        p["edge_sigma"] = args.edge_sigma
        p["threshold_low"] = args.threshold_low
        p["threshold_high"] = args.threshold_high

    elif algorithm == "fractal_anomaly":
        if getattr(args, "fractal_radii_list", None):
            p["radii"] = args.fractal_radii_list

    elif algorithm == "scale_space_surprise":
        if getattr(args, "surprise_scales_list", None):
            p["scales"] = args.surprise_scales_list
        p["enhancement"] = args.surprise_enhancement

    elif algorithm == "multi_light_uncertainty":
        if getattr(args, "ml_azimuths_list", None):
            p["azimuths"] = args.ml_azimuths_list
        p["altitude"] = args.altitude
        p["z_factor"] = args.z_factor
        p["uncertainty_weight"] = args.uncertainty_weight

    return p


__all__ = [
    "PIPELINE_ARGS",
    "ALGORITHM_ARGS",
    "SHARED_ARGS",
    "DASK_ARGS",
    "TILE_ARGS",
    "LIST_FIELDS",
    "add_arguments",
    "parse_list_fields",
    "parse_nodata_override",
    "parse_output_range",
    "build_algo_params",
]
