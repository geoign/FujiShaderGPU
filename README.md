# FujiShaderGPU

FujiShaderGPU is a GPU-accelerated terrain visualization/analysis toolkit for large DEM datasets.

- Linux: Dask-CUDA distributed pipeline
- Windows/macOS: tile-based local GPU pipeline
- Output: COG and Zarr

See architecture details in `ARCHITECTURE.md`.

## Installation

Base install:

```bash
pip install -e .
```

Linux Dask-CUDA runtime:

```bash
pip install -e ".[linux]"
```

Windows/macOS tile runtime:

```bash
pip install -e ".[windows]"
```

The `windows` extra includes `dask[array]` because some tile-path algorithms are executed through the shared Dask bridge.

Development tools:

```bash
pip install -e ".[dev]"
```

## Run

Entry point:

```bash
python -m FujiShaderGPU --help
```

Installed script:

```bash
fujishadergpu --help
```

## Preprocessing (recommended first step)

The main pipeline assumes an **overview-bearing COG** input. Convert arbitrary
rasters (and optionally fill NoData voids) once, up front, with the preprocessing
command:

```bash
python -m FujiShaderGPU.prepare input.(tif|img|vrt|...) output_cog.tif
# console script: fujishadergpu-prepare input.tif output_cog.tif
```

Output is a single-band float32 COG (ZSTD + internal overviews), pipeline-compatible.
CRS / pixel grid are preserved (no reprojection); band 1 is used.

NoData fill modes (`--fill-mode`):

- `none` — no filling; NoData preserved.
- `enclosed` *(default)* — fill only interior voids; border-connected NoData
  (ocean / dataset exterior) is kept.
- `all` — fill every NoData cell (including exterior) and remove NoData entirely
  (dense output; useful for 3D model generation).

Filling is done on a coarse overview grid and upsampled (low-frequency), so the
cost is nearly independent of the full raster size and streams within bounded
memory. When the pipeline is given an input without overviews it still runs but
warns and points to this command (decimated reads fall back to slow full-res reads).

NoData detection / override (all converted to float NaN **before** filling):

- The raster's **declared** NoData is always honored (masked I/O).
- `--nodata VALUE` — treat an explicit sentinel as NoData even when the raster
  declares none (or declares a different value). Accepts `-9999`, `0`, `nan`, …
- **Undeclared NoData auto-detection is ON by default**: a dominant constant
  border (a NoData frame whose tag was lost in conversion) is detected and
  converted to NaN, preventing it from being treated as real terrain (which would
  otherwise produce a halo around the data edge). Disable with `--no-detect-nodata`;
  tune sensitivity with `--nodata-border-fraction` (default `0.5` = the value must
  occupy ≥ 50% of the raster's outer ring).

```bash
# Explicitly declare a lost -9999 sentinel while building the COG
python -m FujiShaderGPU.prepare raw_dem.tif kyoto_cog.tif --nodata -9999 --force
```

The **main pipeline** also accepts `--nodata VALUE` (both backends): the value is
replaced with NaN at load time, before any algorithm runs.

```bash
python -m FujiShaderGPU.prepare raw_dem.tif kyoto_cog.tif --fill-mode enclosed --force
fujishadergpu kyoto_cog.tif kyoto_rvi.tif --algorithm rvi --radii 4,32,256,2048
```

## Input / Output

- Input:
  - COG (`.tif`, `.tiff`) on all paths — overview-bearing COG expected (see Preprocessing)
  - Zarr (`.zarr`) on Dask path
- Output:
  - COG by default
  - Zarr when output path ends with `.zarr` (Dask path)

## Output data type (compact integer COGs)

By default every algorithm is computed and written as **float32** (NoData = NaN).
For delivery you can quantize the result to a compact integer COG — useful for
faster COG builds (less disk I/O), smaller object-storage transfers, and lighter
QGIS reads:

```bash
# 1/4-size uint8 COG for visualization
fujishadergpu in.tif out_rvi.tif --algorithm rvi --output-dtype uint8

# 1/2-size int16 (more tonal levels), explicit range
fujishadergpu in.tif out_slope.tif --algorithm slope --output-dtype int16 --output-range 0,90
```

- `--output-dtype {float32,int16,uint8}` (default `float32`, unchanged behavior).
- **NoData = 0** for both integer types; valid data is stretched to fill the
  remaining codes for maximum tonal resolution. Normalized algorithms (RVI,
  fractal_anomaly, visual_saliency, scale_space_surprise, multiscale_terrain) map
  their robust **p99** value to display magnitude `1` via an overview pre-pass, so
  float32 lands in `-1..1` (signed) / `0..1` (unsigned); physical maps keep their
  native range (slope `0..90`, hillshade/AO/openness `0..1`). int16/uint8 reserve
  a little headroom (value `±1` → ~85% of the code range) so the unclipped tail is
  preserved. Override with `--output-range lo,hi`.
- Signed outputs (RVI / fractal_anomaly): `int16` uses the full symmetric
  `[-32767, +32767]` (DN 0 = value ~0 = flat ground, doubling as NoData — visually
  negligible); `uint8` centers value 0 at `128`.
- GDAL `scale`/`offset` are recorded so the physical value is recoverable
  (`value = scale·DN + offset`); QGIS shows scaled values and treats `0` as nodata.
- Compute stays float32 on the GPU — only the final encoding changes — so the math
  and its accuracy are identical to the float32 output. Available on both backends.

## Algorithms (Dask Registry)

Current Dask algorithms:

- `rvi`
- `hillshade`
- `slope`
- `specular`
- `atmospheric_scattering`
- `multiscale_terrain`
- `curvature`
- `visual_saliency`
- `npr_edges`
- `ambient_occlusion`
- `openness`
- `fractal_anomaly`
- `scale_space_surprise`
- `multi_light_uncertainty`

Windows/macOS tile path uses the same canonical algorithm names as Dask:

- `rvi`
- `hillshade`
- `slope`
- `specular`
- `atmospheric_scattering`
- `multiscale_terrain`
- `curvature`
- `visual_saliency`
- `npr_edges`
- `ambient_occlusion`
- `openness`
- `fractal_anomaly`
- `scale_space_surprise`
- `multi_light_uncertainty`

The tile backend calls tile-native modules where available and uses a Dask-shared bridge for the remaining algorithms.

Current list can always be checked with:

```bash
python -m FujiShaderGPU --help
```

## Examples

Default run:

```bash
fujishadergpu input.tif output.tif
```

Run hillshade:

```bash
fujishadergpu input.tif output.tif --algorithm hillshade
```

Run spatial-mode hillshade with common radii/weights parameters:

```bash
fujishadergpu input.tif output.tif --algorithm hillshade --mode spatial --radii 2,4,8,16 --weights 0.4,0.3,0.2,0.1
```

Run spatial-mode slope (radius-integrated):

```bash
fujishadergpu input.tif output.tif --algorithm slope --mode spatial --radii 4,16,64 --weights 0.5,0.3,0.2
```

Run spatial-mode curvature (radius-integrated):

```bash
fujishadergpu input.tif output.tif --algorithm curvature --mode spatial --radii 4,16,64 --weights 0.5,0.3,0.2
```

Dask/Tile path RVI with explicit radii:

```bash
fujishadergpu input.tif output.tif --algorithm rvi --radii 4,16,64 --weights 0.5,0.3,0.2
```

Compact integer output (uint8, ~1/4 size) for visualization:

```bash
fujishadergpu input.tif output.tif --algorithm rvi --output-dtype uint8
```

int16 output with an explicit quantization range:

```bash
fujishadergpu input.tif output.tif --algorithm slope --output-dtype int16 --output-range 0,90
```

Declare a lost NoData sentinel at load time (both backends):

```bash
fujishadergpu input.tif output.tif --algorithm rvi --nodata -9999
```

RVI radii behavior:

- `--radii` is interpreted in pixels.
- `--weights` is optional; if omitted, uniform weighting is used.
- If `--radii` is omitted, RVI auto-derives radii from terrain characteristics.

Spatial mode auto-preset behavior (all spatial-enabled algorithms):

- If `--mode spatial` and `--radii` is omitted, radii/weights are loaded from `FujiShaderGPU/config/spatial_presets.yaml`.
- Preset selection is based on detected pixel size (meters), for both projected and geographic DEMs.
- If user supplies `--radii`, that explicit value has priority.
- If user supplies `--weights`, it is used only when length matches `--radii`; otherwise fallback weighting is used.
- Safety fallback: when `--mode spatial` is requested with no explicit `--radii/--weights` and input DEM has any side `<= 1024 px`, processing falls back to `--mode local` with a warning.

Dask path with Zarr output:

```bash
fujishadergpu input.tif output.zarr --algorithm scale_space_surprise
```

## Development Checks

```bash
python -m compileall FujiShaderGPU tests
ruff check FujiShaderGPU tests
pytest -q -o addopts='' tests
python -m pip check
```

## Repository Structure (Summary)

- `FujiShaderGPU/prepare.py` (preprocessing CLI: any raster -> pipeline-ready COG)
- `FujiShaderGPU/algorithms/`
  - `dask_registry.py`
  - `dask_shared.py` (re-export hub)
  - `_base.py`, `_nan_utils.py`, `_global_stats.py`, `_normalization.py` (shared utilities)
  - `_impl_*.py` (per-algorithm implementation modules)
  - `tile_shared.py`
  - `common/kernels.py`
  - `common/auto_params.py`
  - `dask/*.py`
  - `tile/*.py`
- `FujiShaderGPU/core/`
  - `dask_processor.py`
  - `dask_cluster.py`
  - `dask_io.py`
  - `tile_processor.py`
  - `tile_io.py`
  - `tile_compute.py`
- `FujiShaderGPU/io/`
  - `dem_preprocess.py` (preprocessing core: COG-ification + overview-based NoData fill + undeclared-NoData detection)
  - `output_encoding.py` (output dtype quantization: float32/int16/uint8 ranges, scale/offset, NoData policy)
  - `cog_builder.py`, `cog_validator.py`, `raster_info.py`

## Notes

- Old monolithic algorithm compatibility layers were removed.
- Legacy algorithm aliases (`rvi_gaussian`, `composite_terrain`) were removed.
- The modular `algorithms/dask/*.py` and `algorithms/tile/*.py` layout is canonical.
- Geographic CRS DEM input is supported in approximation mode:
  - center-latitude based meter conversion computes anisotropic `dx/dy` scales
  - these scales are injected into algorithms via `pixel_scale_x/pixel_scale_y`
  - this is a practical approximation for simple use; wide-area/high-latitude data may need reprojection for best accuracy
- Spatial auto presets are centralized in `FujiShaderGPU/config/spatial_presets.yaml`:
  - pixel-size bins: `<5`, `5~25`, `25~50`, `50~250`, `250~1250`, `1250~5000`, `>5000` (meters)
  - each bin defines default `radii` and `weights`
- Local/spatial mode is unified across these algorithms:
  - `hillshade`
  - `slope`
  - `specular`
  - `atmospheric_scattering`
  - `curvature`
  - `ambient_occlusion`
  - `openness`
  - `multi_light_uncertainty`
  - `--mode local` (adjacent-pixel computation)
  - `--mode spatial --radii ... --weights ...` (multi-radius spatial integration)
