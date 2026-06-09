# FujiShaderGPU

GPU-accelerated terrain visualization for **very large DEMs** (digital elevation
models). Turn a height raster into a hillshade, TopoUSM relief, slope, openness,
and more — written straight to a Cloud-Optimized GeoTIFF (COG) you can open in QGIS.

An **NVIDIA GPU (CUDA 12.x) is required** — the pipeline is built on CuPy/CUDA and
does not run on CPU-only machines or non-NVIDIA GPUs (no macOS support).

- **Windows** — tile-based local GPU pipeline
- **Linux** — Dask-CUDA distributed pipeline (built for huge multi-GB rasters)

The algorithms and command-line options are the **same on every platform**; only the
backend differs.

> 🚀 New here? Jump straight to [Quick start](#quick-start).

## Requirements

- **An NVIDIA GPU with CUDA 12.x** (required — there is no CPU fallback)
- **Python 3.10+**
- **GDAL with its Python bindings** (`osgeo`) — installed separately; see
  [Installing GDAL](#installing-gdal) below. This is the step people most often
  get stuck on, so do it **first** and verify it before installing FujiShaderGPU.

## Installing GDAL

FujiShaderGPU needs the **GDAL Python bindings** (the `osgeo` package), and the
binding version must match the **native GDAL** library it is built against.
(`rasterio` ships its own private copy of GDAL, but the pipeline also calls
`osgeo` directly, so the bindings must be installed separately — they coexist
fine with rasterio's bundled copy.)

The most reliable route on **either OS is conda**, which installs the native
library and the matching bindings together:

```bash
conda install -c conda-forge gdal
# (do NOT also `pip install GDAL` in a conda env — conda already provides osgeo)
```

### Linux (Debian/Ubuntu, without conda)

```bash
# 1) native GDAL + headers + the gdal-config helper
sudo apt-get update && sudo apt-get install -y libgdal-dev gdal-bin

# 2) build the Python binding that matches your native GDAL, for the *active*
#    Python interpreter (point the compiler at GDAL's headers):
export CPLUS_INCLUDE_PATH=/usr/include/gdal C_INCLUDE_PATH=/usr/include/gdal
pip install --no-build-isolation "GDAL==$(gdal-config --version)"
```

### Windows (without conda)

Conda (above) is strongly recommended. If you cannot use it:

- **OSGeo4W** (<https://trac.osgeo.org/osgeo4w/>) — install the `gdal` and
  `python3-gdal` packages, then run FujiShaderGPU from the OSGeo4W shell (or add
  its Python to your `PATH`). QGIS users often already have this.
- or a prebuilt **GDAL wheel** matching your Python version and native GDAL.

### Verify GDAL before continuing

```bash
python -c "from osgeo import gdal; print('GDAL', gdal.__version__)"
```

If that prints a version, you are good. Common errors:

- `ModuleNotFoundError: No module named 'osgeo'` — the bindings are not installed
  in the active environment.
- `No module named 'osgeo._gdal'` / `No module named '_gdal'` — the binding was
  built for a *different* Python version (e.g. a distro `python3-gdal` package).
  Rebuild it for your interpreter:
  `pip install --no-build-isolation --force-reinstall --no-cache-dir "GDAL==$(gdal-config --version)"`.

## Install

With GDAL working, install straight from GitHub — pick the line for your platform:

```bash
pip install "FujiShaderGPU[windows] @ git+https://github.com/geoign/FujiShaderGPU.git"   # Windows (tile pipeline)
pip install "FujiShaderGPU[linux] @ git+https://github.com/geoign/FujiShaderGPU.git"      # Linux (Dask-CUDA pipeline)
```

## Quick start

FujiShaderGPU runs best on a COG that already has overviews. Two steps:

```bash
# 1) Prepare your DEM once (any raster -> pipeline-ready COG, fills NoData holes)
fujishadergpu-prepare raw_dem.tif dem.tif

# 2) Run an algorithm (default is TopoUSM Fast, multiscale topographic unsharp masking)
fujishadergpu dem.tif shaded.tif --algorithm hillshade
```

That's it — `shaded.tif` is a COG you can drop into QGIS.

> Step 1 is optional but recommended: skipping it still works, but runs slower and may
> show artifacts at the data edge. Run `prepare` once and reuse the output.

## Algorithms

Choose one with `--algorithm <name>` (default: `topousm_fast`):

```text
topousm_fast   hillshade   slope   specular   atmospheric_scattering   multiscale_terrain
blur   curvature   visual_saliency   npr_edges   ambient_occlusion   openness
fractal_anomaly   scale_space_surprise   multi_light_uncertainty
```

> 📖 What each algorithm does — and how to tune it — is documented on its own page
> (separate algorithm guide).

## Common options

| Option | What it does | Default |
| --- | --- | --- |
| `--algorithm NAME` | Which terrain analysis to run | `topousm_fast` |
| `--mode spatial\|local` | Multi-scale radius integration vs. single-pixel neighborhood (`local` = `radii=[1]`, the simplest output; explicit `--radii` ignored) | `spatial` |
| `--radii 4,16,64` | Scales (pixels) for spatial mode | auto: `2,8,32,128,512,2048` truncated to ≤ DEM short side ÷ 10 (max 2048) |
| `--weights 0.5,0.3,0.2` | Per-radius weights | auto: 2ⁿ profile, near scales weighted most, normalized to 1 |
| `--output-dtype float32\|int16\|uint8` | Smaller files for viewing (`int16`≈½, `uint8`≈¼) | `float32` |
| `--nodata VALUE` | Treat this value as NoData (e.g. `-9999`, `nan`) | declared |
| `--pixel-size METERS` | Override pixel size | auto-detected |
| `--force` | Overwrite the output if it exists | off |

Run `fujishadergpu --help` for the complete list.

## More examples

```bash
# Hillshade
fujishadergpu dem.tif out.tif --algorithm hillshade

# Spatial (multi-scale) slope with explicit radii + weights
fujishadergpu dem.tif out.tif --algorithm slope --mode spatial --radii 4,16,64 --weights 0.5,0.3,0.2

# TopoUSM Fast with custom radii
fujishadergpu dem.tif out.tif --algorithm topousm_fast --radii 4,16,64,256

# Compact uint8 COG (~1/4 size) for visualization
fujishadergpu dem.tif out.tif --algorithm topousm_fast --output-dtype uint8

# Zarr output (Linux / Dask path)
fujishadergpu dem.tif out.zarr --algorithm scale_space_surprise
```

## Preparing input

`fujishadergpu-prepare` converts any GDAL-readable raster into a pipeline-ready
float32 COG (ZSTD + internal overviews, no reprojection) and optionally fills NoData
voids — done once, up front:

```bash
fujishadergpu-prepare raw.tif dem.tif                 # auto-detect NoData, fill interior holes
fujishadergpu-prepare raw.tif dem.tif --fill-mode all # fill every hole (dense output, no NoData)
fujishadergpu-prepare raw.tif dem.tif --nodata -9999  # force a specific NoData value
```

Fill modes (`--fill-mode`): `enclosed` *(default — interior holes only)* ·
`none` *(keep all NoData)* · `all` *(fill everything)*.

NoData is auto-detected by default (a declared value, a lost `0` / `-9999` sentinel,
or a value dominating the border). This matters because a stray fill value read as
real terrain skews contrast and creates a halo at the data edge.
→ full rules in [ARCHITECTURE.md §13](ARCHITECTURE.md).

## Output formats

- **COG** by default (`.tif`); **Zarr** when the output path ends in `.zarr`
  (Linux / Dask path).
- `--output-dtype` keeps the GPU math in float32 and only changes the final encoding.
  `int16` / `uint8` shrink the file (NoData = `0`) and record GDAL scale/offset so the
  physical value is recoverable. → details in [ARCHITECTURE.md §14](ARCHITECTURE.md).

## Spatial mode in a nutshell

Most algorithms support `--mode spatial`, which integrates the result over several
radii instead of just adjacent pixels. Pass `--radii` / `--weights` yourself, or omit
them and FujiShaderGPU picks sensible defaults from the detected pixel size. (Requested
on a very small raster, it falls back to `local` mode with a warning.)

## Good to know

- **Geographic (lat/lon) DEMs** are supported in approximation mode (center-latitude
  meter conversion). Reproject for best accuracy on wide-area / high-latitude data.
- **Windows + cloud drives:** reading/writing directly on an rclone/FUSE mount
  (e.g. Cloudflare R2) is supported, though a local disk is faster for large writes.
- The output is **QGIS-optimized** (512×512 blocks, multi-level overviews, ZSTD).

## Development

```bash
pip install -e ".[dev]"
python -m compileall FujiShaderGPU tests
ruff check FujiShaderGPU tests
pytest -q -o addopts='' tests
```

Architecture, design rationale, and per-feature specifications live in
**[ARCHITECTURE.md](ARCHITECTURE.md)**.
