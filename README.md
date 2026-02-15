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

`windows` extra now includes `dask[array]` because some tile-path algorithms are executed through the shared Dask bridge.

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

## Input / Output

- Input:
  - COG (`.tif`, `.tiff`) on all paths
  - Zarr (`.zarr`) on Dask path
- Output:
  - COG by default
  - Zarr when output path ends with `.zarr` (Dask path)

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
- `lrm`
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
- `lrm`
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

- `FujiShaderGPU/algorithms/`
  - `dask_registry.py`
  - `dask_shared.py` (re-export hub, 177 lines)
  - `_base.py`, `_nan_utils.py`, `_global_stats.py`, `_normalization.py` (shared utilities)
  - `_impl_*.py` (14 algorithm implementation modules)
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
