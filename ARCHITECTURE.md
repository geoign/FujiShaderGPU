# FujiShaderGPU Architecture

## 1. Purpose
FujiShaderGPU is a Python/CUDA terrain-visualization pipeline for very large DEM rasters.

- Linux path: Dask-CUDA distributed processing for large-scale workloads.
- Windows/macOS path: tile-based local GPU processing without `dask-cuda`.
- Output targets: Cloud Optimized GeoTIFF (COG) and Zarr.

## 2. End-to-End Flow
```text
Input DEM (COG or Zarr)
  -> CLI (platform-specific)
  -> Core orchestration (Dask or Tile)
  -> Algorithm layer (modular per-algorithm files)
  -> Output writer (COG or Zarr)
  -> Validation / logs
```

## 3. Code Topology

### 3.1 Algorithm Layer
`FujiShaderGPU/algorithms/`

- `dask_registry.py`
  - Canonical Dask algorithm registry (`ALGORITHMS`).
  - Used by `core/dask_processor.py`.

- `dask_shared.py`
  - Shared Dask-side algorithm base and internals.

- `tile_shared.py`
  - Shared Tile-side algorithm base and internals.

- `common/kernels.py`
  - Shared CuPy kernels used by both Dask and Tile.

- `common/auto_params.py`
  - Shared auto-parameter helpers (e.g., radii derivation).

- `common/spatial_mode.py`
  - Shared local/spatial execution helpers:
    - spatial radii auto-derivation
    - NaN-aware radius smoothing
    - multi-radius weighted aggregation

- `dask/*.py`
  - One file per Dask algorithm module.

- `tile/*.py`
  - One file per Tile algorithm module.
  - Includes `tile/dask_bridge.py` adapters so canonical Dask algorithm names are callable on tile backend too.

### 3.2 Core Layer
`FujiShaderGPU/core/`

- `dask_processor.py`
  - Dask orchestration layer.
  - Coordinates algorithm execution, parameter flow, and output mode.

- `dask_cluster.py`
  - Dask-CUDA cluster lifecycle and chunk-size strategy.

- `dask_io.py`
  - Dask-side COG/Zarr load and Zarr write helpers.

- `tile_processor.py`
  - Tile orchestration layer.
  - Schedules per-tile work and final build flow.

- `tile_io.py`
  - Tile raster read/write primitives.

- `tile_compute.py`
  - Per-tile algorithm invocation and shared compute helpers.

- `gpu_memory.py`
  - GPU memory pool/context lifecycle helper.

### 3.3 CLI Layer
`FujiShaderGPU/cli/`

- `base.py`
  - Shared CLI scaffold.

- `linux_cli.py`
  - Linux-specific CLI.
  - Supported algorithms are sourced from `algorithms/dask_registry.py`.

- `windows_cli.py`
  - Windows/macOS-specific CLI.
  - Supported algorithms are sourced from `core/tile_processor.py::DEFAULT_ALGORITHMS`.

### 3.4 IO / Config / Utils
- `io/cog_builder.py`, `io/cog_validator.py`, `io/raster_info.py`
- `config/gpu_config_manager.py`, `config/system_config.py`, `config/gdal_config.py`
- `utils/scale_analysis.py`, `utils/nodata_handler.py`, `utils/types.py`

## 4. Dask Algorithm Catalog
Registered in `algorithms/dask_registry.py`:

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

These names are also the canonical CLI names for the tile backend.

## 5. Zarr Support

### 5.1 Input
In Dask path, `.zarr` input is detected and loaded through `xarray.open_zarr(...)`.

### 5.2 Output
If output path ends with `.zarr`, result is written with `xarray.Dataset.to_zarr(...)`.
Otherwise output is written through COG flow.

### 5.3 Runtime Dependency Boundary
`pyproject.toml` is split by runtime mode:

- base dependencies: common minimum
- `optional-dependencies.linux`: Dask-CUDA / `dask[array]` / distributed / xarray / zarr stack
- `optional-dependencies.windows`: tile-path extras (`scipy`, `dask[array]` for bridge-backed algorithms)

## 6. Platform Behavior

### Linux
- Uses `core/dask_processor.py` + `dask_cluster.py` + `dask_io.py`.
- Uses `algorithms/dask_registry.py`.
- Optimized for very large distributed workloads.

### Windows/macOS
- Uses `core/tile_processor.py` + `tile_io.py` + `tile_compute.py`.
- Uses `algorithms/tile/*.py`.
- Canonical algorithm names are synchronized with Dask registry names.
- For algorithms not reimplemented natively in tile code, `algorithms/tile/dask_bridge.py` delegates to Dask shared algorithm classes.
- Optimized for local tile-based processing.

## 7. Design Principles
- No monolithic public entrypoint for algorithms (registry + per-module imports).
- Per-algorithm modular files.
- Shared kernels for duplicate math removal.
- Shared auto-parameter module for consistent behavior.
- Shared local/spatial mode primitives so algorithms can expose:
  - local mode: adjacent-pixel computation
  - spatial mode: multi-radius integration (`radii` + `weights`)
- Hillshade multiscale path also uses the same `radii`/`weights` parameter structure.
- Clear separation:
  - algorithm math
  - orchestration
  - I/O
  - environment/config
- Geographic DEM approximation support:
  - detect geographic CRS and center latitude
  - convert pixel scales to metric `dx/dy`
  - inject anisotropic scales (`pixel_scale_x`, `pixel_scale_y`) through processors into algorithm kernels

## 8. Tests
`tests/` includes baseline regression coverage for the refactored architecture:

- `test_registry_cli_sync.py`
  - Registry and CLI supported-algorithm synchronization.

- `test_algorithm_smoke.py`
  - Dask/Tile smoke execution on small arrays.

- `test_zarr_io.py`
  - Zarr detection and minimal roundtrip behavior.

## 9. Operational Checks
Recommended checks:

```bash
python -m compileall FujiShaderGPU
ruff check FujiShaderGPU
vulture FujiShaderGPU --min-confidence 80
python -m pip check
pytest -q -o addopts='' tests
```

## 10. Developer Navigation Order
1. `FujiShaderGPU/__main__.py`
2. `FujiShaderGPU/cli/base.py`
3. `FujiShaderGPU/cli/linux_cli.py` or `FujiShaderGPU/cli/windows_cli.py`
4. `FujiShaderGPU/core/dask_processor.py` or `FujiShaderGPU/core/tile_processor.py`
5. `FujiShaderGPU/core/dask_cluster.py`, `FujiShaderGPU/core/dask_io.py`, `FujiShaderGPU/core/tile_io.py`, `FujiShaderGPU/core/tile_compute.py`
6. `FujiShaderGPU/algorithms/dask_registry.py`
7. `FujiShaderGPU/algorithms/dask/*.py` and `FujiShaderGPU/algorithms/tile/*.py`
8. `FujiShaderGPU/algorithms/common/kernels.py` and `FujiShaderGPU/algorithms/common/auto_params.py`

## 11. Notes
- Backward-compatibility layers for old monolithic algorithm files are intentionally removed.
- Legacy algorithm aliases are removed; use canonical names from Dask registry.
- Current structure treats modular algorithm files as canonical.
- `hillshade`, `slope`, `specular`, `atmospheric_scattering`, `curvature`, `ambient_occlusion`, `openness`, `multi_light_uncertainty` support unified local/spatial mode on both Dask and tile paths.
