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
  - Shared algorithm base class (`DaskAlgorithm`) and full algorithm implementations.
  - Used by both Dask (`dask/*.py`) and Tile (`tile/*.py` via bridge) backends.

- `tile_shared.py`
  - Tile-side base class (`TileAlgorithm`) and lightweight algorithms that delegate directly to shared kernels.
  - Contains only: `TileAlgorithm`, `ScaleSpaceSurpriseAlgorithm`, `MultiLightUncertaintyAlgorithm`.
  - Most tile algorithms import their implementation from `dask_shared.py` via `tile/dask_bridge.py`.

- `common/kernels.py`
  - Shared CuPy kernels used by both Dask and Tile.

- `common/auto_params.py`
  - Shared auto-parameter helpers (e.g., radii derivation).

- `common/spatial_mode.py`
  - Shared local/spatial execution helpers:
    - spatial radii/weights auto-derivation from YAML presets
    - NaN-aware radius smoothing
    - multi-radius weighted aggregation

- `config/spatial_presets.yaml`
  - Spatial auto presets keyed by pixel-size bins (meters).
  - Current bins:
    - `<5`
    - `5~25`
    - `25~50`
    - `50~250`
    - `250~1250`
    - `1250~5000`
    - `>5000`
  - Each bin defines default `radii` and `weights` used when user does not pass `--radii/--weights`.

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
- `config/auto_tune.py` — VRAM-based dynamic performance parameter computation (single source of truth)
- `config/gpu_config_manager.py`, `config/system_config.py`, `config/gdal_config.py`
- `config/gpu_presets.yaml` — reference anchor values (no longer primary; `auto_tune.py` is authoritative)
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
- Spatial safety fallback:
  - If `--mode spatial` is requested without explicit `--radii/--weights` and input DEM has any side `<= 1024 px`, processor warns and falls back to `local` mode.
- Clear separation:
  - algorithm math
  - orchestration
  - I/O
  - environment/config
- Geographic DEM approximation support:
  - detect geographic CRS and center latitude
  - convert pixel scales to metric `dx/dy`
  - inject anisotropic scales (`pixel_scale_x`, `pixel_scale_y`) through processors into algorithm kernels
  - Hillshade/Specular: `geographic_mode` for azimuth + polarity correction
  - Openness/AO: per-direction physical distance with anisotropic pixel scales
- Dynamic GPU optimization (`config/auto_tune.py`):
  - All performance parameters derived from VRAM at runtime — no static per-GPU presets required
  - Algorithm complexity scaling adjusts chunk/tile sizes and worker counts
  - See §12 for details

## 8. Tests
`tests/` includes baseline regression coverage for the refactored architecture:

- `test_registry_cli_sync.py`
  - Registry and CLI supported-algorithm synchronization.

- `test_algorithm_smoke.py`
  - Dask/Tile smoke execution on small arrays.

- `test_zarr_io.py`
  - Zarr detection and minimal roundtrip behavior.

- `test_fractal_anomaly_normalization.py`
  - Fractal anomaly output range and normalization.

- `test_local_spatial_modes.py`
  - Local/spatial mode switching and radii derivation.

- `test_nodata_handler.py`
  - NoData virtual-fill preprocessing.

- `test_output_nodata_policy.py`
  - Output nodata masking behavior.

- `test_rvi_normalization.py`
  - RVI output normalization stability.

- `test_visual_saliency_normalization.py`
  - Visual saliency normalization.

- `test_visual_saliency_tile_stability.py`
  - Visual saliency tile boundary consistency.

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
- Current structure treats modular algorithm files as canonical.
- Algorithm implementations live in `dask_shared.py`; tile modules import them via bridge.
- `hillshade`, `slope`, `specular`, `atmospheric_scattering`, `curvature`, `ambient_occlusion`, `openness`, `multi_light_uncertainty` support unified local/spatial mode on both Dask and tile paths.

## 12. Dynamic GPU Optimization

### 12.1 Overview
`config/auto_tune.py` dynamically computes all GPU performance parameters from the detected VRAM size.
This replaces the previous approach of static per-GPU presets with a continuous function that works
for any GPU, including hardware not in the preset list.

### 12.2 Anchor-Point Interpolation
Calibration anchors (derived from validated presets for known GPUs) are defined as paired arrays:

| VRAM (GB) | chunk_size | rmm_pool_gb | rmm_fraction |
|-----------|-----------|-------------|--------------|
| 8         | 512       | 4           | 0.50         |
| 12        | 768       | 8           | 0.55         |
| 16        | 1024      | 12          | 0.60         |
| 24        | 2048      | 16          | 0.65         |
| 40        | 8192      | 28          | 0.70         |
| 80        | 14336     | 58          | 0.72         |

- **chunk_size**: interpolated in log₂ space (power-law behavior).
- **rmm_pool_gb / rmm_fraction**: interpolated linearly.
- Values outside anchor range are linearly extrapolated from the nearest segment.

### 12.3 Algorithm Complexity Scaling
`ALGORITHM_COMPLEXITY` in `auto_tune.py` is the single source of truth for per-algorithm cost factors:

```python
ALGORITHM_COMPLEXITY = {
    "hillshade": 0.8,  "slope": 0.8,
    "atmospheric_scattering": 0.9,  "curvature": 1.0,
    "lrm": 1.1,  "npr_edges": 1.1,  "rvi": 1.2,
    "visual_saliency": 1.4,  "specular": 1.5,
    "multiscale_terrain": 1.5,  "fractal_anomaly": 1.6,
    "openness": 1.8,  "ambient_occlusion": 2.0,
}
```

- **chunk_size** is scaled by `1 / complexity^0.4` (complex algorithms get smaller chunks).
- **Worker throttling** uses complexity in per-tile VRAM estimation.
- **Cost warning thresholds** in tile_processor scale with VRAM.

### 12.4 Computed Parameters

| Parameter | Formula | Used By |
|-----------|---------|--------|
| `chunk_size` | log₂ interpolation × complexity⁻⁰·⁴ | Dask pipeline, gpu_config |
| `tile_size` | `chunk_size × 2` | Tile pipeline |
| `dask_chunk` | `chunk_size` × data_gb/VRAM ratio scaling | Dask pipeline only |
| `rmm_pool_size_gb` | linear interpolation, capped by available VRAM | dask_cluster, linux_cli RMM env |
| `rmm_pool_fraction` | linear interpolation (0.50–0.72) | dask_cluster |
| `memory_fraction` | interpolation (0.60–0.85), capped at 0.50 for Colab | dask_cluster |
| `max_workers` | min(CPU-based, VRAM-based, throughput-based) | tile_processor |
| `batch_size` | 2 if VRAM ≥ 40GB, else 1 | tile_processor |
| `prefetch_tiles` | 4 / 3 / 2 by VRAM tier | tile_processor |

### 12.5 Worker Throttling
Worker count is bounded by three independent constraints:

1. **CPU count**: `min(6, cpu_count)` upper bound.
2. **VRAM constraint**: `usable_VRAM / (effective_span² × 4 × 15 × complexity)` per-tile estimate.
3. **Throughput constraint**: GPU compute serialization threshold scaled by `(VRAM/12)^0.3`.
   - ≥ 1.5× threshold → 1 worker
   - ≥ 1.25× threshold → 2 workers
   - ≥ 1.0× threshold → 3 workers

### 12.6 Integration Points

```text
auto_tune()
  ├── system_config.py::get_gpu_config()     → tile_size, max_workers, batch_size
  ├── dask_cluster.py::make_cluster()        → memory_fraction, rmm_pool_size
  ├── dask_processor.py::run_pipeline()      → dask_chunk (VRAM + data_gb aware)
  ├── tile_processor.py::process_dem_tiles() → worker throttle (VRAM + span aware)
  ├── linux_cli.py                           → RMM environment variables
  └── gpu_config_manager.py                  → algorithm_complexity delegation
```

### 12.7 Environment Variable Overrides
Users can still override computed values via environment variables:
- `FUJISHADER_CHUNK_SIZE` — override chunk_size
- `FUJISHADER_RMM_POOL_GB` — override RMM pool size

These are checked in `gpu_config_manager.py::get_preset()` and take precedence over auto_tune.
