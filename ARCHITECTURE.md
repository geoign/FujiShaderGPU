# FujiShaderGPU Architecture

## 1. Purpose

FujiShaderGPU is a Python/CUDA terrain-visualization pipeline for very large DEM rasters.

- Linux path: Dask-CUDA distributed processing for large-scale workloads.
- Windows/macOS path: tile-based local GPU processing without `dask-cuda`.
- Output targets: Cloud Optimized GeoTIFF (COG) and Zarr.

## 2. End-to-End Flow

```text
(optional) Any raster --prepare--> overview-bearing COG (+ NoData fill / detection)
Input DEM (COG with overviews, or Zarr)
  -> CLI (platform-specific)
  -> Core orchestration (Dask or Tile)
  -> NoData -> NaN (declared + optional --nodata override)
  -> Algorithm layer (modular per-algorithm files; float32, NaN-aware)
  -> Output encoding (float32 | int16 | uint8 quantization)
  -> Output writer (COG or Zarr)
  -> Validation / logs
```

The main pipeline **assumes an overview-bearing COG input**.  Use the
preprocessing command (§13) to convert arbitrary rasters and to fill NoData
voids once, up front.  When overviews are missing the pipeline still runs but
warns and points to the preprocessing command (decimated reads fall back to
slow full-resolution reads).

## 3. Code Topology

### 3.1 Algorithm Layer

`FujiShaderGPU/algorithms/`

- `dask_registry.py`
  - Canonical Dask algorithm registry (`ALGORITHMS`).
  - Used by `core/dask_processor.py`.
- `dask_shared.py`
  - **Re-export hub**. Re-exports all public symbols for backward compatibility.
  - Implementations are split into the modules below (Phase 1–3 refactor complete).
  - Used by both Dask (`dask/*.py`) and Tile (`tile/*.py` via bridge) backends.
- **Phase 1 shared-foundation modules**:
  - `_base.py` — `Constants`, `DaskAlgorithm` ABC, `classify_resolution`, `get_gradient_scale_factor`
  - `_nan_utils.py` — NaN handling, spatial smoothing, down/up-sampling, `restore_nan`
  - `_global_stats.py` — `determine_optimal_downsample_factor`, `compute_global_stats`, `apply_global_normalization`
  - `_normalization.py` — per-algorithm stats/normalization functions (`rvi_stat_func`, `npr_stat_func`, etc.)
- **Phase 2–3 algorithm implementation modules** (`_impl_*.py`):
  - `_impl_rvi.py` — RVI (Ridge-Valley Index)
  - `_impl_hillshade.py` — Hillshade
  - `_impl_slope.py` — Slope
  - `_impl_specular.py` — Specular
  - `_impl_atmospheric_scattering.py` — Atmospheric Scattering
  - `_impl_multiscale_terrain.py` — Multiscale Terrain
  - `_impl_curvature.py` — Curvature
  - `_impl_visual_saliency.py` — Visual Saliency
  - `_impl_npr_edges.py` — NPR Edges
  - `_impl_ambient_occlusion.py` — Ambient Occlusion
  - `_impl_lrm.py` — LRM (Local Relief Model)
  - `_impl_openness.py` — Openness
  - `_impl_fractal_anomaly.py` — Fractal Anomaly
  - `_impl_experimental.py` — Scale-Space Surprise + Multi-Light Uncertainty
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
- `io/dem_preprocess.py` — preprocessing core: any GDAL raster -> overview-bearing
  COG (float32, ZSTD) with optional overview-based NoData fill + undeclared-NoData
  detection / override (see §13)
- `io/output_encoding.py` — output dtype encoding: per-algorithm value ranges,
  float32→int16/uint8 quantization (signed/unsigned), NoData policy, GDAL
  scale/offset (see §14). Shared by both backends.
- `prepare.py` — preprocessing CLI (`python -m FujiShaderGPU.prepare`)
- `config/auto_tune.py` — VRAM-based dynamic performance parameter computation (single source of truth)
- `config/gpu_config_manager.py`, `config/system_config.py`, `config/gdal_config.py`
- `config/gpu_presets.yaml` — reference anchor values (no longer primary; `auto_tune.py` is authoritative)
- `utils/scale_analysis.py`, `utils/nodata_handler.py`, `utils/types.py`

## 4. Dask Algorithm Catalog

Registered in `algorithms/dask_registry.py` (and `dask_shared.py` ALGORITHMS):

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
- `test_dem_preprocess.py`
  - `prepare` preprocessing: NoData fill modes and undeclared-NoData detection/override.
- `test_cog_overviews.py`
  - COG overview pyramid presence/structure.
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
- Algorithm implementations live in individual `_impl_*.py` modules; `dask_shared.py` is a thin re-export hub for backward compatibility.
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

## 13. Input Preprocessing (`prepare` command)

### 13.1 Purpose

`prepare.py` / `io/dem_preprocess.py` convert any GDAL-readable raster into a
FujiShaderGPU-ready COG and (optionally) fill NoData voids **once, up front**.
This decouples one-time work (format conversion, COG-ification, hole filling)
from the per-run compute pipeline.

```bash
python -m FujiShaderGPU.prepare input.(tif|img|vrt|...) output_cog.tif
python -m FujiShaderGPU.prepare input.tif out.tif --fill-mode all --force
# console script equivalent: fujishadergpu-prepare input.tif out.tif
```

### 13.2 Output Contract

- Single-band **float32** COG; CRS and pixel grid preserved (**no reprojection**); band 1 used.
- ZSTD compression, `PREDICTOR=3`, `BLOCKSIZE=512` (configurable), internal
  AVERAGE overviews (`OVERVIEW_COUNT=8` by default) — byte-compatible with what
  the pipeline itself writes (`dask_processor.get_cog_options`).
- NoData policy: `none`/`enclosed` emit NaN-nodata; `all` emits a dense raster
  with no NoData.

### 13.3 NoData detection / override (`--nodata`, `--no-detect-nodata`)

Before filling, every NoData cell is normalized to float **NaN** so finite
sentinels are handled identically to declared NoData. Sources of NoData:

1. **Declared** NoData (`src.nodata`) — always honored via masked I/O.
2. **Explicit override** `--nodata VALUE` (`-9999`, `0`, `nan`, …) — treated as
   NoData even when the raster declares none or a different value.
3. **Auto-detected undeclared NoData (default ON)** — `io/dem_preprocess.py::
   _detect_border_nodata` scans a NEAREST-resampled coarse grid: when a single
   finite value dominates the outer ring (≥ `--nodata-border-fraction`, default
   `0.5`) and covers a non-trivial share of the grid, it is treated as a lost
   NoData frame (sea / dataset exterior whose tag was dropped in conversion).
   Disable with `--no-detect-nodata`. Already-declared values are not re-reported.

Rationale: an undeclared constant frame, if left as data, is read as a flat
plateau adjacent to real terrain; large-radius / multiscale operators then render
a halo along the data edge. Converting it to NaN lets the NaN-aware kernels exclude
it. The main pipeline mirrors source (2) via its own `--nodata` (both backends),
applied at load time before any algorithm runs.

### 13.4 NoData Fill Modes (`--fill-mode`)

- `none` — no filling; NoData preserved.
- `enclosed` *(default)* — fill only interior voids (NoData **not** connected to
  the raster border).  Border-connected NoData (ocean / dataset exterior) is kept.
- `all` — fill every NoData cell (including exterior) and remove NoData entirely.
  Useful when NaN/NoData is not allowed (e.g. 3D model generation); large
  exterior fills are a coarse extrapolation.

### 13.5 Method (overview-based, low-frequency fill)

Hole filling is inherently low-frequency, so the expensive work runs on a small
coarse grid and is upsampled — cost is nearly independent of full raster size:

1. Read a coarse decimated grid (`--coarse-max`, default 2048; uses overviews if present).
2. Global edge-connectivity on the coarse grid separates exterior from enclosed voids.
3. Build a smooth coarse fill surface (nearest + valid-weighted Gaussian).
4. Stream the full-resolution output in row bands; bilinearly sample the **global**
   coarse surface at each pixel (seamless across bands) and fill the targeted voids.
5. Convert the streamed temporary GeoTIFF into the final COG with overviews.

Implementation streams windowed reads/writes (bounded memory) so very large
rasters (hundreds of GB) are handled without loading the full array.

### 13.6 Large-radius-from-overview (RVI)

Because the input is an overview-bearing COG, large-radius low-frequency terms
can be taken from the stored overview instead of reading a huge per-chunk/per-tile
halo at full resolution.  For RVI (`Σ wᵢ(block − meanᵢ)`):

- Radii are split at `max(256, chunk_or_tile // 16)`.
- **Small radii** are computed at full resolution with a small halo (as before).
- **Large radii** contribute as `W_large·block − upsample(Σ wᵢ·meanᵢ_overview)`,
  where the coarse mean field is computed once from the overview (decimated read).
  The per-block part is `W_large·block` (no halo); the field is sampled at global
  pixel coords (offset 0 for Dask chunks, tile-window origin for tiles), so the
  result is seam-free across both backends.

This cuts the per-chunk/per-tile halo from `max_radius` to `max(small_radii)`
(e.g. 2064→272 px for radii up to 2048), roughly halving I/O for large radii.
RVI reads the stored overview directly (biggest saving); any failure falls back
transparently to the full-resolution radii path.

The **spatial-mode algorithms** (hillshade, slope, specular, atmospheric_scattering,
curvature, ambient_occlusion, openness, multi_light_uncertainty) apply the same
idea via `_nan_utils.coarse_large_radius_response`: for radii above the threshold,
the per-radius response is computed on a `da.coarsen`-downsampled copy (metric
pixel scales scaled by the factor) and bilinearly upsampled, then combined as
before.  Small radii keep the exact previous full-resolution code path, so typical
runs (preset radii ≤ 64) are unchanged.  This is enabled for **projected DEMs only**
(geographic DEMs use decimation-dependent local scaling); the coarse copy is
derived from the same array, so it is offset-free and seam-free on both backends.

Algorithms that do **not** need this: `multiscale_terrain` already caps its halo
at `Constants.MAX_DEPTH` (150 px), and `fractal_anomaly` / `scale_space_surprise`
/ `visual_saliency` / `npr_edges` use small scales (no large halo) and combine
scales non-linearly (not linearly decomposable).

### 13.7 Pipeline Input Assumption

The main pipeline assumes an overview-bearing COG.  Both backends warn (not fail)
when overviews are missing and point to this command.  NoData fill management is
**owned solely by the preprocessing command**: the legacy per-tile/per-chunk
hole-fill has been removed from both the Dask and tile pipelines (along with the
`--no-fill-dem-holes` / `--hole-fill-max-components` CLI options).  The pipelines
still *mask* NoData and use NaN-aware filters, but they no longer interpolate
voids — run `prepare` (mode `enclosed` or `all`) first if voids must be filled.

### 13.8 NaN-aware coarse resampling (boundary halo prevention)

The large-radius coarse paths (`_nan_utils._downsample_nan_aware` /
`_upsample_to_shape`) are **NaN-aware at the data boundary**: the broad exterior
NoData is preserved as NaN (only thin, well-enclosed interior voids are smoothed),
and upsampling interpolates valid contributors only.  This prevents the exterior
from being filled with a finite plateau (≈ global mean elevation) that would leak
into interior valid pixels and render a halo just inside the data edge — the same
class of artifact that §13.3's NoData detection prevents at the source.

## 14. Output Data Type Encoding (`io/output_encoding.py`)

### 14.1 Purpose

Algorithms always compute in **float32** (NaN = NoData) on the GPU.  As a final,
optional step the result can be quantized to a compact integer COG for delivery:
smaller files → faster COG builds (less disk I/O), cheaper object-storage transfer,
lighter QGIS reads.  `--output-dtype {float32,int16,uint8}` (default `float32`)
selects the encoding; `float32` is unchanged, byte-for-byte previous behavior.

### 14.2 Encoding rules

- **NoData = 0** for both integer types (NaN → 0).
- Each algorithm has a known native value range (`OUTPUT_VALUE_RANGES`), e.g.
  slope `0..90`, AO/hillshade `0..1`, RVI/LRM/fractal `-1.5..1.5`,
  visual_saliency/multiscale_terrain/scale_space_surprise `0..1.5`, npr_edges
  `0.2..1.0`.  `--output-range lo,hi` overrides; unbounded cases (e.g. slope in
  `percent`) fall back to a robust `[p1, p99]` estimate from a strided sample.
- **Unsigned** range → data fills `[1, MAXPOS]` (255 / 32767).
- **Signed** range (lo < 0 < hi) is encoded symmetrically about 0:
  - `int16`: full `[-MAXPOS, +MAXPOS]`; `DN = 0` (value ≈ 0, i.e. flat ground)
    doubles as NoData — visually negligible.
  - `uint8`: value 0 centered at `128`, data in `[1, 255]`.
- GDAL `scale`/`offset` are recorded (`value = scale·DN + offset`) for physical
  recovery; `DN = 0` is NoData (undefined value).

### 14.3 Backend integration

Mapping parameters come from `output_encoding.quantize_params`; both backends use
the same registry so int outputs are consistent.

- **Dask** (`dask_processor.run_pipeline`): a CuPy `map_blocks` quantize step is
  inserted between the result and the host transfer (`_quantize_block_cp`); the
  writers set NoData by dtype (`output_nodata_for_dtype`: NaN for float, 0 for int).
- **Tile** (`tile_processor`): `quantize_array` (NumPy) runs per tile after
  `_format_algorithm_output`; tile profiles and the VRT/COG assembler inherit the
  integer dtype + NoData=0 from the tiles (data-driven, no extra plumbing).
- After the COG is built, `apply_scale_offset` records the band scale/offset
  (best-effort; non-critical).  Float-only display hints (e.g. RVI ±1) are skipped
  for integer outputs.

Quantization is skipped (writes float32) when the algorithm changes the result
shape (e.g. `agg=stack`) or when no range can be resolved.
