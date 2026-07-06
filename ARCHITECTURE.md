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
  - `_base.py` — `Constants`, `DaskAlgorithm` ABC, `classify_resolution`
  - `_nan_utils.py` — NaN handling, spatial smoothing, down/up-sampling, `restore_nan`,
    and the shared **overview large-radius helpers** (`read_overview_coarse_dem`,
    `coarse_large_radius_response`, `multiscale_response_fields`,
    `hybrid_multiscale_response_combine`, `compute_overview_scale_fields`). These are
    **tile-origin aware**: the optional `tile_origin` / `tile_full_shape` arguments let
    the same code sample one global overview field correctly from a single tile
    (default `None` = the Dask whole-raster behavior), which is what makes the tile
    backend seam-free and identical to Dask.
  - `_global_stats.py` — `determine_optimal_downsample_factor`, `compute_global_stats`, `apply_global_normalization`, `apply_display_stretch_dask`
  - `_normalization.py` — per-algorithm stats/normalization functions (`topousm_fast_stat_func`, `robust_unsigned_stretch_stat_func`, etc.)
  - `_norm_stats.py` — **backend-neutral** full-resolution normalization statistics:
    `_NORM_STAT_SPECS` (algorithm → raw block + stat function) and
    `_compute_norm_stats_tiled` (robust display stat pooled over stratified
    **full-resolution** tiles). Used by both backends so integer outputs match.
- **Phase 2–3 algorithm implementation modules** (`_impl_*.py`):
  - `_impl_topousm_fast.py` — TopoUSM Fast
  - `_impl_hillshade.py` — Hillshade
  - `_impl_slope.py` — Slope
  - `_impl_specular.py` — Specular
  - `_impl_atmospheric_scattering.py` — Atmospheric Scattering
  - `_impl_multiscale_terrain.py` — Multiscale Terrain
  - `_impl_blur.py` — Blur (raw Gaussian-smoothed elevation)
  - `_impl_curvature.py` — Curvature
  - `_impl_visual_saliency.py` — Visual Saliency (Itti–Koch–Niebur 1998-style, simplified for single-band terrain: DoG center-surround + gradient-orientation; no N(·) operator / Gabor / colour / attention dynamics)
  - `_impl_npr_edges.py` — NPR Edges
  - `_impl_ambient_occlusion.py` — Ambient Occlusion (stylized SSAO heuristic, not a physically based hemisphere integral / sky-view factor)
  - `_impl_openness.py` — Topographic Openness (Yokoyama et al. 2002; directional mean of per-azimuth horizon angles, positive/negative)
  - `_impl_fractal_anomaly.py` — Fractal Anomaly (FujiShaderGPU-original composite; core = standard fractal-surface roughness-vs-scale log-log regression / Hurst exponent, anomaly feature is bespoke)
  - `_impl_experimental.py` — Scale-Space Surprise (FujiShaderGPU-original Σ|DoG| cross-scale measure on standard Gaussian scale-space; NOT Bayesian Surprise) + Multi-Light Uncertainty
  - `_impl_structure_tensor.py` — Structure Tensor Fabric (Bigün & Granlund 1987; Weickert-coherence + strike orientation via double-angle vector averaging; outputs coherence / orientation / fabric). Also hosts the shared Gaussian-derivative + NaN-fill helpers used by frangi / scale_drift / lic.
  - `_impl_frangi.py` — Frangi Vesselness (Frangi et al. 1998; scale-normalized Hessian eigenvalue ridge/valley network filter; global energy scale `c` from the stats prepass)
  - `_impl_lic.py` — Line Integral Convolution (Cabral & Leedom 1993; flow/contour streamline texture; noise = elevation-value hash so tiles and backends are seam-free by construction; default hillshade composite)
  - `_impl_phase_congruency.py` — Phase Congruency relief (Kovesi 1999 via the monogenic signal, Felsberg & Sommer 2001; amplitude-invariant feature detection; per-block FFT with 2·λ halo, wavelengths clamped to ≤64 px — larger scales are a future overview-path extension)
  - `_impl_tv_decomposition.py` — TV structure–texture decomposition (ROF 1992 / TV-L1 Chan & Esedoglu 2005, Chambolle–Pock 2011 primal-dual; halo = iterations so tiles are exactly chunk-independent; halo-free texture around scarps)
  - `_impl_scale_drift.py` — Scale-Drift Field (FujiShaderGPU-original: Lucas–Kanade drift between adjacent Gaussian scale-space levels = vector version of scale_space_surprise; outputs magnitude / direction / divergence)
  - Some `_impl_*` modules also host **backend-neutral global-stat helpers** that
    both pipelines import (so they are not duplicated, and the Windows tile path does
    not import the Dask-only `core/dask_processor.py`): `_compute_npr_grad_stats`
    (`_impl_npr_edges.py`, global per-radius edge threshold) and
    `_compute_fractal_relief_stats` (`_impl_fractal_anomaly.py`, global roughness
    p10/p75). `core/dask_processor.py` imports both from here.
- `tile_shared.py`
  - Tile-side base class (`TileAlgorithm`) plus re-exports of
    `ScaleSpaceSurpriseAlgorithm` / `MultiLightUncertaintyAlgorithm` (which are
    bridge adapters in `tile/`, like every other tile algorithm).
  - All tile algorithms import their implementation from `dask_shared.py` via `tile/dask_bridge.py`.
- `common/kernels.py`
  - Shared CuPy kernels used by both Dask and Tile.
- `common/spatial_mode.py`
  - Shared local/spatial helpers, including the spatial radii/weights auto rule:
    geometric radii `[2,8,32,128,512,2048]` truncated so the largest is
    `<= min(DEM_short_side/10, 2048)`, with a `2**n` (near-weighted) weight
    profile normalized to 1.  Used by every radius-driven spatial algorithm
    (`RADII_DRIVEN_ALGOS`) on both backends; the orchestrators resolve it once
    from the full-raster short side and inject explicit `radii`/`weights`.
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
- `utils/nodata_handler.py`, `utils/types.py`
- `utils/paths.py` — filesystem path helpers shared across backends. `resolve_tmp_dir`
  (env-driven staging dir), and the **virtual-filesystem-safe** `safe_abspath` /
  `safe_unlink` used when output/temp paths live on a FUSE mount (rclone / cloud
  drive) where `Path.resolve()` raises `WinError 1005` and a just-closed GDAL handle
  delays `unlink` (`WinError 32`). See §6.

## 4. Dask Algorithm Catalog

Registered in `algorithms/dask_registry.py` (and `dask_shared.py` ALGORITHMS):

- `topousm_fast`
- `hillshade`
- `slope`
- `specular`
- `atmospheric_scattering`
- `multiscale_terrain`
- `blur`
- `curvature`
- `visual_saliency`
- `npr_edges`
- `ambient_occlusion`
- `openness`
- `fractal_anomaly`
- `scale_space_surprise`
- `multi_light_uncertainty`
- `structure_tensor`
- `frangi`
- `lic`
- `phase_congruency`
- `tv_decomposition`
- `scale_drift`

These names are also the canonical CLI names for the tile backend.

Notes on the 2026-07 batch (`structure_tensor` … `scale_drift`):

- All six are single-band float32 products (like every other algorithm).
  Direction-carrying outputs are exposed as selectable single-band modes
  (`--st-output orientation` / `--drift-output direction`, angle mapped to
  [0, 1)); an HSV/RGB composite is deferred until the Dask COG writer grows
  multi-band support (the tile writer already handles HxWxC).
- `structure_tensor` / `frangi` are radius-driven spatial algorithms
  (`RADII_DRIVEN_ALGOS`): auto radii, `--weights`, and the unified
  overview large-radius path all apply. `scale_drift` / `phase_congruency`
  are intrinsically multi-scale (`MULTISCALE_REQUIRED_ALGOS`, local mode
  falls back to spatial); `lic` / `tv_decomposition` are mode-independent.
- Bounded-halo designs: `lic` length ≤ 120 px, `phase_congruency`
  wavelengths ≤ 64 px, `tv_decomposition` iterations ≤ 140 — each keeps its
  per-tile halo inside `Constants.MAX_DEPTH` (see
  `_required_padding_for_algorithm`), so the tile output is seam-free
  (verified: tiled vs single-tile runs differ only at the per-block NaN-fill
  ring around NoData, the same behavior as the older algorithms).
- Global statistics (`_NORM_STAT_SPECS`): display stretch for
  structure_tensor / scale_drift, Frangi `c`, phase-congruency noise median,
  and the TV texture tanh scale are all computed once, backend-neutrally.

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
- **Output parity with Dask.** In `spatial` mode the bridge routes every algorithm
  except TopoUSM Fast through the Dask algorithm on the single tile (TopoUSM Fast keeps a bespoke
  direct path; `local` mode keeps the fast direct paths). Combined with the unified
  overview large-radius field (§13.6), shared full-resolution normalization stats
  (§14), identical raw-elevation + metric pixel-scale conventions (projected AND
  geographic), and no tile-side post-normalization, the outputs match the
  Dask-CUDA backend.
- **Virtual filesystem (rclone/FUSE) safety.** Reading or writing on a cloud-mounted
  drive (e.g. Cloudflare R2 via an rclone FUSE mount) is supported: `utils/paths.py`'s
  `safe_abspath` avoids `Path.resolve()` (which raises `WinError 1005` on such mounts)
  and `safe_unlink` retries the temp-file delete that a just-closed GDAL handle blocks
  (`WinError 32`). Both are no-ops on a normal disk.

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
  - detect geographic CRS and center latitude (per-tile latitude on the tile backend)
  - convert pixel scales to metric `dx/dy` and inject them as **real signed
    meters per pixel** (`pixel_scale_x`, `pixel_scale_y`) on BOTH backends; the
    DEM array itself is never rescaled, so elevation-based outputs keep their
    physical magnitude and the shared normalization stats apply unchanged
  - geotransform orientation is handled uniformly via the pixel-scale signs
    (east/north derivative correction inside the kernels); geographic and
    projected DEMs share the exact same shading formulas and tone scale
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
- `test_output_nodata_policy.py`
  - Output nodata masking behavior.
- `test_dem_preprocess.py`
  - `prepare` preprocessing: NoData fill modes and undeclared-NoData detection/override.
- `test_cog_overviews.py`
  - COG overview pyramid presence/structure.
- `test_topousm_fast_normalization.py`
  - TopoUSM Fast output normalization stability.
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
8. `FujiShaderGPU/algorithms/common/kernels.py` and `FujiShaderGPU/algorithms/common/spatial_mode.py`

## 11. Notes

- Current structure treats modular algorithm files as canonical.
- Algorithm implementations live in individual `_impl_*.py` modules; `dask_shared.py` is a thin re-export hub for backward compatibility.
- `hillshade`, `slope`, `specular`, `atmospheric_scattering`, `curvature`, `ambient_occlusion`, `openness`, `multi_light_uncertainty`, `npr_edges` support unified local/spatial mode (`npr_edges` spatial = outlines at multiple smoothing scales; tile path bridges via Dask-shared). Intrinsically multi-scale algorithms (`topousm_fast`, `multiscale_terrain`, `visual_saliency`, `scale_space_surprise`, `fractal_anomaly`) consume the unified `--radii` as their scale set.

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
    "npr_edges": 1.1,  "topousm_fast": 1.2,
    "visual_saliency": 1.4,
    "multiscale_terrain": 1.5,  "fractal_anomaly": 1.6,
    "openness": 1.8,  "ambient_occlusion": 2.0,  "specular": 2.0,
}
```

> `specular` was raised `1.5 → 2.0` so a single padded block (surface normals +
> multiple filters, ~3× baseline per-pixel VRAM) stays within the RMM device pool
> on small-but-large datasets; see `config/auto_tune.py`.

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
  └── linux_cli.py                           → RMM environment variables
```

`gpu_config_manager.py` no longer participates in parameter computation; it only
classifies the detected GPU into a named bucket for the run-description label.

### 12.7 Manual Overrides

Computed values can be overridden from the CLI:

- `--chunk` (Dask) — override the Dask chunk width.
- `--tile-size` (tile) — override the tile size.
- `--memory-fraction` (Dask) — override the device-memory fraction.

There is no longer an environment-variable override path; the CLI flags above are
the single, explicit override surface.

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

### 13.3 NoData detection / override (`--nodata`, default `auto`)

Before filling, every NoData cell is normalized to float **NaN** so finite
sentinels are handled identically to declared NoData. `--nodata auto` (the default)
infers NoData with this rule chain, taking the **union** of what it finds (all on a
NEAREST-resampled coarse grid so the exact sentinel survives; already-declared or
already-found values are not re-reported):

1. **Declared** NoData (`src.nodata`) — always honored via masked I/O.
2. **Undeclared sentinel / extreme dominating the whole grid** —
   `io/dem_preprocess.py::_detect_sentinel_nodata`: the most common finite value,
   when it covers ≥ `--nodata-sentinel-fraction` (default `0.05`) of the grid **and**
   is a known fill (`0`, `-9999`, int8/16/32 min/max, float32 extremes, `_NODATA_SENTINELS`)
   or the data-range extreme. Catches a float DEM padded with `0`/`-9999` whose tag
   was lost in conversion, **including when the fill is spread through the interior**
   (which rule 3 alone misses).
3. **Undeclared value dominating the border** — `_detect_border_nodata`: a single
   value occupying ≥ `--nodata-border-fraction` (default `0.5`) of the outer ring and
   a non-trivial share of the grid — a lost sea / dataset-exterior frame.
4. **Otherwise no NoData.**

Other `--nodata` values: a number **forces** that value (no inference); `none` (or
`--no-detect-nodata`) disables inference; `nan` marks NaN.

Rationale: an undeclared sentinel left as data is read as flat terrain — it skews
every algorithm's normalization range (the stat pre-pass §14 pools it as a huge spike
at one value) and large-radius / multiscale operators render a halo along the data
edge. Converting it to NaN lets the NaN-aware kernels exclude it. The main pipeline
mirrors an **explicit** `--nodata VALUE` (both backends) at load time before any
algorithm runs.

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

### 13.6 Unified large-radius-from-overview (all spatial algorithms, both backends)

Because the input is an overview-bearing COG, large-radius low-frequency terms are
taken from the stored overview instead of reading a huge per-chunk/per-tile halo at
full resolution. **Every spatial-mode algorithm uses this**, so each tile/chunk only
reads a halo for the *small* radii — large radii come from one global overview field,
which is both far cheaper and **seam-free**.

Orchestration (mirrored in `core/dask_processor.py` and `core/tile_processor.py`):

- Read **one** decimated overview of the DEM once (`_nan_utils.read_overview_coarse_dem`)
  and inject it as `_overview_coarse_dem` / `_overview_decimation`.
- **Hybrid algorithms** (`fractal_anomaly`, `visual_saliency`, `scale_space_surprise`)
  additionally precompute per-large-scale response fields from that overview
  (`compute_overview_scale_fields`) → `_<algo>_large_fields`. The algorithm computes
  the small scales at full resolution and samples the large fields with no halo inside
  one depth-0 combine (`hybrid_multiscale_response_combine`).
- The **spatial-switch algorithms** (`hillshade`, `slope`, `specular`,
  `atmospheric_scattering`, `curvature`, `ambient_occlusion`, `openness`,
  `multi_light_uncertainty`, `npr_edges`, `multiscale_terrain`) use
  `multiscale_response_fields` / `coarse_large_radius_response`: each large radius runs
  the per-radius block on the overview and is bilinearly upsampled; small radii keep
  the exact full-resolution path.
- **TopoUSM Fast** keeps a bespoke split (`Σ wᵢ(block − meanᵢ)`: large radii =
  `W_large·block − upsample(Σ wᵢ·meanᵢ_overview)`), the biggest single saving.
- Per-algorithm global stats that would otherwise vary per block (npr edge threshold,
  fractal relief p10/p75) are injected globally (§3.1) so the small-radius part is also
  consistent across tiles.

Tile-origin awareness is the key to backend parity: the coarse field is sampled at
**global** pixel coordinates. On Dask the chunk's `array-location` is already global;
on the tile backend the dask array is a single tile, so `core/tile_processor.py` injects
that tile's window origin as `_tile_origin` (+ global `_tile_full_shape`), which the
shared helpers add to the block coordinates (`tile_origin` / `tile_full_shape` args,
§3.1). The split is sized so the halo covers only the small radii — e.g.
`fractal_anomaly` with radii up to 1024 drops from a 2080 px halo (a 5184 px effective
tile, throttled to one worker, ~hours on a 19469×15478 raster) to a 160 px halo (1344 px
tile, full worker count, minutes).

Enabled for **projected DEMs only** (geographic DEMs use per-tile local latitude
scaling that a single global field cannot match — those fall back to the
full-resolution per-tile path). Any failure to read/build the overview falls back
transparently to the full-resolution radii path.

Known limitation: the hybrid algorithms' final combine must run at depth 0
(`array-location` is only the true global position under `map_blocks`), and that
combine contains small-kernel operations (fractal: sigma-1.2 smoothing + 3x3
median; saliency/surprise: a 1-px gradient).  On the Dask backend this can leave
a faint, few-pixel-wide difference along chunk borders for those algorithms; the
tile backend is unaffected (each tile carries its own halo and is trimmed after
the combine).

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
- Normalized algorithms (TopoUSM Fast, fractal_anomaly, visual_saliency, scale_space_surprise,
  multiscale_terrain, and the data-driven `ambient_occlusion` / `openness` stretch)
  derive their display scale from a **full-resolution stratified pre-pass**, not a
  decimated overview: `algorithms/_norm_stats.py::_compute_norm_stats_tiled` runs the
  algorithm's raw block (`normalize=False`) on several **full-resolution** windows
  stratified across the valid extent, pools the interior pixels, and takes the robust
  stat (`_NORM_STAT_SPECS` maps each algorithm to its raw block + stat function; the
  robust **p99** = `NORMAL_PERCENTILE` — a p1→0 / p99→1 stretch for the unsigned maps,
  a robust absolute-p99 scale for the signed TopoUSM Fast). Full
  resolution keeps scale-sensitive magnitudes correct (decimating shrinks the detail
  and under-estimates the scale, which over-amplifies the output and blows out the
  integer encoding); pooling the whole extent is robust to off-center / sparse data
  and dilutes singular outliers into the unclipped `>1` tail. **Both backends import
  the same `_compute_norm_stats_tiled`**, so the displayed contrast matches. The
  normalized float is **unclipped**, so the tail runs a
  little past `±1`.
- **Normalization is owned by the algorithms.** Every display stretch /
  normalization happens inside `.process()` (`apply_display_stretch_dask` in
  `ambient_occlusion` / `openness`; internal normalization with the injected
  global stats in TopoUSM Fast / multiscale_terrain / visual_saliency /
  scale_space_surprise / fractal_anomaly), identically on both backends.  The
  tile pipeline applies **no post-normalization of its own** — the old tile-only
  generic p1–p99 stretch made the Windows output diverge from Dask.
- Native value ranges (`OUTPUT_VALUE_RANGES`): slope `0..90`, AO/hillshade/openness
  `0..1` (physically bounded, no pre-stat), npr_edges `0.2..1.0`; the normalized set
  uses `±1.176` (signed) / `0..1.176` (unsigned), where `1.176 = 1/0.85` reserves
  int16/uint8 headroom so value `±1` lands at ~85% of the code range and the tail up
  to `±1.176` is encoded before clipping.  `--output-range lo,hi` overrides; unbounded
  cases (e.g. slope in `percent`) fall back to a robust `[p1, p99]` estimate.
- **Unsigned** range → data fills `[1, MAXPOS]` (255 / 32767).
- **Signed** range (lo < 0 < hi) is encoded symmetrically about 0:
  - `int16`: full `[-MAXPOS, +MAXPOS]`; `DN = 0` (value ≈ 0, i.e. flat ground)
    doubles as NoData — visually negligible.
  - `uint8`: value 0 centered at `128`, data in `[1, 255]`.
- Integer outputs are plain DN **display products**: no GDAL `scale`/`offset` is
  embedded (QGIS would auto-unscale the band, and editing a finished COG in place
  breaks its layout guarantee — GDAL 3.8+ refuses the update).  The DN↔value
  mapping (`value = scale·DN + offset`) is logged by both backends and recorded
  in the Dask COG attrs; `DN = 0` is NoData (undefined value).

### 14.3 Backend integration

Mapping parameters come from `output_encoding.quantize_params`; both backends use
the same registry so int outputs are consistent.

- **Dask** (`dask_processor.run_pipeline`): a CuPy `map_blocks` quantize step is
  inserted between the result and the host transfer (`_quantize_block_cp`); the
  writers set NoData by dtype (`output_nodata_for_dtype`: NaN for float, 0 for int).
- **Tile** (`tile_processor`): `quantize_array` (NumPy) runs per tile after
  `_format_algorithm_output`; tile profiles and the VRT/COG assembler inherit the
  integer dtype + NoData=0 from the tiles (data-driven, no extra plumbing).
- The finished COG is never edited in place (no post-hoc scale/offset or display
  hints — those edits break the COG layout and GDAL 3.8+ rejects them).

Quantization is skipped (writes float32) when the algorithm changes the result
shape (e.g. `agg=stack`) or when no range can be resolved.
