# FujiShaderGPU — Comprehensive Code Review

**Reviewer**: Antigravity AI Assistant  
**Date**: 2026-02-12  
**Scope**: Full codebase review of `FujiShaderGPU/` against `ARCHITECTURE.md`  
**Focus**: Obvious errors, improper dependencies, unused functions, inefficient routines

---

## Executive Summary

Overall, the FujiShaderGPU codebase is **well-architected** with a clear separation of concerns across algorithmic, core, and CLI layers. The dual-backend design (Dask-CUDA for Linux, tile-based for Windows/macOS) is thoughtfully implemented with shared kernel code. The dynamic GPU auto-tuning system (`auto_tune.py`) is particularly elegant.

However, the review identified **23 findings** across 5 categories:

| Category | Critical | Medium | Low |
|---|---|---|---|
| 🐛 Bugs / Correctness | 2 | 3 | 1 |
| 🔗 Dependency Issues | 0 | 2 | 2 |
| 🗑️ Dead / Unused Code | 0 | 3 | 3 |
| ⚡ Efficiency | 0 | 3 | 2 |
| 📝 Architecture / Docs Drift | 0 | 1 | 1 |

---

## 🐛 Bugs / Correctness Issues

### B-1 [CRITICAL] `dask_cluster.py:64` — Hardcoded Linux path as `local_directory`

```python
cluster = LocalCUDACluster(
    ...
    local_directory='/tmp',  # ← HARDCODED UNIX PATH
)
```

**Problem**: This file is imported by the Linux CLI, but the hardcoded `/tmp` would fail if the code was ever invoked on a non-Linux platform. More importantly, the Dask-CUDA cluster always writes spill-to-disk data to `/tmp`, which may not be the fastest filesystem.

**Recommendation**: Use `tempfile.gettempdir()` or make it configurable:
```python
import tempfile
local_directory=os.environ.get('FUJISHADER_SPILL_DIR', tempfile.gettempdir()),
```

---

### B-2 [CRITICAL] `dask_processor.py:220-222` — Brittle GDAL version parsing

```python
def check_gdal_version() -> tuple:
    version = gdal.VersionInfo("VERSION_NUM")
    major = int(version[0])
    minor = int(version[1:3])
    return major, minor
```

**Problem**: `gdal.VersionInfo("VERSION_NUM")` returns a numeric string like `"3080100"` (GDAL 3.8.1). The code parses it by indexing characters: `version[0]` → `"3"`, `version[1:3]` → `"08"`. This works for GDAL 3.x. **However, if GDAL ever reaches version 10.x or higher**, `version[0]` would only capture `"1"`, and the minor version would be wrong.

**More robust approach**:
```python
def check_gdal_version() -> tuple:
    ver_num = int(gdal.VersionInfo("VERSION_NUM"))
    major = ver_num // 1_000_000
    minor = (ver_num % 1_000_000) // 10_000
    return major, minor
```

---

### B-3 [MEDIUM] `scale_analysis.py:96-97` — Incorrect sample reshaping after NoData removal

```python
def _analyze_scale_variances_ultra_fast(dem_sample, ...):
    sample_size = int(np.sqrt(len(dem_sample)))
    dem_2d = dem_sample[:sample_size*sample_size].reshape(sample_size, sample_size)
```

**Problem**: The `dem_sample` parameter arrives from `analyze_terrain_scales` where, after NoData removal (line 71), the array is **1D** (only valid pixels). Taking `int(np.sqrt(len(1d_array)))` and reshaping introduces spatial distortion — the reconstructed 2D grid no longer represents the actual terrain layout. Gaussian filter operations on this distorted grid produce meaningless spatial frequency analysis.

**Recommendation**: Perform the reshape *before* NoData removal (or pass the full 2D sample and fill NoData with `np.nan` for the `cpx_ndimage.gaussian_filter` call):
```python
# Keep 2D structure; mask NoData as NaN instead of removing
if nodata is not None:
    dem_sample[dem_sample == nodata] = np.nan
dem_2d = dem_sample  # already 2D from src.read()
```

---

### B-4 [MEDIUM] `tile_processor.py:1430-1433` — Multi-band result core extraction misses 3D

```python
result_core = result_tile[
    core_y_in_win : core_y_in_win + core_h,
    core_x_in_win : core_x_in_win + core_w,
]
```

**Problem**: If `result_tile` has shape `(H, W, C)` (multi-band), this slicing only selects rows and columns but doesn't break — Python's slice syntax handles the trailing dimension correctly. However, this relies on implicit behavior and could break if an algorithm returned `(C, H, W)` ordering. Consider being explicit for safety.

**Risk**: Low, since `tile_compute.py` enforces `HxWxC` format, but worth a guard.

---

### B-5 [MEDIUM] `cog_builder.py:471` — VRT filename always `rvi_tiles.vrt`

```python
vrt_path = os.path.join(tmp_tile_dir, "rvi_tiles.vrt")
```

**Problem**: The VRT filename is hardcoded as `rvi_tiles.vrt` regardless of which algorithm was run. While functionally harmless (it's a temporary file), it's misleading during debugging. Renaming to `tiles.vrt` or `{algorithm}_tiles.vrt` would improve clarity.

---

### B-6 [LOW] `dask_processor.py:873-874` — `client.close()` then `client.shutdown()`

```python
client.close(timeout=10)
client.shutdown()  # Already closed above
```

**Problem**: Calling `client.shutdown()` after `client.close()` is redundant at best, and may raise exceptions (which are caught). The `close()` call with a timeout should suffice. `shutdown()` is typically used on the scheduler side in Dask Distributed.

---

## 🔗 Dependency Issues

### D-1 [MEDIUM] `dask_processor.py` — Top-level imports of Dask/CuPy/GPUtil

The `dask_processor.py` module imports `cupy`, `dask.array`, `dask_cuda`, and `GPUtil` at the top level. On Windows/macOS, these packages may not be installed (they are `linux` optional deps). This is partially mitigated by the guarded import in `core/__init__.py`, but **if any code path accidentally imports `dask_processor` on Windows**, it will crash immediately with `ImportError`.

**Recommendation**: The current architecture in `core/__init__.py` handles this with platform checks. However, the top-level imports in `dask_processor.py` could be wrapped in a try block for robustness, or the entire module could use lazy imports.

---

### D-2 [MEDIUM] `scale_analysis.py` — `cupy` (GPU) import in utils module

```python
import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
```

This utility module is imported by `tile_processor.py` on all platforms. While it has a scipy CPU fallback, the `cupy` import at the top level will fail if CuPy is not installed. Since this is a *core* pipeline dependency, it's acceptable, but the module could benefit from a lazy import pattern for the GPU path.

---

### D-3 [LOW] `spatial_mode.py:8` — `cupy` import in shared algorithm module

The `spatial_mode.py` module (in `algorithms/common/`) imports `cupy` at the top level. This module is shared between Dask and tile backends. Both backends require CuPy, so this is correct, but it means the YAML loading/preset selection logic (which is pure Python) cannot be used independently for testing without a GPU.

---

### D-4 [LOW] `pyproject.toml` — `scipy` is a Windows-only dependency but used broadly

`scipy` is listed only under `[project.optional-dependencies] windows`. However, `nodata_handler.py` and `scale_analysis.py` use `scipy.ndimage` with graceful fallbacks. Since these modules are used in the tile pipeline (which runs on all platforms), consider adding `scipy` to core dependencies or documenting this optional behavior.

---

## 🗑️ Dead / Unused Code

### U-1 [MEDIUM] `dask_processor.py:86-92` — Thin wrapper functions

```python
def get_optimal_chunk_size(gpu_memory_gb=40):
    return _cluster_optimal_chunk_size(gpu_memory_gb)

def make_cluster(memory_fraction=0.6):
    return _cluster_make_cluster(memory_fraction)
```

These are one-line wrappers around `dask_cluster.py` functions. They add an indirection layer without value. The `run_pipeline` function calls `make_cluster()` directly, which is the same function imported under a different name. Consider removing these wrappers and importing directly.

Similarly, lines 591-601:
```python
def _is_zarr_path(path): return _io_is_zarr_path(path)
def _load_input_dataarray(src_path, chunk): return _io_load_input_dataarray(src_path, chunk)
def _write_zarr_output(data, dst, show_progress=True): ...
```

These are trivial proxies for `dask_io.py`. They were likely introduced during a refactor. Consider using the imports directly.

---

### U-2 [MEDIUM] `dask_processor.py:188-190` — `determine_optimal_radii` wrapper

```python
def determine_optimal_radii(terrain_stats):
    return determine_optimal_radii_shared(terrain_stats)
```

This is a single-line proxy for the shared `auto_params.determine_optimal_radii`. The callers could import the shared function directly.

---

### U-3 [MEDIUM] ~~`dask_shared.py` — 3,264-line monolith~~ **[RESOLVED]**

**Status**: ✅ Resolved by Phase 1–3 refactoring (2026-02-15).

The `dask_shared.py` file has been reduced from **3,264 lines** to **177 lines** (a 94.6% reduction). It is now a pure re-export hub for backward compatibility. All algorithm implementations have been moved to individual modules:

- **4 shared utility modules**: `_base.py`, `_nan_utils.py`, `_global_stats.py`, `_normalization.py`
- **14 algorithm implementation modules**: `_impl_rvi.py`, `_impl_hillshade.py`, `_impl_slope.py`, `_impl_specular.py`, `_impl_atmospheric_scattering.py`, `_impl_multiscale_terrain.py`, `_impl_curvature.py`, `_impl_visual_saliency.py`, `_impl_npr_edges.py`, `_impl_ambient_occlusion.py`, `_impl_lrm.py`, `_impl_openness.py`, `_impl_fractal_anomaly.py`, `_impl_experimental.py`

---

### U-4 [LOW] `tile_shared.py` imports `kernel_scale_space_surprise` and `kernel_multi_light_uncertainty`

These two kernel-based tile algorithms are direct implementations, while all other tile algorithms go through the `DaskSharedTileAdapter` bridge. This inconsistency means tile_shared.py has its own `ScaleSpaceSurpriseAlgorithm` and `MultiLightUncertaintyAlgorithm`, while the rest are bridged. This is *intentional* (documented: "shared kernels delegate directly") but worth noting for consistency.

---

### U-5 [LOW] `analyze_terrain_characteristics` FFT code path (`include_fft=True`)

In `dask_processor.py:98-185`, the FFT terrain analysis code path is never called with `include_fft=True` in production. The only call site is:
```python
terrain_stats = analyze_terrain_characteristics(gpu_arr, sample_ratio=0.01, include_fft=False)
```

The `dominant_scales` and `dominant_freqs` keys are populated but never read by `determine_optimal_radii`. This entire FFT branch is dead code.

---

### U-6 [LOW] `dask_processor.py:780-782` — `agg == "stack"` branch

```python
if agg == "stack" and 'sigmas' in params and params['sigmas'] is not None:
    dims = ("scale", *dem.dims)
    coords = {"scale": params['sigmas'], **dem.coords}
```

No algorithm currently produces stacked multi-scale output in the Dask path. The `sigmas` parameter is a legacy from an earlier design. This branch may never execute.

---

## ⚡ Efficiency Issues

### E-1 [MEDIUM] `tile_processor.py:1919-1928` — All tiles submitted simultaneously

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_tile = {
        executor.submit(process_single_tile, ...): tile_info
        for tile_info in tile_infos
    }
```

**Problem**: All tile futures are submitted at once in a dict comprehension. For large rasters (e.g., 50,000+ tiles), this creates all future objects immediately, consuming significant memory. Dask's own tile pipeline uses streaming, but the ThreadPoolExecutor version doesn't.

**Recommendation**: Use `itertools.islice` or a semaphore-based pattern to limit the number of pending futures:
```python
# Submit in batches of max_workers * 2
batch_size = max_workers * 2
for batch in _iter_chunks(tile_infos, batch_size):
    futures = {executor.submit(...): info for info in batch}
    for future in as_completed(futures):
        ...
```

---

### E-2 [MEDIUM] `dask_processor.py:300-304` — CuPy memory pool check at startup

```python
available_memory = cp.get_default_memory_pool().free_bytes() / (1024**3)
if available_memory < 10:
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
```

**Problem**: `free_bytes()` reports bytes in the CuPy pool that are currently allocated but freed back to the pool (recyclable). This is **not** the same as free VRAM. The heuristic `< 10 GB` could trigger unnecessary cleanup. Use `cp.cuda.runtime.memGetInfo()` for actual free VRAM.

---

### E-3 [MEDIUM] `cog_validator.py` — `print()` for all output

All output in `_validate_cog_for_qgis` uses `print()` instead of `logging`. This means validation output always goes to stdout and can't be filtered by log level. Given that validation runs after every COG generation, this creates noise.

**Recommendation**: Replace `print()` with `logger.info()` / `logger.warning()`.

---

### E-4 [LOW] `cog_builder.py` — Same pattern: `print()` for output

Similarly, `cog_builder.py` uses `print()` for progress reporting (lines 160, 175, 278, etc.) instead of using the logging framework. This is inconsistent with the rest of the codebase.

---

### E-5 [LOW] `raster_info.py:57-61` — `print()` in utility function

```python
def detect_pixel_size_from_cog(input_cog_path: str) -> float:
    ...
    print(f"Geographic CRS: center latitude {lat_center:.3f} deg")
    print(f"Converted meters: {abs(scale_x):.3f}m x {abs(scale_y):.3f}m")
```

A utility function shouldn't have print side-effects. Use logging or return the metadata for the caller to log.

---

## 📝 Architecture / Documentation Drift

### A-1 [MEDIUM] `ARCHITECTURE.md` — Missing `multi_light_uncertainty` algorithm

The `ARCHITECTURE.md` "Dask Algorithm Catalog" section (reviewed earlier) does not list `multi_light_uncertainty` as an algorithm, yet it's fully implemented in both `dask_registry.py` (line 19) and `tile_shared.py`. The documentation should be updated.

---

### A-2 [LOW] `ARCHITECTURE.md` — Does not document COG backend selection

The architecture document describes dual pipelines (Dask vs Tile) but doesn't mention the `cog_backend` parameter (`internal` | `external` | `auto`) that allows using external GDAL CLI tools for COG generation. This is a significant operational feature worth documenting.

---

## Summary of Recommended Actions

### Priority 1 (Fix Now)
1. **B-1**: Replace hardcoded `/tmp` with `tempfile.gettempdir()` in `dask_cluster.py`
2. **B-2**: Fix brittle GDAL version parsing in `dask_processor.py`
3. **B-3**: Fix scale analysis sample reshaping to preserve spatial structure

### Priority 2 (Fix Soon)
4. **E-1**: Implement batched tile submission to reduce memory pressure
5. **E-3/E-4/E-5**: Replace `print()` with `logging` in `cog_validator.py`, `cog_builder.py`, `raster_info.py`
6. **B-5**: Rename hardcoded `rvi_tiles.vrt` to generic name
7. **D-4**: Document scipy's optional status or add to core deps
8. **A-1**: Update ARCHITECTURE.md to include `multi_light_uncertainty`

### Priority 3 (Nice to Have)
9. **U-1/U-2**: Remove trivial wrapper functions in `dask_processor.py`
10. **U-5**: Remove or flag dead FFT analysis code
11. ~~**U-3**: Consider splitting 3,264-line `dask_shared.py` monolith~~ **[RESOLVED]**
12. **A-2**: Document COG backend selection in ARCHITECTURE.md

---

## Positive Observations

1. **Excellent shared kernel design**: The `algorithms/common/kernels.py` + `tile/dask_bridge.py` pattern elegantly shares GPU code between Dask and tile backends without duplication.

2. **Robust NoData handling**: Multiple layers of NoData detection (metadata, border inference, implicit candidate warning) provide excellent user experience.

3. **Dynamic auto-tuning**: The `auto_tune.py` anchor-point interpolation system is a sophisticated and maintainable approach to GPU optimization that avoids per-GPU presets.

4. **Geographic DEM support**: Per-tile latitude-aware meter conversion in `process_single_tile` is a technically impressive feature that handles anisotropic pixel scales correctly.

5. **Error resilience**: The tile pipeline's error-per-tile tracking, COG-only resume mode, and writable tmp dir fallback provide excellent operational robustness.

6. **Consistent naming**: Algorithm names are consistent across `dask_registry.py`, `DEFAULT_ALGORITHMS` in `tile_processor.py`, and the `tile/` + `dask/` module names.
