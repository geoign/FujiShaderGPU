# Architecture Decision Record: Consolidating onto the tile backend (tile vs dask-cuda)

- **Status**: Under discussion (undecided)
- **Created**: 2026-06-01
- **Scope**: Whether FujiShaderGPU's two backends (Linux: dask-cuda / Windows/macOS:
  tile) should eventually be consolidated onto the tile approach.
- **Trigger**: Discussion around multi-GPU acceleration (e.g. RTX 3090 × 8) and the
  write-side bottleneck / heavy dependency stack of the dask-cuda implementation.

---

## 1. Background and questions

FujiShaderGPU currently has two processing backends.

- **dask-cuda path (Linux)**: `LocalCUDACluster` (1 GPU = 1 worker) + lazy graph
  (`map_overlap` for halo-padded compute) → COG/Zarr output.
- **tile path (Windows/macOS)**: a `ThreadPoolExecutor` where each tile reads its own
  window (core + padding) from the COG, computes on the GPU, and writes a tile
  GeoTIFF → VRT+COG consolidation.

Questions for discussion:

1. Is the tile approach architecturally better suited to multi-GPU?
2. If so, should we consolidate onto the tile approach?
3. Are there situations where dask-cuda has a large advantage over the tile approach?

---

## 2. Multi-GPU suitability: the tile approach is inherently favorable

The tile approach is the canonical "independent-tile parallelism" pattern for
large-raster processing, and it satisfies the following.

1. **Tiles are fully independent**: each tile reads its own window (core + halo)
   directly from the COG, so **no inter-GPU halo exchange is needed**.
   - By contrast, dask-cuda's `map_overlap` creates chunk-boundary dependencies, and
     halos are transferred between GPUs via the host (`dask_cluster.py` already
     disables P2P rechunk as a non-HMM workaround = transfer overhead).
2. **Distributed writes**: each worker writes its own tile GeoTIFF independently, and
   only the final VRT+COG step consolidates them.
   - By contrast, dask-cuda's large-output path gathers compute results to the client
     and writes them sequentially (`_write_cog_da_chunked_impl`). This becomes the
     **effective ceiling on multi-GPU scaling**.
3. **Trivial work assignment**: just round-robin tiles across GPUs. No scheduler,
   spill, or RMM-pool tuning required.
4. **Naturally extends to multi-node**: each node reads tiles from a shared/cloud COG,
   writes tiles to shared storage, and consolidates at the end — simpler than
   dask-cuda multi-node.

Key point: the tile approach takes the trade-off of "**re-reading halos redundantly
from disk instead of communicating**", which is the source of its scalability. The
overview optimizations added recently make halos small, so the cost of redundant
reads is also small.

---

## 3. Accurate picture of the current state (important caveats)

Facts to avoid mistaking consolidation for a "simple win".

1. **The current tile path also depends on dask.** `DaskSharedTileAdapter` turns a
   single tile into a single-chunk dask array and runs `map_overlap` on it. Fully
   removing dask requires replacing the bridge with "direct calls to the cupy block
   functions (`compute_*_block`) + manual window-padding handling" (technically
   straightforward, since the core compute is already cupy functions).
2. **The current tile path is single-GPU.** All threads of the `ThreadPoolExecutor`
   share the default device. The cleanest path to multi-GPU is **1 GPU = 1 process**
   (pin each process with `CUDA_VISIBLE_DEVICES` and split the tile set), which avoids
   cupy's "1 process = 1 default device" and the GIL constraint.
3. Features present in the dask path but not in the tile path: **Zarr I/O** and
   "streaming beyond VRAM".

---

## 4. Situations where dask-cuda is inherently superior

The tile approach's strength (tile independence) is, flipped around, the weakness of
"**cannot handle coupling that crosses tile boundaries**". That is dask-cuda's
(distributed-array model) domain of advantage.

### 4.1 Globally-coupled algorithms (most important; specific to terrain tools)

Independent tiles **cannot correctly compute processes that propagate across the whole
raster** (always discontinuous/wrong at boundaries).

- **Hydrology**: flow direction / **flow accumulation** / watershed /
  **depression filling** (priority-flood) / channel-network extraction.
  Flow depends on the entire upstream area of the DEM, and the upstream area spans
  tiles over tens of km → impossible.
- **Visibility**: long-range viewshed / cumulative viewshed.
- **Cost distance / least-cost path**, full-extent **connected-component labeling**,
  true global statistics/propagation.

These do not close under a local halo and require whole-extent data movement /
iterative propagation, so dask (or a dedicated global algorithm) is needed.
**All of FujiShaderGPU's current algorithms are local** (hillshade/TopoUSM Fast/openness etc. =
finite halo), so this does not apply today, but it becomes **a decisive branch point
if hydrology/visibility is added in the future**.

### 4.2 Out-of-core (spill) safety net

In pathological cases where one unit (chunk + halo, or a huge intermediate) exceeds
VRAM/RAM, dask-cuda automatically spills device→host→disk and can finish without
crashing. The tile approach is designed to fit a tile in VRAM so this is usually
unnecessary, but dask is more robust against unexpectedly large halos/intermediates.

### 4.3 Dynamic load balancing / fault tolerance (at scale)

- When tile cost is non-uniform (skip on heavy NoData vs heavy regions), distributed's
  work-stealing rebalances automatically (the tile path's static assignment skews
  easily; a shared queue can mitigate it).
- Task retries on worker death, nanny restarts. Fault tolerance for multi-node,
  long-running jobs.

### 4.4 Lazy-graph optimization / rechunk / ecosystem

- Multi-stage pipeline fusion / non-materialization of intermediates, whole-extent
  rechunk/transpose.
- Mature integration with **xarray / Zarr** (N-dimensional, time series, lazy eval).

> Note: even for global processing, generic dask is not always fastest (priority-flood
> etc. favor dedicated streaming/external-memory implementations). dask's real value is
> that it makes "**generic global array operations relatively easy to write**".

---

## 5. Mapping to the current workload

| Aspect | Current (all algorithms local + COG-centric) | Does dask's advantage apply? |
|---|---|---|
| hillshade / TopoUSM Fast / openness / AO / curvature etc. | Closes under a local halo | ✗ (tile is sufficient — if anything, superior) |
| Hydrology / visibility / cost distance | **Not implemented** | ◎ (if added later, dask or a dedicated global impl is required) |
| Pathological beyond-VRAM cases | Avoided via tile size | △ (dask is a safety net) |
| Zarr / multi-band / time series | COG-centric | ○ (dask if needed) |
| Multi-node / fault tolerance | Single-node operation | ○ (when scaling up) |
| Multi-GPU scaling | — | ✗ (**tile is favorable**: independent, distributed writes, communication-free) |

---

## 6. Pros / cons of consolidation

### Pros (consolidating onto tile)

- **Major dependency-stack simplification**: potentially remove dask-cuda / cudf-cu12 /
  rmm-cu12 / distributed. This is the heaviest and most fragile part of the Runpod
  setup (installing cudf/rmm from the NVIDIA index), so a big operational improvement.
- **Structurally** free of sequential writes, inter-GPU transfer, and spill complexity.
- **Single code path**: resolves the current "dask implementation + tile bridge" dual
  structure.
- Natural extension to multi-GPU / multi-node.

### Cons / risks

- If globally-coupled algorithms (hydrology, visibility, etc.) are pursued later, they
  cannot be implemented in the tile approach alone.
- The current tile path also depends on dask and is single-GPU, so consolidation
  carries non-trivial implementation and regression risk.
- Zarr I/O / beyond-VRAM streaming would need to be reimplemented on the tile side
  (if needed).
- Both paths currently work, so there is little reason to rush.

---

## 7. Decision axis: depends on the algorithm roadmap

- **Mostly local algorithms for the foreseeable future** → consolidating onto tile is
  fine (simple, high-scale). The downside of removing dask is small.
- **Plans to add global processing (hydrology, visibility, etc.) in the future** →
  those parts cannot be implemented in tile, so it is reasonable to **keep dask**, or
  to provide **a dedicated implementation just for that global processing** (parallel
  priority-flood, streaming flow-accumulation, etc.) while keeping local processing on
  tile — a **hybrid**.

---

## 8. Recommendation: a phased approach

A single big-bang replacement is high-risk, so a phased rollout is recommended.

- **Step 0 (measure first)**: with the recent optimizations (preprocessing COG-ization
  / decoupled hole filling / large-radius-from-overview / halo reduction / parallel
  writes), **a single GPU may already be fast enough**. Measure first on a single GPU
  (A100 80GB or RTX 4090) to determine whether multi-GPU is really needed.
- **Step 1 (high ROI, low risk)**: add "**1 GPU = 1 process**" multi-GPU execution to
  the tile approach. Split tiles across GPUs; each process reads / computes / writes its
  own tiles, then consolidate at the end. Keep the dask path for now. This alone yields
  independent-tile + distributed-write multi-GPU scaling.
- **Step 2 (simplification, medium risk)**: replace the bridge with direct cupy calls,
  remove the dask-cuda dependency, and make tile the single official backend. Use tile
  on Linux too. The dependency stack shrinks dramatically.
- **(Conditional) global-algorithm support**: only if hydrology/visibility is
  introduced, provide dask or a dedicated global implementation for that feature
  separately (hybrid).

---

## 9. Conclusion (as of now)

- **Multi-GPU/multi-node suitability is clearly higher for the tile approach**
  (independence, distributed writes, communication-free).
- For the **current local-algorithm + COG + single/few-GPU** use case, dask-cuda has no
  large advantage, and consolidating onto tile is strategically reasonable (gains both
  simplification/robustness and scaling).
- However, **dask-cuda is inherently superior for globally-coupled algorithms
  (especially hydrology/visibility)**, and whether those are pursued in the future is
  the biggest decision axis for consolidation.
- If we proceed, **a phased rollout** is safest (① add per-process-GPU multi-GPU to
  tile → ② remove dask and consolidate, keeping a hybrid for global processing only
  when needed).
- **Measure on a single GPU first** to confirm whether multi-GPU is still needed after
  the optimizations.

---

## 10. Open items / next actions

- [ ] Single-GPU measurement (A100 80GB / RTX 4090): wall-clock time and bottleneck
  (compute vs write/I-O).
- [ ] Confirm the algorithm roadmap (whether to add global processing such as
  hydrology/visibility).
- [ ] (If proceeding) design/implement Step 1 "per-process-GPU multi-GPU execution on
  tile".
- [ ] (If proceeding) a migration plan for Step 2 "direct cupy bridge / dask-cuda
  removal".
