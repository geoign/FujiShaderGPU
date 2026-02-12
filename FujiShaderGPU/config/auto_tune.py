"""
FujiShaderGPU/config/auto_tune.py

VRAM・アルゴリズム複雑度から全パフォーマンスパラメータを動的に算出する。
検証済みアンカーポイント（旧gpu_presets.yaml由来）からの補間で任意のGPUに対応。
"""

import math
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm complexity map (single source of truth)
# Higher = more VRAM per pixel, more compute.  1.0 = baseline.
# ---------------------------------------------------------------------------
ALGORITHM_COMPLEXITY: Dict[str, float] = {
    "hillshade": 0.8,
    "slope": 0.8,
    "atmospheric_scattering": 0.9,
    "curvature": 1.0,
    "lrm": 1.1,
    "npr_edges": 1.1,
    "rvi": 1.2,
    "visual_saliency": 1.4,
    "specular": 1.5,
    "multiscale_terrain": 1.5,
    "fractal_anomaly": 1.6,
    "openness": 1.8,
    "ambient_occlusion": 2.0,
}

# ---------------------------------------------------------------------------
# Calibration anchors derived from validated gpu_presets.yaml
# ---------------------------------------------------------------------------
_ANCHOR_VRAM =      [8,    12,   16,   24,   40,    80]
_ANCHOR_CHUNK =     [512,  768,  1024, 2048, 8192,  14336]
_ANCHOR_RMM_GB =    [4.0,  8.0,  12.0, 16.0, 28.0,  58.0]
_ANCHOR_RMM_FRAC =  [0.50, 0.55, 0.60, 0.65, 0.70,  0.72]


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _lerp(x: float, xs: List[float], ys: List[float]) -> float:
    """Piecewise linear interpolation with linear extrapolation."""
    if x <= xs[0]:
        if len(xs) >= 2 and xs[1] != xs[0]:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] + slope * (x - xs[0])
        return ys[0]
    if x >= xs[-1]:
        if len(xs) >= 2 and xs[-1] != xs[-2]:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (x - xs[-1])
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _log_lerp(x: float, xs: List[float], ys: List[float]) -> float:
    """Interpolation in log2 space -- suitable for power-law relationships."""
    log_xs = [math.log2(max(v, 1e-9)) for v in xs]
    log_ys = [math.log2(max(v, 1e-9)) for v in ys]
    log_result = _lerp(math.log2(max(x, 1e-9)), log_xs, log_ys)
    return 2 ** log_result


def _round_to(value: float, unit: int) -> int:
    """Round to nearest multiple of *unit*, minimum *unit*."""
    return max(unit, int(round(value / unit)) * unit)


# ---------------------------------------------------------------------------
# Individual parameter computation
# ---------------------------------------------------------------------------
def compute_chunk_size(vram_gb: float, *, algorithm: str = "") -> int:
    """Compute optimal Dask chunk / tile-base size from VRAM.

    Uses log-space interpolation of validated anchor points, then adjusts
    for algorithm complexity.  Result is rounded to a 256-pixel boundary.
    """
    raw = _log_lerp(max(vram_gb, 4.0), _ANCHOR_VRAM, _ANCHOR_CHUNK)
    complexity = ALGORITHM_COMPLEXITY.get(algorithm, 1.0)
    adjusted = raw / (complexity ** 0.4)
    return _round_to(adjusted, 256)


def compute_dask_chunk(
    vram_gb: float,
    *,
    data_gb: float = 0.0,
    algorithm: str = "",
) -> int:
    """Chunk size for the Dask pipeline -- considers both VRAM and dataset size."""
    base = compute_chunk_size(vram_gb, algorithm=algorithm)
    if data_gb > 0:
        ratio = data_gb / max(vram_gb, 1.0)
        if ratio > 10:
            base = int(base * 0.4)
        elif ratio > 5:
            base = int(base * 0.6)
        elif ratio > 2:
            base = int(base * 0.8)
    return _round_to(base, 256)


def compute_tile_size(vram_gb: float, *, algorithm: str = "") -> int:
    """Tile size for the tile pipeline (2x chunk_size)."""
    return compute_chunk_size(vram_gb, algorithm=algorithm) * 2


def compute_rmm_pool_gb(vram_gb: float) -> int:
    """RMM managed pool size in GB."""
    raw = _lerp(max(vram_gb, 4.0), _ANCHOR_VRAM, _ANCHOR_RMM_GB)
    return max(2, int(round(raw)))


def compute_rmm_fraction(vram_gb: float) -> float:
    """RMM pool fraction (0-1)."""
    raw = _lerp(max(vram_gb, 4.0), _ANCHOR_VRAM, _ANCHOR_RMM_FRAC)
    return round(min(0.85, max(0.40, raw)), 2)


def compute_memory_fraction(
    vram_gb: float, *, is_colab: bool = False
) -> float:
    """Dask device_memory_limit fraction."""
    if is_colab:
        return 0.50
    base = _lerp(max(vram_gb, 4.0), [8, 16, 40, 80], [0.60, 0.70, 0.80, 0.85])
    return round(min(0.90, max(0.50, base)), 2)


def compute_max_workers(
    vram_gb: float,
    *,
    effective_span: int = 0,
    cpu_count: int = 4,
    algorithm: str = "",
) -> int:
    """Concurrent tile-workers bounded by CPU count, VRAM, and tile footprint."""
    base_workers = min(6, cpu_count)
    if effective_span <= 0:
        return base_workers

    vram = max(vram_gb, 4.0)
    complexity = ALGORITHM_COMPLEXITY.get(algorithm, 1.0)

    # VRAM constraint: estimated peak per-tile VRAM consumption
    overhead = 15.0 * complexity
    gb_per_tile = (effective_span ** 2 * 4.0 * overhead) / (1024 ** 3)
    usable_vram = vram * 0.70
    vram_workers = max(1, int(usable_vram / max(gb_per_tile, 0.001)))

    # Throughput constraint: large tiles serialise GPU compute.
    # Scale threshold by VRAM -- larger GPUs handle bigger tiles.
    throughput_base = 1600.0 * (vram / 12.0) ** 0.3
    if effective_span >= throughput_base * 1.5:
        throughput_workers = 1
    elif effective_span >= throughput_base * 1.25:
        throughput_workers = 2
    elif effective_span >= throughput_base:
        throughput_workers = 3
    else:
        throughput_workers = base_workers

    return min(base_workers, vram_workers, throughput_workers)


def compute_batch_size(vram_gb: float) -> int:
    """Tiles to process in a micro-batch."""
    if vram_gb >= 40:
        return 2
    return 1


def compute_prefetch_tiles(vram_gb: float) -> int:
    """Tiles to prefetch for I/O overlap."""
    if vram_gb >= 40:
        return 4
    if vram_gb >= 20:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def auto_tune(
    vram_gb: float,
    *,
    algorithm: str = "",
    data_gb: float = 0.0,
    effective_span: int = 0,
    cpu_count: int = 4,
    is_colab: bool = False,
) -> Dict[str, Any]:
    """Compute all performance parameters from hardware specs.

    Parameters
    ----------
    vram_gb : float
        Total GPU VRAM in GB.
    algorithm : str
        Algorithm name (for complexity adjustment).
    data_gb : float
        Total input data size in GB (for Dask chunk sizing).
    effective_span : int
        tile_size + 2*padding in pixels (for worker throttling).
    cpu_count : int
        Available CPU cores.
    is_colab : bool
        Google Colab environment flag.

    Returns
    -------
    dict
        chunk_size, tile_size, rmm_pool_size_gb, rmm_pool_fraction,
        memory_fraction, max_workers, batch_size, prefetch_tiles,
        dask_chunk, algorithm_complexity, vram_gb
    """
    vram = max(vram_gb, 4.0)
    complexity = ALGORITHM_COMPLEXITY.get(algorithm, 1.0)

    result = {
        "chunk_size": compute_chunk_size(vram, algorithm=algorithm),
        "tile_size": compute_tile_size(vram, algorithm=algorithm),
        "rmm_pool_size_gb": compute_rmm_pool_gb(vram),
        "rmm_pool_fraction": compute_rmm_fraction(vram),
        "memory_fraction": compute_memory_fraction(vram, is_colab=is_colab),
        "max_workers": compute_max_workers(
            vram,
            effective_span=effective_span,
            cpu_count=cpu_count,
            algorithm=algorithm,
        ),
        "batch_size": compute_batch_size(vram),
        "prefetch_tiles": compute_prefetch_tiles(vram),
        "dask_chunk": compute_dask_chunk(
            vram, data_gb=data_gb, algorithm=algorithm,
        ),
        "algorithm_complexity": complexity,
        "vram_gb": vram,
    }

    logger.info(
        "auto_tune: VRAM=%.1fGB algo=%s complexity=%.1f -> "
        "chunk=%d tile=%d rmm=%dGB workers=%d",
        vram, algorithm or "(none)", complexity,
        result["chunk_size"], result["tile_size"],
        result["rmm_pool_size_gb"], result["max_workers"],
    )
    return result
