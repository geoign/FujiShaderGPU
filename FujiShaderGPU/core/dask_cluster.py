"""Dask cluster setup helpers."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Tuple

import GPUtil
import cupy as cp

from ..config.auto_tune import auto_tune, compute_chunk_size
from dask import config as dask_config
from dask_cuda import LocalCUDACluster
from distributed import Client

logger = logging.getLogger(__name__)


def get_optimal_chunk_size(gpu_memory_gb: float = 40) -> int:
    """Auto-tuned chunk size from VRAM (anchor-interpolated)."""
    return compute_chunk_size(max(gpu_memory_gb, 4.0))


def make_cluster(memory_fraction: float = None) -> Tuple[LocalCUDACluster, Client]:
    is_colab = 'google.colab' in sys.modules

    gpus = GPUtil.getGPUs()
    gpu_memory_gb = (gpus[0].memoryTotal / 1024) if gpus else 16

    try:
        meminfo = cp.cuda.runtime.memGetInfo()
        available_gb = meminfo[0] / (1024**3)
    except Exception:
        available_gb = gpu_memory_gb * 0.8

    tuned = auto_tune(gpu_memory_gb, is_colab=is_colab)
    if memory_fraction is None:
        memory_fraction = tuned["memory_fraction"]
    if is_colab:
        memory_fraction = min(memory_fraction, 0.50)
        logger.info('Google Colab environment detected')
    device_memory_fraction = max(0.1, min(float(memory_fraction), 0.95))
    device_limit_gb = max(1.0, min(gpu_memory_gb, available_gb) * device_memory_fraction)
    # Keep the eager RMM pool below Dask's device_memory_limit.  A pool larger
    # than the worker limit can stall or restart the nanny during startup.
    rmm_cap_gb = max(1, int(device_limit_gb * 0.80))
    rmm_size = min(
        tuned["rmm_pool_size_gb"],
        int(available_gb * tuned["rmm_pool_fraction"]),
        rmm_cap_gb,
    )
    # Let the pool grow to nearly the full device budget.  Capping it at
    # rmm_size*1.2 left only ~3GB of headroom above the eager pool, so the
    # combine-heavy algorithms (visual_saliency / fractal_anomaly) -- whose many
    # small alloc/free cycles fragment the pool -- ran out of room even at tiny
    # chunks.  Staying just under device_memory_limit lets dask spill instead of
    # hard-failing the RMM pool.
    rmm_max_gb = max(rmm_size + 1, int(device_limit_gb * 0.97))
    spill_dir = os.environ.get('FUJISHADER_SPILL_DIR', tempfile.gettempdir())
    logger.info("Dask spill directory: %s", spill_dir)
    logger.info(
        "Dask CUDA memory: total=%.1fGB available=%.1fGB device_limit=%.1fGB "
        "rmm_pool=%dGB rmm_max=%dGB",
        gpu_memory_gb,
        available_gb,
        device_limit_gb,
        rmm_size,
        rmm_max_gb,
    )

    dask_config.set({
        # dask-cuda P2P rechunk/shuffle may inspect CuPy buffers from CPU and
        # fail on non-HMM systems.  The task-based rechunk path is slower but
        # robust for FujiShaderGPU's single-GPU Runpod/Colab workloads.
        'array.rechunk.method': 'tasks',
        'distributed.worker.memory.target': 0.70,
        'distributed.worker.memory.spill': 0.75,
        'distributed.worker.memory.pause': 0.85,
        'distributed.worker.memory.terminate': 0.95,
        'distributed.admin.tick.limit': '15s',
    })

    logging.getLogger('distributed.core').setLevel(logging.WARNING)

    cluster_kwargs = {
        'device_memory_limit': device_memory_fraction,
        'rmm_pool_size': f'{rmm_size}GB',
        'threads_per_worker': 1,
        'silence_logs': logging.WARNING,
        'death_timeout': '60s' if is_colab else '30s',
        'interface': 'lo' if is_colab else None,
        'rmm_maximum_pool_size': f'{rmm_max_gb}GB',
        'enable_cudf_spill': True,
        'local_directory': spill_dir,
    }

    try:
        cluster = LocalCUDACluster(**cluster_kwargs)
    except TypeError as exc:
        if 'enable_cudf_spill' not in str(exc):
            raise
        # Older dask-cuda releases do not expose cuDF native spilling.
        cluster_kwargs.pop('enable_cudf_spill', None)
        logger.info("dask-cuda does not support enable_cudf_spill; using default spilling")
        cluster = LocalCUDACluster(**cluster_kwargs)

    client = Client(cluster)
    return cluster, client
