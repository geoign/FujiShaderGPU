"""Dask cluster setup helpers."""
from __future__ import annotations

import logging
import sys
from typing import Tuple

import GPUtil
import cupy as cp

from ..config.auto_tune import auto_tune, compute_chunk_size, compute_rmm_pool_gb
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
    rmm_size = min(tuned["rmm_pool_size_gb"], int(available_gb * tuned["rmm_pool_fraction"]))

    dask_config.set({
        'distributed.worker.memory.target': 0.70,
        'distributed.worker.memory.spill': 0.75,
        'distributed.worker.memory.pause': 0.85,
        'distributed.worker.memory.terminate': 0.95,
        'distributed.admin.tick.limit': '15s',
    })

    logging.getLogger('distributed.core').setLevel(logging.WARNING)

    cluster = LocalCUDACluster(
        device_memory_limit=max(0.1, min(float(memory_fraction), 0.95)),
        jit_unspill=True,
        rmm_pool_size=f'{rmm_size}GB',
        threads_per_worker=1,
        silence_logs=logging.WARNING,
        death_timeout='60s' if is_colab else '30s',
        interface='lo' if is_colab else None,
        rmm_maximum_pool_size=f'{int(rmm_size * 1.2)}GB',
        enable_cudf_spill=True,
        local_directory='/tmp',
    )

    client = Client(cluster)
    return cluster, client
