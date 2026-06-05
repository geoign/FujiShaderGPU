"""Container-aware system memory detection.

``psutil.virtual_memory()`` reports the **host** machine's RAM.  Inside a
container with a cgroup memory limit (RunPod / Colab / k8s / Docker) the process
can actually only use the cgroup cap, which is often an order of magnitude
smaller than the host total (e.g. host 1007GB but cgroup 116GB).  Sizing GDAL
caches, prefetch buffers, or write thresholds from the host total leads to
over-allocation and OOM kills (SIGKILL 9).

The helpers below read the cgroup limit (v2 first, then v1) and clamp the
psutil figures to it, so callers see the memory the process may *actually* use.
A limit of ``"max"`` / unset / an unlimited sentinel (>= host physical total)
is treated as "no container limit" and the host figure is returned unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# cgroup v2 (unified hierarchy) and v1 (legacy) memory accounting files.
_CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"
_CGROUP_V2_USAGE = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"

_GIB = 1024 ** 3


def _read_int(path: str) -> Optional[int]:
    """Read a single integer from a cgroup pseudo-file, or None on failure.

    cgroup v2 ``memory.max`` may contain the literal ``max`` (no limit), which
    is not parseable as an int and correctly returns None here.
    """
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the container's cgroup memory limit in bytes, or None if unlimited.

    cgroup v1 reports "unlimited" as a huge sentinel (``PAGE_COUNTER_MAX`` *
    page size, near INT64 max); any value at or above the host's physical RAM is
    therefore treated as "no real limit".
    """
    limit = _read_int(_CGROUP_V2_LIMIT)
    if limit is None:
        limit = _read_int(_CGROUP_V1_LIMIT)
    if limit is None or limit <= 0:
        return None

    # Reject unlimited sentinels: anything >= host physical RAM is not a
    # meaningful container cap (v1 unlimited is ~9.2e18).
    try:
        host_total = psutil.virtual_memory().total
    except Exception:
        host_total = 0
    if host_total and limit >= host_total:
        return None
    return limit


def cgroup_memory_usage_bytes() -> Optional[int]:
    """Return current cgroup memory usage in bytes, or None if unavailable."""
    usage = _read_int(_CGROUP_V2_USAGE)
    if usage is None:
        usage = _read_int(_CGROUP_V1_USAGE)
    return usage


def container_memory_total_gb() -> float:
    """Total memory the process may use (cgroup limit, clamped to host total)."""
    host_total_gb = psutil.virtual_memory().total / _GIB
    limit = cgroup_memory_limit_bytes()
    if limit is None:
        return host_total_gb
    return min(host_total_gb, limit / _GIB)


def container_memory_available_gb() -> float:
    """Memory currently available to the process, container-aware.

    Returns ``min(host available, cgroup headroom)`` where the cgroup headroom is
    ``limit - current usage``.  Falls back to the host's available figure when no
    cgroup limit is in effect.
    """
    host_avail_gb = psutil.virtual_memory().available / _GIB
    limit = cgroup_memory_limit_bytes()
    if limit is None:
        return host_avail_gb

    usage = cgroup_memory_usage_bytes()
    if usage is None:
        # Limit known but usage unreadable: cap available at the limit itself.
        return min(host_avail_gb, limit / _GIB)

    headroom_gb = max(0.0, (limit - usage) / _GIB)
    return min(host_avail_gb, headroom_gb)
