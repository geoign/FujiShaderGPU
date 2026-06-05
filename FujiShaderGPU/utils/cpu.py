"""Container-aware CPU budget detection.

``os.cpu_count()`` / ``multiprocessing.cpu_count()`` report the **host** core
count.  Inside a container with a CFS CPU quota (RunPod / Colab / k8s / Docker
``--cpus``) the process is throttled to a fraction of those cores even though
``nproc`` and the cpuset still show all of them.  Sizing a worker pool from the
host count then oversubscribes the real budget: dozens of runnable processes
contend for a handful of effective cores, thrashing caches and context switches
and running *slower* than a right-sized pool.

``container_cpu_count()`` returns the effective core budget: the minimum of the
host count, the cpuset allowance, and the CFS-quota-derived core count.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# cgroup v1 (legacy) and v2 (unified) CPU-accounting files.
_CFS_QUOTA_V1 = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CFS_PERIOD_V1 = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_CPU_MAX_V2 = "/sys/fs/cgroup/cpu.max"
_CPUSET_V1 = "/sys/fs/cgroup/cpuset/cpuset.cpus"
_CPUSET_V2 = "/sys/fs/cgroup/cpuset.cpus.effective"


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _cfs_quota_cores() -> Optional[float]:
    """Effective cores from the CFS quota (v2 first, then v1), or None if unset."""
    # cgroup v2: "<quota> <period>" or "max <period>".
    v2 = _read_text(_CPU_MAX_V2)
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = float(parts[0]), float(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass

    # cgroup v1: separate quota / period files; quota == -1 means unlimited.
    q = _read_text(_CFS_QUOTA_V1)
    p = _read_text(_CFS_PERIOD_V1)
    if q is not None and p is not None:
        try:
            quota, period = float(q), float(p)
            if quota > 0 and period > 0:
                return quota / period
        except ValueError:
            pass
    return None


def _cpuset_count() -> Optional[int]:
    """Number of CPUs allowed by the cpuset (e.g. '0-7,16' -> 9), or None."""
    text = _read_text(_CPUSET_V2) or _read_text(_CPUSET_V1)
    if not text:
        return None
    total = 0
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                total += int(hi) - int(lo) + 1
            except ValueError:
                return None
        else:
            total += 1
    return total or None


def container_cpu_count() -> int:
    """Effective usable core count for this container (>= 1).

    ``min(host cores, cpuset count, round(CFS quota))`` -- the tightest binding
    limit.  The CFS quota is rounded to the nearest whole core (e.g. 6.8 -> 7) so
    a CPU-bound pool stays just full enough to hide I/O latency without
    oversubscribing.
    """
    host = os.cpu_count() or 1
    candidates = [host]

    cpuset = _cpuset_count()
    if cpuset:
        candidates.append(cpuset)

    quota_cores = _cfs_quota_cores()
    if quota_cores:
        candidates.append(max(1, round(quota_cores)))

    return max(1, min(candidates))
