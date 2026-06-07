"""Filesystem path helpers shared across backends."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Environment variables, in precedence order, that redirect large temporary
# staging files away from a small default ``/tmp``.  ``FUJISHADER_TMP_DIR`` is
# the project-specific override; the rest are the conventional GDAL / POSIX
# temp-dir variables.  Kept in sync with the chunked-write path in
# ``core.dask_processor._select_chunk_temp_parent``.
TMP_DIR_ENV_VARS: Tuple[str, ...] = (
    "FUJISHADER_TMP_DIR",
    "CPL_TMPDIR",
    "TMPDIR",
    "TMP",
    "TEMP",
)


def resolve_tmp_dir(default: Path) -> Tuple[Path, Optional[str]]:
    """Resolve the directory for large temporary staging files.

    Returns ``(directory, source_env_var)`` where ``source_env_var`` is the name
    of the environment variable that supplied the path, or ``None`` when none was
    set and ``default`` was used.  The returned directory is created if needed.
    """
    selected_from: Optional[str] = None
    chosen = default
    for env_name in TMP_DIR_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            selected_from = env_name
            chosen = Path(value)
            break

    chosen.mkdir(parents=True, exist_ok=True)
    return chosen, selected_from


def safe_abspath(path: Union[str, "os.PathLike[str]"]) -> Path:
    """Absolute, normalised path *without* opening a filesystem handle.

    ``Path.resolve()`` on Windows calls ``GetFinalPathNameByHandle``, which some
    virtual filesystems do not implement -- notably rclone FUSE mounts, which
    raise ``OSError: [WinError 1005] The volume does not contain a recognized
    file system`` even though ordinary reads/writes on the mount succeed.  The
    callers here only need an absolute path (to derive a sibling staging
    directory), so use ``os.path.abspath`` -- pure string + cwd work that never
    touches the volume.  Symlinks are intentionally *not* resolved.
    """
    return Path(os.path.abspath(os.fspath(path)))


def safe_unlink(
    path: Union[str, "os.PathLike[str]"],
    *,
    retries: int = 5,
    delay: float = 0.2,
) -> None:
    """Best-effort delete of a temporary file; never raises.

    On Windows a just-closed GDAL/rasterio dataset can keep a file handle open
    for a brief moment after the Python object is dropped, so an immediate
    ``unlink`` raises ``PermissionError: [WinError 32] The process cannot access
    the file because it is being used by another process``.  Retry a few times
    to let the handle drain, then give up with a warning: these are temporary
    staging files, and failing to remove one must not fail an otherwise
    successful job (the OS / temp-dir sweep will reclaim it).
    """
    p = Path(path)
    for attempt in range(retries):
        try:
            p.unlink(missing_ok=True)
            return
        except OSError as exc:
            if attempt == retries - 1:
                logger.warning(
                    "Could not delete temporary file %s (%s); leaving it for "
                    "OS/temp-dir cleanup.",
                    p, exc,
                )
                return
            time.sleep(delay)
