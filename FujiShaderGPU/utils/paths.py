"""Filesystem path helpers shared across backends."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

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
