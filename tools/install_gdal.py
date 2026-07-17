#!/usr/bin/env python3
"""
FujiShaderGPU GDAL bootstrapper.

Installs the GDAL Python bindings (``osgeo``) into the *active* Python
environment, matched to the native GDAL library, then verifies them.  GDAL is
the step people most often get stuck on; this script auto-detects the
environment and picks the most reliable method:

    conda env            -> conda/mamba install -c conda-forge gdal
    native GDAL present  -> pip build of the exact matching binding
    Debian/Ubuntu (apt)  -> apt install libgdal-dev gdal-bin, then pip build
    macOS (Homebrew)     -> brew install gdal, then pip build
    Windows (no conda)   -> print OSGeo4W / conda guidance (cannot auto-install)

Standalone: depends only on the Python standard library, so it runs *before*
FujiShaderGPU (or its dependencies) are importable.

Usage:
    python tools/install_gdal.py            # detect + install + verify
    python tools/install_gdal.py --dry-run  # show what it would do
    python tools/install_gdal.py --yes      # don't prompt before apt/brew/conda
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys


def _run(cmd: list[str], *, env: dict | None = None, dry: bool = False) -> int:
    print("  $ " + " ".join(cmd))
    if dry:
        return 0
    try:
        return subprocess.call(cmd, env=env)
    except OSError as exc:
        print(f"  command could not be started: {exc}")
        return 127


def _verify() -> str | None:
    """Return the GDAL version string if `from osgeo import gdal` works, else None."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", "from osgeo import gdal; print(gdal.__version__)"],
            stderr=subprocess.STDOUT, text=True,
        )
        lines = out.strip().splitlines()
        return lines[-1].strip() if lines and lines[-1].strip() else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _gdal_config_version() -> str | None:
    gc = shutil.which("gdal-config")
    if not gc:
        return None
    try:
        return subprocess.check_output([gc, "--version"], text=True).strip()
    except Exception:
        return None


def _gdal_include_dir() -> str | None:
    gc = shutil.which("gdal-config")
    if gc:
        try:
            cflags = subprocess.check_output([gc, "--cflags"], text=True).strip()
            for tok in cflags.split():
                if tok.startswith("-I"):
                    return tok[2:]
        except Exception:
            pass
    for cand in ("/usr/include/gdal", "/usr/local/include/gdal"):
        if os.path.isdir(cand):
            return cand
    return None


def _in_conda() -> bool:
    # CONDA_PREFIX can leak into a subprocess that is using a different Python.
    # Only the active interpreter's prefix proves that this interpreter is conda-managed.
    return os.path.isdir(os.path.join(sys.prefix, "conda-meta"))


def _sudo_prefix() -> list[str]:
    # Root needs no sudo; otherwise use sudo if available.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo"]
    raise RuntimeError(
        "apt-get requires root privileges, but this process is not root and `sudo` is unavailable"
    )


def _pip_build_binding(version: str | None, *, dry: bool) -> bool:
    """Build/install the osgeo binding for the ACTIVE interpreter, matched to the
    native GDAL version.  Force-reinstall so a distro python3-gdal built for a
    *different* Python version is replaced (the classic 'No module named _gdal')."""
    env = os.environ.copy()
    inc = _gdal_include_dir()
    if inc:
        for var in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
            env[var] = inc + (os.pathsep + env[var] if env.get(var) else "")
    # numpy headers let the build include osgeo.gdal_array (used by the pipeline).
    if _run([sys.executable, "-m", "pip", "install", "numpy"], env=env, dry=dry) != 0:
        print("  installing the NumPy build dependency failed")
        return False
    spec = f"GDAL=={version}" if version else "GDAL"
    base = [
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
        "--force-reinstall", "--no-cache-dir",
    ]
    if _run(base + [spec], env=env, dry=dry) == 0:
        return True
    # Fallback: compatible-release if the exact pin is not on PyPI.
    if version:
        mm = ".".join(version.split(".")[:2])
        print(f"  exact pin failed; trying compatible release GDAL~={mm}.0")
        return _run(base + [f"GDAL~={mm}.0"], env=env, dry=dry) == 0
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Install/verify GDAL for FujiShaderGPU.")
    ap.add_argument("--dry-run", action="store_true", help="print actions without running them")
    ap.add_argument("--yes", "-y", action="store_true", help="assume yes for system package installs")
    args = ap.parse_args()
    dry = args.dry_run

    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # A dry run must describe actions only; even verification imports can have
    # environment-dependent side effects and would make the plan misleading.
    existing = None if dry else _verify()
    if existing:
        print(f"[OK] GDAL bindings already work: osgeo {existing} -- nothing to do.")
        return 0

    system = platform.system().lower()

    def confirm(what: str) -> bool:
        if args.yes or dry:
            return True
        try:
            return input(f"Proceed to {what}? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    # 1) conda environment -> conda-forge gdal (most reliable; native + binding together)
    if _in_conda():
        tool = shutil.which("mamba") or shutil.which("conda")
        if tool:
            print("Detected conda environment.")
            if confirm(f"run `{os.path.basename(tool)} install -c conda-forge gdal`"):
                rc = _run([tool, "install", "-y", "-c", "conda-forge", "gdal"], dry=dry)
                if rc != 0:
                    print(f"[!] conda install failed with exit code {rc}.")
                    return 1
                if dry:
                    print("[DRY RUN] Conda installation would be followed by verification.")
                    return 0
                v = _verify()
                if v:
                    print(f"[OK] GDAL installed via conda-forge: osgeo {v}")
                    return 0
                print("[!] conda install did not yield working osgeo bindings.")
        else:
            print("[!] conda env detected but neither `conda` nor `mamba` is on PATH.")

    # 2) native GDAL already present (gdal-config) -> just build the matching binding
    ver = _gdal_config_version()
    if ver:
        print(f"Detected native GDAL {ver} (gdal-config). Building the matching Python binding.")
        if _pip_build_binding(ver, dry=dry):
            if dry:
                print("[DRY RUN] The binding build would be followed by verification.")
                return 0
            v = _verify()
            if v:
                print(f"[OK] GDAL bindings installed: osgeo {v}")
                return 0
        print("[!] Building the binding against the existing native GDAL failed.")

    # 3) Debian/Ubuntu -> apt install native GDAL, then build the binding
    if system == "linux" and shutil.which("apt-get"):
        print("Detected Debian/Ubuntu (apt). Installing native GDAL, then the binding.")
        if confirm("install libgdal-dev + gdal-bin via apt-get"):
            try:
                sudo = _sudo_prefix()
            except RuntimeError as exc:
                print(f"[FAILED] {exc}")
                return 2
            rc = _run(sudo + ["apt-get", "update"], dry=dry)
            if rc != 0:
                print(f"[FAILED] apt-get update failed with exit code {rc}.")
                return 1
            rc = _run(
                sudo + ["apt-get", "install", "-y", "libgdal-dev", "gdal-bin"],
                dry=dry,
            )
            if rc != 0:
                print(f"[FAILED] apt-get install failed with exit code {rc}.")
                return 1
            ver = _gdal_config_version()
            if _pip_build_binding(ver, dry=dry):
                if dry:
                    print("[DRY RUN] apt and pip installation would be followed by verification.")
                    return 0
                v = _verify()
                if v:
                    print(f"[OK] GDAL installed (apt + pip): osgeo {v}")
                    return 0
            print("[!] apt + pip build did not yield working bindings.")

    # 4) macOS Homebrew
    elif system == "darwin" and shutil.which("brew"):
        print("Detected macOS Homebrew. Installing native GDAL, then the binding.")
        if confirm("install gdal via Homebrew"):
            rc = _run(["brew", "install", "gdal"], dry=dry)
            if rc != 0:
                print(f"[FAILED] brew install failed with exit code {rc}.")
                return 1
            if _pip_build_binding(_gdal_config_version(), dry=dry):
                if dry:
                    print("[DRY RUN] Homebrew and pip installation would be followed by verification.")
                    return 0
                v = _verify()
                if v:
                    print(f"[OK] GDAL installed (brew + pip): osgeo {v}")
                    return 0

    # 5) Windows without conda -> cannot reliably auto-install
    elif system == "windows":
        print(
            "\nNo conda environment detected. On Windows the reliable options are:\n"
            "  * conda:    conda install -c conda-forge gdal   (recommended)\n"
            "  * OSGeo4W:  install the 'gdal' and 'python3-gdal' packages and run\n"
            "              FujiShaderGPU from the OSGeo4W shell.\n"
            "  * a prebuilt GDAL wheel matching your Python version and native GDAL.\n"
            "Then re-run this script to verify."
        )
        return 2

    print(
        "\n[FAILED] Could not install working GDAL bindings automatically.\n"
        "See the 'Installing GDAL' section of the README and install manually, then\n"
        "verify with:  python -c \"from osgeo import gdal; print(gdal.__version__)\""
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
