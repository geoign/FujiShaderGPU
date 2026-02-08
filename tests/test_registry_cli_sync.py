import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_linux_cli_matches_dask_registry():
    from FujiShaderGPU.algorithms.dask_registry import ALGORITHMS
    from FujiShaderGPU.cli.linux_cli import LinuxCLI

    cli = LinuxCLI()
    assert cli.get_supported_algorithms() == list(ALGORITHMS.keys())


def test_windows_cli_matches_tile_defaults():
    from FujiShaderGPU.cli.windows_cli import WindowsCLI
    from FujiShaderGPU.core.tile_processor import DEFAULT_ALGORITHMS

    cli = WindowsCLI()
    assert cli.get_supported_algorithms() == list(DEFAULT_ALGORITHMS.keys())
