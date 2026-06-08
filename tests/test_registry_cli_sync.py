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


def test_both_clis_expose_same_algorithms():
    from FujiShaderGPU.cli.linux_cli import LinuxCLI
    from FujiShaderGPU.cli.windows_cli import WindowsCLI

    assert LinuxCLI().get_supported_algorithms() == WindowsCLI().get_supported_algorithms()


def test_shared_arguments_present_on_both_clis():
    # Every shared algorithm/output/spatial argument must exist identically on
    # both platform parsers (single source of truth in cli/args.py).
    from FujiShaderGPU.cli.linux_cli import LinuxCLI
    from FujiShaderGPU.cli.windows_cli import WindowsCLI
    from FujiShaderGPU.cli.args import SHARED_ARGS

    def flags(cli):
        return {s for a in cli.parser._actions for s in a.option_strings}

    lin, win = flags(LinuxCLI()), flags(WindowsCLI())
    for spec_flags, _ in SHARED_ARGS:
        for f in spec_flags:
            assert f in lin, f"{f} missing from Linux CLI"
            assert f in win, f"{f} missing from Windows CLI"


def test_build_algo_params_parity_across_platforms():
    # The same argv must yield the same algorithm params on both backends, apart
    # from intentional platform-only extras (color_mode on tile, verbose on dask).
    from FujiShaderGPU.cli.linux_cli import LinuxCLI
    from FujiShaderGPU.cli.windows_cli import WindowsCLI
    from FujiShaderGPU.cli.args import build_algo_params, parse_list_fields

    cases = [
        ["i", "o", "--algorithm", "rvi", "--radii", "4,16,64", "--weights", "0.5,0.3,0.2"],
        ["i", "o", "--algorithm", "hillshade", "--azimuth", "300", "--multiscale"],
        ["i", "o", "--algorithm", "blur", "--blur-radius", "9"],
        ["i", "o", "--algorithm", "blur", "--radii", "5,20"],
        ["i", "o", "--algorithm", "multiscale_terrain", "--scales", "1,10,50"],
        ["i", "o", "--algorithm", "openness", "--radius", "12", "--num-directions", "8"],
        ["i", "o", "--algorithm", "fractal_anomaly", "--fractal-radii", "2,4,8,16,32"],
        ["i", "o", "--algorithm", "multi_light_uncertainty", "--ml-azimuths", "315,45"],
    ]
    lin, win = LinuxCLI(), WindowsCLI()
    for argv in cases:
        la = lin.parser.parse_args(argv); parse_list_fields(la, lin.parser)
        wa = win.parser.parse_args(argv); parse_list_fields(wa, win.parser)
        pl = {k: v for k, v in build_algo_params(la).items() if k != "verbose"}
        pw = {k: v for k, v in build_algo_params(wa).items() if k != "color_mode"}
        assert pl == pw, f"param mismatch for {argv}: {pl} != {pw}"
