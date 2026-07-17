"""Regression tests for P3 error-path and resource-management fixes."""
from pathlib import Path
import importlib
import sys
import types

import pytest


def _src(rel: str) -> str:
    import FujiShaderGPU

    return (Path(FujiShaderGPU.__file__).parent / rel).read_text(encoding="utf-8")


def _import_dask_processor(monkeypatch):
    """Import error-path helpers without requiring dask-cuda on Windows."""
    if "dask_cuda" not in sys.modules:
        fake_dask_cuda = types.ModuleType("dask_cuda")
        fake_dask_cuda.LocalCUDACluster = object
        monkeypatch.setitem(sys.modules, "dask_cuda", fake_dask_cuda)
    return importlib.import_module("FujiShaderGPU.core.dask_processor")


def test_overview_validation_failure_restores_original_cog(tmp_path, monkeypatch):
    dp = _import_dask_processor(monkeypatch)

    dst = tmp_path / "result.tif"
    dst.write_bytes(b"original-cog-body")

    monkeypatch.setattr(dp, "_get_overview_count", lambda _path: 0)

    def _build_repaired(_src, repaired, _options):
        repaired.write_bytes(b"rebuilt-but-invalid")

    monkeypatch.setattr(dp, "build_cog_with_overviews", _build_repaired)

    with pytest.raises(ValueError, match="has no overviews"):
        dp._ensure_cog_has_overviews(dst, {})

    assert dst.read_bytes() == b"original-cog-body"
    assert not dst.with_suffix(".no_overviews.tif").exists()
    assert not dst.with_suffix(".with_overviews.tif").exists()


def test_cog_driver_progress_bar_closes_on_translate_failure(tmp_path, monkeypatch):
    dp = _import_dask_processor(monkeypatch)

    class _Progress:
        n = 0
        closed = False

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    pbar = _Progress()
    monkeypatch.setattr(dp, "tqdm", lambda **_kwargs: pbar)
    monkeypatch.setattr(dp.gdal, "GetDriverByName", lambda _name: object())
    monkeypatch.setattr(
        dp.gdal,
        "Translate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("translate failed")),
    )

    assert not dp._build_cog_via_cog_driver(
        tmp_path / "src.tif", tmp_path / "dst.tif", {}
    )
    assert pbar.closed


def test_failed_cog_builder_removes_destination_and_sidecars(tmp_path, monkeypatch):
    from FujiShaderGPU.io import cog_builder

    dst = tmp_path / "result.tif"
    artifacts = [dst, Path(f"{dst}.tmp"), Path(f"{dst}.ovr"), Path(f"{dst}.ovr.tmp")]

    def _fail_impl(*_args, **_kwargs):
        for artifact in artifacts:
            artifact.write_bytes(b"partial")
        raise RuntimeError("simulated GDAL failure")

    monkeypatch.setattr(cog_builder, "_create_cog_ultra_fast_impl", _fail_impl)

    with pytest.raises(RuntimeError, match="simulated GDAL failure"):
        cog_builder._create_cog_ultra_fast(
            str(tmp_path / "input.vrt"), str(dst), gpu_config={}
        )

    assert all(not artifact.exists() for artifact in artifacts)


def test_streaming_translate_creates_output_parent(tmp_path, monkeypatch):
    from FujiShaderGPU.io import dem_preprocess

    dst = tmp_path / "new" / "nested" / "result.tif"
    monkeypatch.setattr(dem_preprocess.gdal, "GetDriverByName", lambda _name: object())
    monkeypatch.setattr(dem_preprocess.gdal, "Translate", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dem_preprocess, "_validate_cog_overviews", lambda _path: None)

    dem_preprocess._translate_to_cog(
        tmp_path / "source.tif",
        dst,
        block_size=512,
        overview_count=8,
        zstd_level=1,
        num_threads="1",
    )

    assert dst.parent.is_dir()


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--tile-size", "0"],
        ["--padding", "-1"],
        ["--nodata", "not-a-number"],
        ["--output-range", "10,1"],
    ],
)
def test_windows_cli_reports_invalid_values_as_parser_errors(bad_args):
    from FujiShaderGPU.cli.windows_cli import WindowsCLI

    with pytest.raises(SystemExit) as exc_info:
        WindowsCLI().parse_args(["input.tif", "output.tif", *bad_args])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tile_size": 0}, "tile_size must be a positive integer"),
        ({"tile_size": 1.5}, "tile_size must be a positive integer"),
        ({"padding": -1}, "padding must be >= 0"),
    ],
)
def test_tile_api_rejects_invalid_size_and_padding(kwargs, message):
    from FujiShaderGPU.core.tile_processor import process_dem_tiles

    with pytest.raises(ValueError, match=message):
        process_dem_tiles("missing.tif", "output.tif", **kwargs)


def test_remaining_p3_resource_contracts_are_present():
    dask_src = _src("core/dask_processor.py")
    tile_src = _src("core/tile_processor.py")
    cog_src = _src("io/cog_builder.py")
    preprocess_src = _src("io/dem_preprocess.py")
    args_src = _src("cli/args.py")

    assert "def _stop_writer()" in dask_src
    assert "client.cancel(list(fut_meta.keys()), force=True)" in dask_src
    assert "os.close(_vrt_fd)" in preprocess_src
    assert '.replace(".vrt"' not in cog_src
    assert cog_src.count(".with_name(_vrt.stem + \"_files.txt\")") == 2
    assert '(("--keep-tiles",)' in args_src
    assert "shutil.rmtree(tmp_tile_dir)" in tile_src
    assert "Removed temporary tile directory" in tile_src
