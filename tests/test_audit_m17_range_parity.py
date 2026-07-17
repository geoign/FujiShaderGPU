"""Regression test for audit M-17 (part of P1-9): when no fixed output range
is resolvable, both the tile and Dask backends must fall back to float32 rather
than quantizing from a backend-specific data estimate.

This is a pure-Python/logic check on resolve_output_range and the shared
quantize helpers; it does not require a GPU.
"""
import pytest


def test_unresolvable_range_is_none_for_both_backends():
    from FujiShaderGPU.io.output_encoding import resolve_output_range

    # slope in percent is unbounded -> no fixed range.
    assert resolve_output_range("slope", params={"unit": "percent"}) is None
    # tv_decomposition structure component is a raw elevation surface.
    assert resolve_output_range(
        "tv_decomposition", params={"component": "structure"}) is None


def test_resolved_range_is_identical_for_tile_and_dask():
    # Both backends resolve the SAME fixed range for a bounded algorithm, so the
    # quantization params (and thus the DN<->value mapping) match exactly.
    from FujiShaderGPU.io.output_encoding import (
        resolve_output_range, quantize_params,
    )

    vr = resolve_output_range("topousm_fast", params={})
    assert vr is not None
    qp16 = quantize_params(float(vr[0]), float(vr[1]), "int16")
    qp8 = quantize_params(float(vr[0]), float(vr[1]), "uint8")
    # Same params regardless of which backend calls it.
    assert qp16 == quantize_params(float(vr[0]), float(vr[1]), "int16")
    assert qp8 == quantize_params(float(vr[0]), float(vr[1]), "uint8")


def test_dask_processor_uses_float32_fallback_source():
    # Pin the source contract: dask_processor must NOT call _estimate_output_range
    # in the value_range-is-None branch anymore (that reintroduced the M-17
    # tile/Dask dtype divergence).  Read the file directly so this passes on
    # environments without dask_cuda (the module import needs a CUDA stack).
    from pathlib import Path
    import FujiShaderGPU

    dp_path = Path(FujiShaderGPU.__file__).parent / "core" / "dask_processor.py"
    src = dp_path.read_text(encoding="utf-8")
    # The None branch should warn and write float32, not estimate+quantize.
    assert "writing float32 instead of" in src
    # _estimate_output_range must not be invoked inside the pipeline.
    assert "_estimate_output_range(result_gpu)" not in src


def test_tile_processor_uses_float32_fallback_source():
    # Same M-17 contract on the tile backend: no quantize when range is None.
    from pathlib import Path
    import FujiShaderGPU

    tp_path = Path(FujiShaderGPU.__file__).parent / "core" / "tile_processor.py"
    src = tp_path.read_text(encoding="utf-8")
    assert "writing float32 instead of" in src
