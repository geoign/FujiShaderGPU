import cupy as cp
from FujiShaderGPU.algorithms._normalization import NORMAL_PERCENTILE


def test_visual_saliency_not_collapsed_to_narrow_band():
    from FujiShaderGPU.algorithms.dask_shared import (
        compute_visual_saliency_block,
        visual_saliency_stat_func,
    )

    x = cp.linspace(-3, 3, 512, dtype=cp.float32)
    y = cp.linspace(-3, 3, 512, dtype=cp.float32)
    xx, yy = cp.meshgrid(x, y)
    terrain = (
        (cp.sin(xx * 2.1) + cp.cos(yy * 1.6)) * 100
        + cp.exp(-((xx - 0.8) ** 2 + (yy + 0.3) ** 2) * 2.0) * 250
    ).astype(cp.float32)

    raw = compute_visual_saliency_block(terrain, normalize=False)
    norm_min, norm_scale = visual_saliency_stat_func(raw)
    out = compute_visual_saliency_block(
        terrain, normalize=True, norm_min=norm_min, norm_scale=norm_scale
    )

    p05, p95 = [float(v) for v in cp.percentile(out, cp.asarray([5, 95], dtype=cp.float32))]
    # Unsigned, 0-anchored, normalized so the robust p99 maps to ~1.0 (unclipped).
    valid = out[~cp.isnan(out)]
    p99 = float(cp.percentile(valid, NORMAL_PERCENTILE))
    assert float(cp.nanmin(out)) >= 0.0
    assert 0.7 <= p99 <= 1.3
    assert (p95 - p05) > 0.2
