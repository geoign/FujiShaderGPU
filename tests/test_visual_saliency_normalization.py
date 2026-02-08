import cupy as cp


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
    norm_min, norm_max = visual_saliency_stat_func(raw)
    out = compute_visual_saliency_block(
        terrain, normalize=True, norm_min=norm_min, norm_max=norm_max
    )

    p05, p95 = [float(v) for v in cp.percentile(out, cp.asarray([5, 95], dtype=cp.float32))]
    assert 0.0 <= float(cp.nanmin(out)) <= 1.0
    assert 0.0 <= float(cp.nanmax(out)) <= 1.0
    assert (p95 - p05) > 0.2
