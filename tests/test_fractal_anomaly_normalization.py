import cupy as cp


def test_fractal_anomaly_normalization_is_signed_and_not_collapsed():
    from FujiShaderGPU.algorithms.dask_shared import (
        compute_fractal_dimension_block,
        fractal_stat_func,
    )

    x = cp.linspace(-4, 4, 512, dtype=cp.float32)
    y = cp.linspace(-4, 4, 512, dtype=cp.float32)
    xx, yy = cp.meshgrid(x, y)

    terrain = (
        cp.sin(xx * 1.7) * 120.0
        + cp.cos(yy * 2.1) * 80.0
        + cp.exp(-((xx + 1.2) ** 2 + (yy - 0.8) ** 2) * 1.6) * 260.0
        - cp.exp(-((xx - 1.8) ** 2 + (yy + 1.5) ** 2) * 2.5) * 180.0
    ).astype(cp.float32)

    raw = compute_fractal_dimension_block(terrain, normalize=False)
    center, scale = fractal_stat_func(raw)
    out = compute_fractal_dimension_block(
        terrain, normalize=True, mean_global=center, std_global=scale
    )

    p05, p95 = [float(v) for v in cp.percentile(out, cp.asarray([5, 95], dtype=cp.float32))]
    assert -1.0 <= float(cp.nanmin(out)) <= 1.0
    assert -1.0 <= float(cp.nanmax(out)) <= 1.0
    assert p05 < 0.0 < p95
    assert (p95 - p05) > 0.2


def test_fractal_anomaly_smoothing_reduces_noise():
    from FujiShaderGPU.algorithms.dask_shared import compute_fractal_dimension_block

    x = cp.linspace(-5, 5, 256, dtype=cp.float32)
    y = cp.linspace(-5, 5, 256, dtype=cp.float32)
    xx, yy = cp.meshgrid(x, y)
    terrain = (
        cp.sin(xx * 7.0) * 8.0
        + cp.cos(yy * 6.5) * 8.0
        + cp.random.normal(0.0, 4.0, size=xx.shape, dtype=cp.float32)
    ).astype(cp.float32)

    raw_low = compute_fractal_dimension_block(
        terrain, normalize=False, smoothing_sigma=0.0
    )
    raw_high = compute_fractal_dimension_block(
        terrain, normalize=False, smoothing_sigma=1.8
    )

    # stronger smoothing should suppress high-frequency variance
    assert float(cp.nanstd(raw_high)) < float(cp.nanstd(raw_low))
