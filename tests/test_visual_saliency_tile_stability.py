import cupy as cp


def test_visual_saliency_tile_split_stability():
    from FujiShaderGPU.algorithms.dask_shared import compute_visual_saliency_block

    x = cp.linspace(-4, 4, 512, dtype=cp.float32)
    y = cp.linspace(-3, 3, 384, dtype=cp.float32)
    xx, yy = cp.meshgrid(x, y)
    dem = (
        cp.sin(xx * 2.2) * 120
        + cp.cos(yy * 1.3) * 90
        + cp.exp(-((xx - 0.7) ** 2 + (yy + 0.5) ** 2) * 3.0) * 260
    ).astype(cp.float32)

    full = compute_visual_saliency_block(dem, normalize=False)

    seam_x = 256
    pad = 128
    left = dem[:, : seam_x + pad]
    right = dem[:, seam_x - pad :]
    left_res = compute_visual_saliency_block(left, normalize=False)
    right_res = compute_visual_saliency_block(right, normalize=False)

    stitched = cp.concatenate([left_res[:, :seam_x], right_res[:, pad:]], axis=1)
    mae = float(cp.nanmean(cp.abs(stitched - full)))
    assert mae < 0.02
