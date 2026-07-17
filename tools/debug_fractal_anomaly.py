from __future__ import annotations

import argparse
from typing import Iterable

import cupy as cp
import numpy as np
import rasterio
from cupyx.scipy.ndimage import gaussian_filter

from FujiShaderGPU.algorithms._impl_fractal_anomaly import (
    compute_fractal_dimension_block,
    compute_roughness_multiscale,
    fractal_stat_func,
)


def _parse_radii(text: str) -> list[int]:
    radii = [max(2, int(round(float(token)))) for token in text.split(",") if token.strip()]
    if len(radii) < 3:
        raise argparse.ArgumentTypeError("at least three comma-separated radii are required")
    return radii


def _percentiles(values: cp.ndarray, quantiles: Iterable[float]) -> list[float]:
    quantiles = list(quantiles)
    valid = values[~cp.isnan(values)]
    if valid.size == 0:
        return [float("nan")] * len(quantiles)
    result = cp.percentile(valid, cp.asarray(quantiles, dtype=cp.float32))
    return [float(value) for value in result]


def _read_dem(path: str, max_size: int) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        scale = max(height / max_size, width / max_size, 1.0)
        out_height = max(1, int(height / scale))
        out_width = max(1, int(width / scale))
        masked = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.nearest,
            out_dtype=np.float32,
            masked=True,
        )
        array = masked.filled(np.nan).astype(np.float32, copy=False)
        nodata = src.nodata
        if nodata is None:
            # Match the tile pipeline's border-based nodata=0 inference.
            border_width = int(min(64, max(1, out_height // 20), max(1, out_width // 20)))
            border = np.concatenate(
                (
                    array[:border_width, :].ravel(),
                    array[-border_width:, :].ravel(),
                    array[:, :border_width].ravel(),
                    array[:, -border_width:].ravel(),
                )
            )
            if border.size:
                zero_ratio = float(np.count_nonzero(border == 0.0) / border.size)
                nonzero_present = bool(np.any(border != 0.0))
                if zero_ratio >= 0.6 or (zero_ratio >= 0.3 and nonzero_present):
                    nodata = 0.0
        if nodata is not None:
            array[np.isclose(array, nodata, rtol=0.0, atol=1e-6)] = np.nan
    return array, nodata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run production fractal_anomaly code and print noise diagnostics"
    )
    parser.add_argument("input_tif")
    parser.add_argument("--radii", type=_parse_radii, default=_parse_radii("4,8,16,32,64"))
    parser.add_argument("--smoothing-sigma", type=float, default=1.2)
    parser.add_argument("--despeckle-threshold", type=float, default=0.35)
    parser.add_argument("--despeckle-alpha-max", type=float, default=0.30)
    parser.add_argument("--detail-boost", type=float, default=0.35)
    parser.add_argument("--max-size", type=int, default=2048, help="downsample long edge for debug")
    args = parser.parse_args()
    if args.max_size < 1:
        parser.error("--max-size must be at least 1")

    array, nodata = _read_dem(args.input_tif, args.max_size)
    dem = cp.asarray(array, dtype=cp.float32)
    kwargs = {
        "radii": args.radii,
        "smoothing_sigma": args.smoothing_sigma,
        "despeckle_threshold": args.despeckle_threshold,
        "despeckle_alpha_max": args.despeckle_alpha_max,
        "detail_boost": args.detail_boost,
    }

    # Use the same production entry points as the Dask pipeline so this tool
    # cannot drift from the fractal regression and despeckling implementation.
    raw = compute_fractal_dimension_block(dem, normalize=False, **kwargs)
    center, scale = fractal_stat_func(raw)
    output = compute_fractal_dimension_block(
        dem,
        normalize=True,
        mean_global=center,
        std_global=scale,
        **kwargs,
    )
    roughness = compute_roughness_multiscale(dem, args.radii, window_mult=3, detrend=True)

    filled = cp.where(cp.isnan(output), cp.float32(0), output)
    high_frequency = output - gaussian_filter(filled, sigma=2.0, mode="nearest")
    high_frequency = cp.where(cp.isnan(output), cp.nan, high_frequency)

    print("=== Fractal Debug (production implementation) ===")
    print(
        f"shape={array.shape}, radii={args.radii}, smoothing_sigma={args.smoothing_sigma}, "
        f"despeckle_threshold={args.despeckle_threshold}, "
        f"despeckle_alpha_max={args.despeckle_alpha_max}, "
        f"detail_boost={args.detail_boost}, nodata={nodata}"
    )
    print(f"DEM p05/p50/p95: {_percentiles(dem, [5, 50, 95])}")
    for index, radius in enumerate(args.radii):
        print(f"sigma(r={radius}) p50/p95: {_percentiles(roughness[:, :, index], [50, 95])}")
    print(f"RawFeature p05/p50/p95: {_percentiles(raw, [5, 50, 95])}")
    print(f"normalization center={center:.6g}, scale={scale:.6g}")
    print(f"Output p05/p50/p95: {_percentiles(output, [5, 50, 95])}")

    output_valid = output[~cp.isnan(output)]
    high_frequency_valid = high_frequency[~cp.isnan(high_frequency)]
    if output_valid.size and high_frequency_valid.size:
        noise_ratio = float(
            cp.std(high_frequency_valid) / cp.maximum(cp.std(output_valid), cp.float32(1e-8))
        )
    else:
        noise_ratio = float("nan")
    print(f"high_freq_noise_ratio={noise_ratio:.6f}")


if __name__ == "__main__":
    main()
