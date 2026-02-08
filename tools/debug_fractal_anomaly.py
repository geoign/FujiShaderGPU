from __future__ import annotations

import argparse
from typing import List

import cupy as cp
import numpy as np
import rasterio
from cupyx.scipy.ndimage import gaussian_filter, median_filter

from FujiShaderGPU.algorithms.dask_shared import compute_roughness_multiscale


def _parse_radii(text: str) -> List[int]:
    vals = []
    for t in text.split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(max(2, int(round(float(t)))))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug metrics for fractal_anomaly noise diagnostics")
    ap.add_argument("input_tif")
    ap.add_argument("--radii", default="4,8,16,32,64")
    ap.add_argument("--smoothing-sigma", type=float, default=1.2)
    ap.add_argument("--despeckle-threshold", type=float, default=0.35)
    ap.add_argument("--despeckle-alpha-max", type=float, default=0.30)
    ap.add_argument("--detail-boost", type=float, default=0.35)
    ap.add_argument("--max-size", type=int, default=2048, help="downsample long edge for debug")
    args = ap.parse_args()

    radii = _parse_radii(args.radii)
    if len(radii) < 3:
        raise SystemExit("Need at least 3 radii")

    with rasterio.open(args.input_tif) as src:
        h, w = src.height, src.width
        scale = max(h / args.max_size, w / args.max_size, 1.0)
        out_h = max(128, int(h / scale))
        out_w = max(128, int(w / scale))
        arr_ma = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
            out_dtype=np.float32,
            masked=True,
        )
        arr = arr_ma.filled(np.nan).astype(np.float32, copy=False)
        nodata = src.nodata
        if nodata is None:
            # Align with tile pipeline behavior: infer nodata=0 from border when metadata is absent.
            bw = int(min(64, max(1, out_h // 20), max(1, out_w // 20)))
            border = np.concatenate(
                [
                    arr[:bw, :].ravel(),
                    arr[-bw:, :].ravel(),
                    arr[:, :bw].ravel(),
                    arr[:, -bw:].ravel(),
                ]
            )
            if border.size > 0:
                zero_ratio = float(np.count_nonzero(border == 0.0) / border.size)
                nonzero_present = np.any(border != 0.0)
                if zero_ratio >= 0.6 or (zero_ratio >= 0.3 and nonzero_present):
                    nodata = 0.0
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan

    dem = cp.asarray(arr, dtype=cp.float32)
    nan_mask = cp.isnan(dem)
    valid_dem = dem[~nan_mask]

    sigmas = compute_roughness_multiscale(dem, radii, window_mult=3, detrend=True)
    all_scales = cp.asarray(radii, dtype=cp.float32)
    fit_scales = all_scales
    fit_sigmas = sigmas
    log_scales = cp.log(fit_scales)
    log_sigmas = cp.log(cp.maximum(fit_sigmas, 1e-4))

    scale_w = cp.sqrt(fit_scales)
    scale_w = scale_w / cp.sum(scale_w)
    w3 = scale_w.reshape(1, 1, -1)
    mx = cp.sum(log_scales * scale_w)
    my = cp.sum(log_sigmas * w3, axis=2)
    xb = log_scales.reshape(1, 1, -1)
    cov = cp.sum((xb - mx) * (log_sigmas - my[:, :, None]) * w3, axis=2)
    varx = cp.sum(((log_scales - mx) ** 2) * scale_w)
    beta = cov / (varx + 1e-10)

    y_fit = my[:, :, None] + beta[:, :, None] * (xb - mx)
    ss_res = cp.sum(((log_sigmas - y_fit) ** 2) * w3, axis=2)
    ss_tot = cp.sum(((log_sigmas - my[:, :, None]) ** 2) * w3, axis=2)
    r2 = cp.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0)
    r2 = cp.where(nan_mask, cp.nan, r2)

    rmse = cp.sqrt(cp.maximum(ss_res, 0.0))
    beta_dev = cp.clip(beta - 1.2, -4.0, 4.0)
    rmse_comp = cp.log1p(cp.maximum(rmse, 0.0))

    roughness = cp.mean(sigmas, axis=2)
    rv = roughness[~nan_mask]
    if rv.size > 0:
        r_p10 = float(cp.percentile(rv, 10))
        r_p75 = float(cp.percentile(rv, 75))
    else:
        r_p10, r_p75 = 0.0, 1.0
    relief_conf = cp.clip((roughness - r_p10) / max(r_p75 - r_p10, 1e-6), 0.0, 1.0)

    fine_i = 0
    coarse_i = min(2, log_sigmas.shape[2] - 1)
    fine_ratio = log_sigmas[:, :, fine_i] - log_sigmas[:, :, coarse_i]
    max_i = log_sigmas.shape[2] - 1
    macro_i = max(max_i - 2, 0)
    macro_ratio = log_sigmas[:, :, max_i] - log_sigmas[:, :, macro_i]
    feature_raw = (
        0.75 * beta_dev
        + 0.45 * rmse_comp
        + 0.35 * macro_ratio * relief_conf
        + float(args.detail_boost) * 0.18 * fine_ratio * relief_conf
    )

    if args.smoothing_sigma > 0:
        d_sm = gaussian_filter(cp.where(nan_mask, 0, feature_raw), sigma=args.smoothing_sigma, mode="nearest")
        w_sm = gaussian_filter((~nan_mask).astype(cp.float32), sigma=args.smoothing_sigma, mode="nearest")
        d_smooth = cp.where(w_sm > 0, d_sm / cp.maximum(w_sm, 1e-6), cp.nan)
    else:
        d_smooth = feature_raw

    alpha_core = cp.clip((r2 - 0.35) / 0.6, 0.0, 1.0) * relief_conf
    alpha = 0.50 + 0.50 * alpha_core
    feat_smooth = d_smooth if args.smoothing_sigma > 0 else feature_raw
    d_map = alpha * feature_raw + (1.0 - alpha) * feat_smooth
    d_med = median_filter(d_map, size=3, mode="nearest")
    thr = max(0.05, float(args.despeckle_threshold))
    thr_map = thr * (0.7 + 1.1 * alpha)
    d_map = cp.where(
        (cp.abs(d_map - d_med) > thr_map)
        & (alpha < float(args.despeckle_alpha_max))
        & (relief_conf < 0.45),
        d_med,
        d_map,
    )
    hp = d_map - gaussian_filter(cp.where(cp.isnan(d_map), 0, d_map), sigma=2.0, mode="nearest")
    hp = cp.where(cp.isnan(d_map), cp.nan, hp)

    def p(arr_cp, qs):
        v = arr_cp[~cp.isnan(arr_cp)]
        if v.size == 0:
            return [float("nan")] * len(qs)
        return [float(x) for x in cp.percentile(v, cp.asarray(qs, dtype=cp.float32))]

    print("=== Fractal Debug ===")
    print(f"shape={arr.shape}, radii={radii}, fit_scales={cp.asnumpy(fit_scales).tolist()}, smoothing_sigma={args.smoothing_sigma}, despeckle_threshold={args.despeckle_threshold}, despeckle_alpha_max={args.despeckle_alpha_max}, detail_boost={args.detail_boost}, nodata={nodata}")
    print(f"DEM p05/p50/p95: {p(valid_dem, [5,50,95])}")
    for i, r in enumerate(radii):
        s = sigmas[:, :, i]
        print(f"sigma(r={r}) p50/p95: {p(s, [50,95])}")
    print(f"R2 p05/p50/p95: {p(r2, [5,50,95])}")
    print(f"Relief p05/p50/p95: {p(roughness, [5,50,95])}")
    print(f"ReliefConf p05/p50/p95: {p(relief_conf, [5,50,95])}")
    print(f"Alpha p05/p50/p95: {p(alpha, [5,50,95])}")
    print(f"Beta p05/p50/p95: {p(beta, [5,50,95])}")
    print(f"RMSE p05/p50/p95: {p(rmse, [5,50,95])}")
    print(f"MacroRatio p05/p50/p95: {p(macro_ratio, [5,50,95])}")
    print(f"FeatureRaw p05/p50/p95: {p(feature_raw, [5,50,95])}")
    print(f"D_out p05/p50/p95: {p(d_map, [5,50,95])}")
    d_valid = d_map[~cp.isnan(d_map)]
    hp_valid = hp[~cp.isnan(hp)]
    if d_valid.size > 0 and hp_valid.size > 0:
        noise_ratio = float(cp.std(hp_valid) / cp.maximum(cp.std(d_valid), 1e-8))
    else:
        noise_ratio = float("nan")
    print(f"high_freq_noise_ratio={noise_ratio:.6f}")


if __name__ == "__main__":
    main()
