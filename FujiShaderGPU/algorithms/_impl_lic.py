"""
FujiShaderGPU/algorithms/_impl_lic.py

Line Integral Convolution (LIC) flow texture for terrain.

Cabral & Leedom 1993 (SIGGRAPH): convolve a noise image along the streamlines
of a vector field derived from the DEM -- ``flow`` (down the gradient; drainage
fabric) or ``contour`` (along contours / strike).  The result is a brushed,
direction-correlated texture; multiplied with a standard hillshade (default) it
reads as a classic relief with the drainage/strike fabric etched into it.

Determinism without global coordinates: the noise is a hash of the elevation
value itself (position-independent), so tiles, chunk layouts, and backends all
see the identical noise field -- no seams by construction.  Perfectly flat
areas hash to a constant (no texture), which is fine: they have no flow
direction either.

Single map_overlap pass; halo = integration length + smoothing support, capped
well inside Constants.MAX_DEPTH.
"""
from __future__ import annotations
import logging
import math

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

from ._base import Constants, DaskAlgorithm
from ._nan_utils import restore_nan
from ._impl_structure_tensor import nan_filled

logger = logging.getLogger(__name__)

# Integration length hard cap: keeps L + smoothing support < MAX_DEPTH.
LIC_MAX_LENGTH = 120


def _elevation_noise(filled: cp.ndarray) -> cp.ndarray:
    """Position-independent white-ish noise from the elevation value.

    float64 keeps the hash chaotic for metre-scale elevations with millimetre
    differences; identical input values give identical noise on every backend.
    """
    z = filled.astype(cp.float64)
    n = cp.sin(z * 127.1 + 311.7) * 43758.5453123
    return (n - cp.floor(n)).astype(cp.float32)


def compute_lic_block(block, *, length=20, lic_field='flow', composite='hillshade',
                      flow_sigma=1.5, azimuth=None, altitude=None,
                      pixel_size=1.0, pixel_scale_x=None, pixel_scale_y=None,
                      normalize=True, **_ignored):
    """LIC texture on one CuPy block (NaN-aware)."""
    filled, nan_mask = nan_filled(block)
    h, w = int(block.shape[0]), int(block.shape[1])
    if h < 4 or w < 4:
        return restore_nan(cp.full(block.shape, 0.5, dtype=cp.float32), nan_mask)

    L = int(max(1, min(int(length), LIC_MAX_LENGTH)))
    max_flow_sigma = max(0.5, (Constants.MAX_DEPTH - L - 4) / 4.0)
    flow_sigma = min(max(0.5, float(flow_sigma)), max_flow_sigma)
    smooth = gaussian_filter(filled, sigma=flow_sigma,
                             mode='nearest')
    gy, gx = cp.gradient(smooth)
    if str(lic_field or 'flow').lower() == 'contour':
        vx, vy = -gy, gx  # rotate 90 deg: along-contour (strike) direction
    else:
        vx, vy = gx, gy   # flow: along the gradient (sign irrelevant to LIC)
    mag = cp.sqrt(vx * vx + vy * vy)
    eps = cp.float32(1e-12)
    inv = cp.where(mag > eps, 1.0 / (mag + eps), cp.float32(0.0))
    vx = (vx * inv).astype(cp.float32)
    vy = (vy * inv).astype(cp.float32)

    noise = _elevation_noise(filled)
    yy, xx = cp.meshgrid(cp.arange(h, dtype=cp.float32),
                         cp.arange(w, dtype=cp.float32), indexing='ij')
    acc = noise.copy()
    total = cp.ones(block.shape, dtype=cp.float32)
    px = cp.empty_like(xx)
    py = cp.empty_like(yy)
    coords = cp.empty((2, h, w), dtype=cp.float32)
    for direction in (1.0, -1.0):
        px[...] = xx
        py[...] = yy
        dvx = direction * vx
        dvy = direction * vy
        for _ in range(L):
            coords[0] = py
            coords[1] = px
            sx = map_coordinates(dvx, coords, order=1, mode='nearest')
            sy = map_coordinates(dvy, coords, order=1, mode='nearest')
            cp.add(px, sx, out=px)
            cp.add(py, sy, out=py)
            coords[0] = py
            coords[1] = px
            acc += map_coordinates(noise, coords, order=1, mode='nearest')
            total += 1.0
    lic = acc / total

    # Streamline averaging shrinks the noise variance ~1/(2L+1); re-expand the
    # contrast deterministically (no data-dependent stats -> seam-free).
    gain = cp.float32(math.sqrt(2.0 * L + 1.0))
    lic = cp.clip(0.5 + (lic - 0.5) * gain, 0.0, 1.0)

    if str(composite or 'hillshade').lower() == 'hillshade':
        az = math.radians(float(azimuth if azimuth is not None
                                else Constants.DEFAULT_AZIMUTH))
        alt = math.radians(float(altitude if altitude is not None
                                 else Constants.DEFAULT_ALTITUDE))
        step_y = abs(float(pixel_scale_y if pixel_scale_y is not None else pixel_size)) or 1.0
        step_x = abs(float(pixel_scale_x if pixel_scale_x is not None else pixel_size)) or 1.0
        dy, dx = cp.gradient(filled, step_y, step_x)
        denom = cp.sqrt(dx * dx + dy * dy + 1.0)
        nxl = -dx / denom
        nyl = -dy / denom
        nzl = 1.0 / denom
        lx = math.sin(az) * math.cos(alt)
        ly = math.cos(az) * math.cos(alt)
        lz = math.sin(alt)
        hs = cp.maximum(0.0, cp.float32(lx) * nxl + cp.float32(ly) * nyl
                        + cp.float32(lz) * nzl)
        lic = lic * hs

    return restore_nan(lic.astype(cp.float32), nan_mask)


class LICAlgorithm(DaskAlgorithm):
    """LIC flow texture (drainage / strike fabric rendering)."""

    def process(self, gpu_arr, **params):
        L = int(max(1, min(int(params.get('length', 20)), LIC_MAX_LENGTH)))
        if int(params.get('length', 20)) > LIC_MAX_LENGTH:
            logger.info("lic: length clamped to %d px (halo budget).",
                        LIC_MAX_LENGTH)
        requested_flow_sigma = float(params.get('flow_sigma', 1.5))
        max_flow_sigma = max(0.5, (Constants.MAX_DEPTH - L - 4) / 4.0)
        flow_sigma = min(max(0.5, requested_flow_sigma), max_flow_sigma)
        if flow_sigma != requested_flow_sigma:
            logger.warning(
                "lic: flow_sigma %.3g clamped to %.3g to fit the halo budget.",
                requested_flow_sigma, flow_sigma,
            )
        depth = min(int(L + 4 * flow_sigma + 4), Constants.MAX_DEPTH)
        try:
            min_chunk = min(min(ax) for ax in gpu_arr.chunks)
            depth = max(1, min(depth, int(min_chunk) - 1))
        except Exception:
            pass
        return gpu_arr.map_overlap(
            compute_lic_block, depth=depth, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            length=L, lic_field=params.get('lic_field', 'flow'),
            composite=params.get('composite', 'hillshade'),
            flow_sigma=flow_sigma,
            azimuth=params.get('azimuth', None),
            altitude=params.get('altitude', None),
            pixel_size=float(params.get('pixel_size', 1.0)),
            pixel_scale_x=params.get('pixel_scale_x', None),
            pixel_scale_y=params.get('pixel_scale_y', None))

    def get_default_params(self):
        return {
            'length': 20, 'lic_field': 'flow', 'composite': 'hillshade',
            'flow_sigma': 1.5,
        }


__all__ = ["compute_lic_block", "LICAlgorithm", "LIC_MAX_LENGTH"]
