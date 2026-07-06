"""
FujiShaderGPU/algorithms/_impl_phase_congruency.py

Phase Congruency relief via the monogenic signal.

Features are detected where the Fourier components agree in phase (Morrone &
Owens 1987; Kovesi 1999), which makes the response AMPLITUDE-INVARIANT: a
decimetre-scale terrace riser scores like a mountain front.  The 2D odd
(quadrature) part comes from the Riesz transform of log-Gabor bandpasses --
the monogenic signal (Felsberg & Sommer 2001).

Per wavelength lambda_i (pixels):

    even_i        = IFFT( F * G_i )                 (log-Gabor bandpass)
    (odd1, odd2)_i = IFFT( F * G_i * (i u/|u|, i v/|u|) )   (Riesz)
    A_i = sqrt(even^2 + odd1^2 + odd2^2)            (local amplitude)

    E  = |sum_i (even, odd1, odd2)_i|               (local energy)
    PC = W * max(E - T, 0) / (sum_i A_i + eps)

``T`` is the noise-energy threshold derived from the GLOBAL median of the
smallest-scale amplitude (Rayleigh model, simplified from Kovesi's estimator);
``W`` is the frequency-spread weighting.  Wavelengths are clamped to
``PC_MAX_WAVELENGTH`` so the FFT halo (2 * lambda_max) stays inside
Constants.MAX_DEPTH -- larger scales are a documented future extension via the
overview path.

--feature-type: ``both`` (default; polarity-signed, 0.5-centred),
``edge`` (unsigned PC), ``ridge`` / ``valley`` (polarity-selected).
"""
from __future__ import annotations
import logging
import math

import cupy as cp

from ._base import Constants, DaskAlgorithm
from ._nan_utils import restore_nan
from ._impl_structure_tensor import nan_filled

logger = logging.getLogger(__name__)

PC_MIN_WAVELENGTH = 3.0
PC_MAX_WAVELENGTH = 64.0
_DEFAULT_WAVELENGTHS = (4.0, 8.0, 16.0, 32.0, 64.0)


def resolve_pc_wavelengths(radii) -> list:
    """Clamp/dedupe --radii into usable wavelengths (fallback: default bank)."""
    out = []
    for r in (radii or []):
        try:
            v = float(r)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append(min(max(v, PC_MIN_WAVELENGTH), PC_MAX_WAVELENGTH))
    out = sorted(set(out))
    if len(out) < 2:
        return list(_DEFAULT_WAVELENGTHS)
    return out


def compute_phase_congruency_block(block, *, wavelengths=None, radii=None,
                                   sigma_onf=0.55, noise_k=2.0,
                                   feature_type='both', normalize=True,
                                   global_stats=None, pixel_size=1.0,
                                   pixel_scale_x=None, pixel_scale_y=None,
                                   **_ignored):
    """Monogenic phase congruency on one CuPy block.

    ``normalize=False`` (stats prepass) returns the smallest-scale local
    amplitude ``A_min`` -- the input for the global noise median.  The main
    pass reads that median from ``global_stats[1]`` (fallback: block median).
    """
    filled, nan_mask = nan_filled(block)
    h, w = int(block.shape[0]), int(block.shape[1])
    if h < 8 or w < 8:
        return restore_nan(cp.zeros(block.shape, dtype=cp.float32), nan_mask)

    scales = resolve_pc_wavelengths(wavelengths if wavelengths else radii)

    work = filled - cp.mean(filled)
    F = cp.fft.fft2(work)
    fy = cp.fft.fftfreq(h).astype(cp.float32)
    fx = cp.fft.fftfreq(w).astype(cp.float32)
    u = fx[None, :]
    v = fy[:, None]
    rho = cp.sqrt(u * u + v * v)
    rho_safe = rho.copy()
    rho_safe[0, 0] = 1.0
    # Riesz transfer functions (i u/|u|, i v/|u|).
    r1 = 1j * (u / rho_safe)
    r2 = 1j * (v / rho_safe)
    # Gentle lowpass keeps the highest-frequency log-Gabor from ringing at Nyquist.
    lowpass = 1.0 / (1.0 + (rho / 0.45) ** 20)
    log_sigma = math.log(max(0.1, min(float(sigma_onf), 0.999)))
    denom_lg = cp.float32(2.0 * log_sigma * log_sigma)

    sum_e = cp.zeros(block.shape, dtype=cp.float32)
    sum_o1 = cp.zeros(block.shape, dtype=cp.float32)
    sum_o2 = cp.zeros(block.shape, dtype=cp.float32)
    sum_a = cp.zeros(block.shape, dtype=cp.float32)
    a_max = cp.zeros(block.shape, dtype=cp.float32)
    a_min_scale = None
    for lam in scales:
        f0 = 1.0 / float(lam)
        g = cp.exp((cp.log(rho_safe / cp.float32(f0)) ** 2) / denom_lg * -1.0)
        g = (g * lowpass).astype(cp.float32)
        g[0, 0] = 0.0
        fg = F * g
        even = cp.real(cp.fft.ifft2(fg)).astype(cp.float32)
        odd1 = cp.real(cp.fft.ifft2(fg * r1)).astype(cp.float32)
        odd2 = cp.real(cp.fft.ifft2(fg * r2)).astype(cp.float32)
        a_i = cp.sqrt(even * even + odd1 * odd1 + odd2 * odd2)
        if a_min_scale is None:  # scales are sorted ascending
            a_min_scale = a_i
        sum_e += even
        sum_o1 += odd1
        sum_o2 += odd2
        sum_a += a_i
        a_max = cp.maximum(a_max, a_i)

    if not normalize:
        return restore_nan(a_min_scale, nan_mask)

    med = None
    if isinstance(global_stats, (tuple, list)) and len(global_stats) >= 2 \
            and float(global_stats[1]) > 1e-12:
        med = float(global_stats[1])
    if med is None:
        vals = a_min_scale[~nan_mask]
        med = float(cp.median(vals)) if vals.size else 0.0

    # Rayleigh noise model on the smallest scale, propagated to the total
    # energy with the white-noise amplitude ratio sqrt(lam_0 / lam_i) per scale
    # (simplified from Kovesi 1999).
    tau = med / math.sqrt(math.log(4.0)) if med > 0 else 0.0
    amp_factor = sum(math.sqrt(scales[0] / s) for s in scales)
    noise_mean = tau * amp_factor * math.sqrt(math.pi / 2.0)
    noise_sd = tau * amp_factor * math.sqrt((4.0 - math.pi) / 2.0)
    T = cp.float32(noise_mean + float(noise_k) * noise_sd)

    energy = cp.sqrt(sum_e * sum_e + sum_o1 * sum_o1 + sum_o2 * sum_o2)
    eps = cp.float32(1e-6)
    # Frequency-spread weighting (Kovesi): penalize single-scale responses.
    n_s = float(len(scales))
    width = (sum_a / (a_max + eps)) / n_s
    weight = 1.0 / (1.0 + cp.exp(cp.float32(10.0) * (cp.float32(0.5) - width)))
    pc = weight * cp.maximum(energy - T, 0.0) / (sum_a + eps)

    ft = str(feature_type or 'both').lower()
    if ft == 'edge':
        out = pc
    elif ft == 'ridge':
        out = cp.where(sum_e > 0, pc, cp.float32(0.0))
    elif ft == 'valley':
        out = cp.where(sum_e < 0, pc, cp.float32(0.0))
    else:  # both: polarity-signed, 0.5-centred
        out = 0.5 + 0.5 * cp.sign(sum_e) * pc
    return restore_nan(out.astype(cp.float32), nan_mask)


def pc_noise_stat_func(data):
    """Global noise statistic: median of the smallest-scale amplitude."""
    valid = data[~cp.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    med = float(cp.median(valid))
    return (0.0, med if med > 1e-12 else 1e-12)


class PhaseCongruencyAlgorithm(DaskAlgorithm):
    """Amplitude-invariant feature relief from monogenic phase congruency."""

    def process(self, gpu_arr, **params):
        scales = resolve_pc_wavelengths(params.get('radii'))
        if params.get('radii') and len(scales) != len(list(params['radii'])):
            logger.info(
                "phase_congruency: wavelengths resolved from --radii: %s "
                "(clamped to [%g, %g] px)", scales, PC_MIN_WAVELENGTH,
                PC_MAX_WAVELENGTH)
        depth = min(int(2 * max(scales)) + 16, Constants.MAX_DEPTH)
        try:
            min_chunk = min(min(ax) for ax in gpu_arr.chunks)
            depth = max(1, min(depth, int(min_chunk) - 1))
        except Exception:
            pass
        stats = params.get('global_stats', None)
        return gpu_arr.map_overlap(
            compute_phase_congruency_block, depth=depth, boundary='reflect',
            dtype=cp.float32, meta=cp.empty((0, 0), dtype=cp.float32),
            wavelengths=scales,
            sigma_onf=float(params.get('sigma_onf', 0.55)),
            noise_k=float(params.get('noise_k', 2.0)),
            feature_type=params.get('feature_type', 'both'),
            normalize=True, global_stats=stats,
            pixel_size=float(params.get('pixel_size', 1.0)),
            pixel_scale_x=params.get('pixel_scale_x', None),
            pixel_scale_y=params.get('pixel_scale_y', None))

    def get_default_params(self):
        return {
            'sigma_onf': 0.55, 'noise_k': 2.0, 'feature_type': 'both',
            'mode': 'spatial', 'radii': None,
        }


__all__ = [
    "compute_phase_congruency_block", "pc_noise_stat_func",
    "resolve_pc_wavelengths", "PhaseCongruencyAlgorithm",
    "PC_MIN_WAVELENGTH", "PC_MAX_WAVELENGTH",
]
