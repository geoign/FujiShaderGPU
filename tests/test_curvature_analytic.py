"""Analytic regression tests for planform / profile curvature.

For ``z = a*(x - x0)**2`` the contours are straight vertical lines, so the
plan (contour) curvature is exactly zero while the profile curvature (along
the gradient) is non-zero.  A regression here means the two formulas are
swapped (the original bug) or otherwise wrong.
"""
import pytest

cp = pytest.importorskip("cupy")

# Display value of zero curvature: tanh(0)=0 -> (0+1)/2=0.5 -> gamma 1/2.2.
_ZERO_DISP = 0.5 ** (1.0 / 2.2)


def _parabolic_cylinder():
    h, w = 64, 64
    x = cp.arange(w, dtype=cp.float32)[None, :].repeat(h, axis=0)
    return (0.01 * (x - 32.0) ** 2).astype(cp.float32)


def _interior(arr):
    return arr[20:44, 20:44]


def test_planform_zero_on_straight_contours():
    from FujiShaderGPU.algorithms._impl_curvature import compute_curvature_block

    out = compute_curvature_block(_parabolic_cylinder(), curvature_type="planform")
    dev = float(cp.abs(_interior(out) - _ZERO_DISP).max())
    assert dev < 0.02


def test_profile_nonzero_on_parabolic_slope():
    from FujiShaderGPU.algorithms._impl_curvature import compute_curvature_block

    out = compute_curvature_block(_parabolic_cylinder(), curvature_type="profile")
    dev = float(cp.abs(_interior(out) - _ZERO_DISP).max())
    assert dev > 0.05


def test_dome_has_both_curvatures():
    from FujiShaderGPU.algorithms._impl_curvature import compute_curvature_block

    h, w = 64, 64
    y = cp.arange(h, dtype=cp.float32)[:, None].repeat(w, axis=1)
    x = cp.arange(w, dtype=cp.float32)[None, :].repeat(h, axis=0)
    dome = (-0.01 * ((x - 32.0) ** 2 + (y - 32.0) ** 2)).astype(cp.float32)
    for ct in ("planform", "profile"):
        out = compute_curvature_block(dome, curvature_type=ct)
        # Exclude the apex (gradient ~0 makes plan/profile ill-defined there).
        ring = out[20:44, 20:44]
        dev = float(cp.abs(ring - _ZERO_DISP).max())
        assert dev > 0.05, ct
