"""Regression tests for atmospheric_scattering's slope-angle handling.

The original implementation passed the gradient *magnitude* (a tan value) to
``cos()``/``sin()`` as if it were an angle, so the shading wrapped around on
steep slopes (e.g. tan=4 gave a darker value than tan=8).  The fixed version
uses the unit surface normal, whose response is monotonic in slope for a
fixed orientation.
"""
import pytest

cp = pytest.importorskip("cupy")


def _uniform_slope(k: float):
    h, w = 64, 64
    x = cp.arange(w, dtype=cp.float32)[None, :].repeat(h, axis=0)
    return (k * x).astype(cp.float32)


def test_no_wraparound_on_steep_slopes():
    from FujiShaderGPU.algorithms._impl_atmospheric_scattering import (
        compute_atmospheric_scattering_block,
    )

    values = []
    for k in (1.0, 2.0, 4.0, 8.0):
        out = compute_atmospheric_scattering_block(_uniform_slope(k), pixel_size=1.0)
        values.append(float(out[32, 32]))

    # On the steep side (tan >= 1) the display value must decrease monotonically
    # with slope for this east-facing plane.  The cos(magnitude) bug wrapped
    # around instead: tan=4 went fully black (0.0) and tan=8 bounced back up.
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-4, values
    assert values[-1] > 0.05, "steep slope should not collapse to black"


def test_flat_ground_value_is_finite_and_bright():
    from FujiShaderGPU.algorithms._impl_atmospheric_scattering import (
        compute_atmospheric_scattering_block,
    )

    out = compute_atmospheric_scattering_block(cp.zeros((16, 16), dtype=cp.float32))
    v = float(out[8, 8])
    assert 0.5 < v <= 1.0
