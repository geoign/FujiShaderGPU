import pytest

from FujiShaderGPU.algorithms.common.spatial_mode import (
    AUTO_RADII_SEQUENCE,
    AUTO_RADIUS_MAX,
    LOCAL_RADII,
    LOCAL_WEIGHTS,
    MULTISCALE_REQUIRED_ALGOS,
    RADII_DRIVEN_ALGOS,
    auto_spatial_profile,
    auto_spatial_radii,
    auto_spatial_weights,
)


def test_local_profile_is_single_pixel():
    assert LOCAL_RADII == [1]
    assert LOCAL_WEIGHTS == [1.0]


def test_multiscale_required_algos_are_excluded_from_radii_driven():
    # The single-scale (local) rule must never be applied to algorithms that are
    # undefined at one scale; they fall back to spatial instead.
    assert MULTISCALE_REQUIRED_ALGOS == {
        "fractal_anomaly", "scale_space_surprise", "visual_saliency",
        "scale_drift", "phase_congruency",
    }
    assert MULTISCALE_REQUIRED_ALGOS.isdisjoint(RADII_DRIVEN_ALGOS)


@pytest.mark.parametrize(
    "short_side, expected",
    [
        (30000, [2, 8, 32, 128, 512, 2048]),  # /10 huge -> capped at 2048
        (20480, [2, 8, 32, 128, 512, 2048]),  # 2048 == short/10 -> included
        (10000, [2, 8, 32, 128, 512]),        # 2048 > 1000 dropped
        (5120, [2, 8, 32, 128, 512]),         # 512 == short/10 -> included (boundary)
        (5119, [2, 8, 32, 128]),              # 512 > 511.9 dropped
        (300, [2, 8]),                        # 32 > 30 dropped
        (15, [2]),                            # short/10 < 2 -> keep at least one
    ],
)
def test_auto_radii_truncated_by_short_side(short_side, expected):
    assert auto_spatial_radii(short_side) == expected


def test_auto_radii_cap_is_2048():
    assert max(auto_spatial_radii(10**9)) == AUTO_RADIUS_MAX
    assert max(AUTO_RADII_SEQUENCE) == AUTO_RADIUS_MAX


def test_auto_radii_none_returns_full_sequence():
    assert auto_spatial_radii(None) == list(AUTO_RADII_SEQUENCE)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_auto_weights_are_pow2_normalized_and_descending(n):
    w = auto_spatial_weights(n)
    assert len(w) == n
    assert sum(w) == pytest.approx(1.0)
    # Nearer (smaller) radii weigh more: strictly descending.
    assert all(w[i] > w[i + 1] for i in range(n - 1))
    # 2**n ratio: each weight is double the next.
    for i in range(n - 1):
        assert w[i] == pytest.approx(2.0 * w[i + 1])


def test_profile_honours_explicit_radii_with_pow2_weights():
    radii, weights = auto_spatial_profile(10000, radii=[4, 16, 64, 256])
    assert radii == [4, 16, 64, 256]
    assert weights == pytest.approx([8 / 15, 4 / 15, 2 / 15, 1 / 15])
