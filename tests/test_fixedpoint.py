"""Q16.16 conversion and the associativity property the whole exact chain
depends on."""

import numpy as np
import pytest

from snnkern.fixedpoint import quantize, dequantize, assert_headroom, SCALE, INT32_MAX


def test_roundtrip_within_resolution():
    w = np.array([0.05, -0.20, 0.0, 1.5, -3.25])
    err = np.abs(dequantize(quantize(w)) - w.astype(np.float32))
    assert np.all(err <= 1.0 / float(SCALE))


def test_dtypes():
    assert quantize(np.array([0.05])).dtype == np.int32
    assert dequantize(np.array([3277], dtype=np.int32)).dtype == np.float32


def test_rounding_is_half_away_from_zero():
    """Not banker's rounding: ties must go away from zero, so the rule is
    reimplementable on any backend without matching NumPy's tie handling."""
    half = 0.5 / float(SCALE)
    assert int(quantize(np.array([half]))[0]) == 1
    assert int(quantize(np.array([-half]))[0]) == -1


def test_integer_accumulation_is_order_independent():
    """The property the exact chain rests on. Float fails this; int does not."""
    rng = np.random.default_rng(0)
    vals = quantize(rng.normal(0, 0.05, 10_000))
    sums = {int(vals[rng.permutation(vals.size)].sum()) for _ in range(20)}
    assert len(sums) == 1

    f = rng.normal(0, 0.05, 10_000).astype(np.float32)
    fsums = {float(f[rng.permutation(f.size)].sum()) for _ in range(20)}
    assert len(fsums) > 1, "float32 summation was order-independent here; the " \
                           "motivation for fixed-point is not being demonstrated"


def test_headroom_accepts_realistic_network(net):
    assert_headroom(net.w_fixed, int(net.in_degrees().max()))


def test_headroom_rejects_overflow():
    big = np.array([INT32_MAX // 100], dtype=np.int32)
    with pytest.raises(OverflowError):
        assert_headroom(big, 1000)


def test_quantize_rejects_unrepresentable_weight():
    with pytest.raises(OverflowError):
        quantize(np.array([1e6]))