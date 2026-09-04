"""Q16.16 fixed-point for synaptic current accumulation.
"""

from __future__ import annotations

import numpy as np

FRAC_BITS = 16
SCALE = np.int64(1) << FRAC_BITS            # 65536
INV_SCALE = np.float32(1.0) / np.float32(SCALE)
INT32_MAX = np.int64(2**31 - 1)


def quantize(w) -> np.ndarray:
    """float -> Q16.16 int32, rounding half away from zero.

    Called once at network build and nowhere else, so every implementation
    starts from bit-identical integer weights. Done in float64 so the rounding
    decision itself is not subject to float32 error.
    """
    w = np.asarray(w, dtype=np.float64)
    q = np.trunc(w * float(SCALE) + np.where(w >= 0, 0.5, -0.5))
    if np.any(np.abs(q) > INT32_MAX):
        raise OverflowError("weight does not fit in Q16.16 int32")
    return q.astype(np.int32)


def dequantize(q) -> np.ndarray:
    """Q16.16 int32 -> float32. Exactly once per neuron per step, at the point
    where accumulated current is added to the membrane potential."""
    return (np.asarray(q, dtype=np.float32) * INV_SCALE).astype(np.float32)


def assert_headroom(w_fixed: np.ndarray, max_fan_in: int) -> None:
    """Reject a network whose worst-case single step could wrap the accumulator.
    """
    worst = np.int64(np.abs(w_fixed).max()) * np.int64(max_fan_in)
    if worst > INT32_MAX:
        raise OverflowError(
            f"Q16.16 accumulator can overflow: worst case |current| = {worst} "
            f"> int32 max {INT32_MAX}. Lower SCALE or accumulate in int64."
        )