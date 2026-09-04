"""Naive Dense Baseline on CPU.

Note: every constant is an np.float32 and every intermediate must
stay one. A bare Python float anywhere in the update promotes the expression to
float64 and silently breaks agreement with the vectorised implementations.
"""

from __future__ import annotations

import numpy as np

from ..fixedpoint import quantize, INV_SCALE
from ..record import SpikeRecord
from ..spec import LIFParams, F32


def _dense_matrix(net, arith: str):
    """W[i,j] with multi-edges summed. int64 in fixed mode so the build itself
    cannot overflow and per-element values stay well inside int32."""
    dtype = np.float32 if arith == "float" else np.int64
    W = np.zeros((net.n, net.n), dtype=dtype)
    src = np.repeat(np.arange(net.n, dtype=np.int32), net.fan_out)
    w = net.w_float if arith == "float" else net.w_fixed
    np.add.at(W, (src, net.indices), w)
    return W


def simulate(net, stim, lif: LIFParams, arith: str = "float") -> SpikeRecord:
    if arith not in ("float", "fixed"):
        raise ValueError(f"arith must be 'float' or 'fixed', got {arith!r}")

    n = net.n
    W = _dense_matrix(net, arith)

    decay = lif.decay                      # np.float32
    thresh = F32(lif.v_thresh)
    reset = F32(lif.v_reset)
    inv = INV_SCALE                        # np.float32

    # External amplitude enters the SAME accumulator as synaptic current, so in
    # fixed mode it is quantised with the same rule as the weights.
    ext_amp = (F32(stim.amp[0]) if arith == "float"
               else int(quantize(np.array([stim.amp[0]]))[0]))

    v = [reset] * n
    refrac_until = [0] * n
    prev_spikes: list[int] = []
    out: list[np.ndarray] = []

    for t in range(stim.n_steps):
        #synaptic current from spikes emitted at t-1
        acc = [0] * n if arith == "fixed" else [F32(0.0)] * n
        for i in prev_spikes:
            row = W[i]
            for j in range(n):
                x = row[j]
                if x:
                    acc[j] += x

        # external drive at t, into the same accumulator
        ids, _ = stim.at(t)
        for j in ids:
            acc[j] += ext_amp

        #integrate, threshold, reset
        spikes = []
        for i in range(n):
            if t < refrac_until[i]:
                v[i] = reset                       # clamped, input discarded
                continue
            # F32(...) before the multiply: np.int64 * np.float32 would
            # promote to float64 and break float32 discipline.
            cur = F32(acc[i]) * inv if arith == "fixed" else acc[i]
            v[i] = decay * v[i] + cur
            if v[i] >= thresh:
                v[i] = reset
                refrac_until[i] = t + lif.refrac_steps
                spikes.append(i)

        prev_spikes = spikes
        out.append(np.asarray(spikes, dtype=np.int32))

    return SpikeRecord(n, stim.n_steps, out)