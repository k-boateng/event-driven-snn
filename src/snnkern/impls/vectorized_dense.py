"""Vectorized dense CPU (NumPy).

This is kept float only deliberately. Since NumPy has no BLAS path for integer matmul .i.e int32 @
int32 falls back to a generic loop ~20x slower than the float path. A
fixed-point version here would be a baseline whose slowness is because of
NumPy's dispatch rather than of the dense algorithm, and every speedup measured
against it would be inflated.

Cost per step: 4*n^2 bytes of weight traffic, independent of activity. Two FLOPs
per four bytes read, so it is hard memory-bound.

Memory: the dense matrix is 4*n^2 bytes. n=30k is 3.6 GB; this does not scale
past ~30k neurons on any machine.
"""

from __future__ import annotations

import numpy as np

from ..record import SpikeRecord
from ..spec import LIFParams, F32


def simulate(net, stim, lif: LIFParams) -> SpikeRecord:
    n = net.n
    W = net.dense_float()          # multi-edges summed; see network.dense_float

    decay = lif.decay
    thresh = F32(lif.v_thresh)
    reset = F32(lif.v_reset)

    v = np.full(n, reset, dtype=F32)
    refrac_until = np.zeros(n, dtype=np.int32)
    s_prev = np.zeros(n, dtype=F32)     # spike vector from step t-1
    out: list[np.ndarray] = []

    for t in range(stim.n_steps):
        # synaptic current from spikes emitted at t-1.
        # s_prev is 0/1, so this sums the rows of W belonging to neurons that
        # fired. Expressed as a GEMV so it dispatches to BLAS.
        i_syn = s_prev @ W

        # external drive at t. bincount, not add.at: add.at is the unbuffered
        # ufunc fallback and ~20x slower, which would inflate this baseline.
        ids, amp = stim.at(t)
        i_syn += np.bincount(ids, weights=amp, minlength=n).astype(F32)

        # integrate where free, clamp to reset where refractory.
        free = t >= refrac_until
        v = np.where(free, decay * v + i_syn, reset).astype(F32)

        # threshold. The `free` term is redundant -- a clamped neuron sits at
        # reset, below threshold -- but it keeps the line readable in isolation
        # and matches the written contract.
        fired = free & (v >= thresh)

        # reset and refractory
        v[fired] = reset
        idx = np.flatnonzero(fired).astype(np.int32)
        refrac_until[idx] = t + lif.refrac_steps

        s_prev = fired.astype(F32)
        out.append(idx)

    return SpikeRecord(n, stim.n_steps, out)