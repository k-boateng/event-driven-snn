"""
  SCATTER       propagate only from neurons that fired, through their CSR
                out-edge slice. O(spikes * K) instead of O(n^2).

  ACTIVE SET    update only neurons that received current this step.

  LAZY DECAY    a neuron's potential is reconstructed on demand as
                v * decay^(t - last_update), rather than decayed every step.
                Exact for LIF: decay only moves v toward rest, so an untouched
                neuron cannot cross threshold and skipping it loses nothing.
"""

from __future__ import annotations

import numpy as np

from ..fixedpoint import quantize, INV_SCALE
from ..record import SpikeRecord
from ..spec import LIFParams, F32, decay_table, DECAY_TABLE_SIZE


def simulate(net, stim, lif: LIFParams, forced_dense: bool = False, return_state: bool = False) -> SpikeRecord:
    n, K = net.n, net.fan_out

    # Fixed fan-out means a neuron's out-edges are a contiguous row, so the
    # whole firing set's edges gather in one fancy-index. This is the payoff
    # for storing no indptr.
    idx2d = net.indices.reshape(n, K)
    w2d = net.w_fixed.reshape(n, K)

    decay = lif.decay
    thresh = F32(lif.v_thresh)
    reset = F32(lif.v_reset)
    inv = INV_SCALE

    amps = np.unique(stim.amp)
    if amps.size != 1:
        raise ValueError("non-uniform kick amplitude not supported")
    ext_amp = float(quantize(np.array([amps[0]]))[0])

    v = np.full(n, reset, dtype=F32)
    refrac_until = np.zeros(n, dtype=np.int32)
    # -1, not 0: at t=0 the dense path applies one decay to the initial
    # potential, so elapsed must be 1 on the first touch. Invisible when
    # v_reset == 0 (zero times anything is zero) and wrong otherwise.
    last_update = np.full(n, -1, dtype=np.int32)

    all_neurons = np.arange(n, dtype=np.int32)
    prev_spikes = np.zeros(0, dtype=np.int32)
    out: list[np.ndarray] = []
    # One definition of decay^k, shared with the Triton kernel. See
    # spec.decay_table for why this is not np.power.
    dtable = decay_table(decay)

    for t in range(stim.n_steps):
        # scatter: current from spikes emitted at t-1
        # bincount returns float64 when weights are given -- EXCEPT when the
        # weights array is empty, where it returns int64. That happens on step 0
        # before anything has fired, so the dtype is forced rather than assumed.
        # Values here are ~1e6, far inside float64's exact-integer range, so the
        # accumulation is lossless and stays associative: the fixed-point
        # reproducibility guarantee is preserved.
        tgt = idx2d[prev_spikes].ravel()
        wts = w2d[prev_spikes].ravel()
        acc = np.bincount(tgt, weights=wts, minlength=n).astype(np.float64)

        #external drive at t, into the same accumulator
        ids, _ = stim.at(t)
        acc += np.bincount(ids, weights=np.full(ids.size, ext_amp), minlength=n)

        #active set
        # flatnonzero collapses duplicate targets automatically: a neuron fed
        # by three sources appears once.
        active = all_neurons if forced_dense else np.flatnonzero(acc).astype(np.int32)
        if active.size == 0:
            out.append(np.zeros(0, dtype=np.int32))
            prev_spikes = out[-1]
            continue

        #update the active set
        free = t >= refrac_until[active]
        clamped = active[~free]
        act = active[free]

        # Refractory neurons are pinned to reset and their input is discarded.
        v[clamped] = reset

        if act.size:
            if forced_dense:
                base = np.float64(decay)
            else:
                elapsed = np.minimum(t - last_update[act], DECAY_TABLE_SIZE - 1)
                base = np.float64(dtable[elapsed])

            # Single rounding, matching Triton's fused multiply-add.
            cur64 = np.float64(acc[act].astype(np.int32).astype(F32)) * np.float64(inv)
            vv = (base * np.float64(v[act]) + cur64).astype(F32)

            fired = vv >= thresh
            vv[fired] = reset
            v[act] = vv
            last_update[act] = t

            sp = act[fired]
            refrac_until[sp] = t + lif.refrac_steps
            # v is known to stay at reset through the whole refractory window,
            # so record validity as of its LAST clamped step. Reconstruction
            # then resumes from the right base even when reset != 0.
            last_update[sp] = t + lif.refrac_steps - 1
        else:
            sp = np.zeros(0, dtype=np.int32)

        out.append(np.sort(sp).astype(np.int32))
        prev_spikes = out[-1]

    rec = SpikeRecord(n, stim.n_steps, out)

    if return_state:
        # Debugging hook: spike trains diverge long after the membrane
        # potentials do, so comparing v is how you find where a mismatch
        # actually starts.
        return rec, {"v": v.copy(), "last": last_update.copy(),
                     "refrac": refrac_until.copy()}
    return rec