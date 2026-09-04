"""Dense GPU baseline (PyTorch, fp32).
"""

from __future__ import annotations

import numpy as np

from ..record import SpikeRecord
from ..spec import LIFParams


def build_dense(net, device, dtype=None):
    """Scatter the CSR edges into a dense matrix on the device.

    index_put_ with accumulate=True SUMS duplicate edges. Plain assignment
    would drop them and disagree with every other implementation on a handful
    of synapses which is a spec mismatch that looks exactly like a kernel bug.
    """
    import torch

    dtype = dtype or torch.float32
    n, K = net.n, net.fan_out
    W = torch.zeros((n, n), dtype=dtype, device=device)
    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(K)
    tgt = torch.from_numpy(net.indices.astype(np.int64)).to(device)
    w = torch.from_numpy(net.w_float).to(device=device, dtype=dtype)
    W.index_put_((src, tgt), w, accumulate=True)
    return W


def simulate(net, stim, lif: LIFParams, device: str = "cuda",
             record: bool = True, W=None) -> SpikeRecord | None:
    import torch

    n = net.n
    dev = torch.device(device)
    # Caller may pass a prebuilt matrix so that construction stays OUTSIDE the
    # timed region. At n=1.3e5 the matrix is 68 GB; rebuilding it once per
    # warmup and repeat both contaminates the measurement and risks OOM from
    # allocator fragmentation.
    W = build_dense(net, dev) if W is None else W

    decay = float(lif.decay)          # float32 value, widened for the scalar op
    thresh = float(lif.v_thresh)
    reset = float(lif.v_reset)

    v = torch.full((n,), reset, dtype=torch.float32, device=dev)
    refrac_until = torch.zeros(n, dtype=torch.int32, device=dev)
    s_prev = torch.zeros(n, dtype=torch.float32, device=dev)

    # Whole frozen stimulus resident on the device: transferring per step would
    # measure PCIe, not the algorithm.
    stim_ids = torch.from_numpy(stim.ids.astype(np.int64)).to(dev)
    stim_amp = torch.from_numpy(stim.amp).to(dev)
    k = stim.k

    fired_all = (torch.zeros((stim.n_steps, n), dtype=torch.bool, device=dev)
                 if record else None)

    for t in range(stim.n_steps):
        i_syn = s_prev @ W                       # dense GEMV, 4*n^2 bytes read

        lo = t * k
        i_syn.index_add_(0, stim_ids[lo:lo + k], stim_amp[lo:lo + k])

        free = refrac_until <= t
        v = torch.where(free, decay * v + i_syn,
                        torch.full_like(v, reset))

        fired = free & (v >= thresh)
        v = torch.where(fired, torch.full_like(v, reset), v)
        refrac_until = torch.where(fired,
                                   torch.full_like(refrac_until, t + lif.refrac_steps),
                                   refrac_until)

        s_prev = fired.to(torch.float32)
        if record:
            fired_all[t] = fired

    if not record:
        return None

    # One host transfer, after the loop.
    fa = fired_all.cpu().numpy()
    per_step = [np.flatnonzero(fa[t]).astype(np.int32) for t in range(stim.n_steps)]
    return SpikeRecord(n, stim.n_steps, per_step)