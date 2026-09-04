from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .record import SpikeRecord

@dataclass
class Divergence:
    step: int
    only_a: np.ndarray
    only_b: np.ndarray
    def __str__(self):
        return (f"step {self.step}: {self.only_a.size} only in A {self.only_a[:8].tolist()}, "
                f"{self.only_b.size} only in B {self.only_b[:8].tolist()}")

def first_divergence(a, b, horizon=None):
    if a.n != b.n: raise ValueError("size mismatch")
    steps = min(a.n_steps, b.n_steps)
    if horizon is not None: steps = min(steps, horizon)
    for t in range(steps):
        sa, sb = a.at(t), b.at(t)
        if sa.size == sb.size and np.array_equal(sa, sb): continue
        return Divergence(t, np.setdiff1d(sa, sb), np.setdiff1d(sb, sa))
    return None

def compare(a, b, name_a="A", name_b="B", horizon=None):
    lines = [f"{name_a}: {a.summary()}", f"{name_b}: {b.summary()}"]
    d = first_divergence(a, b, horizon)
    if d is None:
        lines.append(f"IDENTICAL over {'full run' if horizon is None else f'first {horizon} steps'}")
    else:
        lines.append(f"DIVERGES {d}")
        ca, cb = a.counts_per_step, b.counts_per_step
        m = min(ca.size, cb.size)
        lines.append(f"  steps with equal spike count: {int((ca[:m]==cb[:m]).sum())}/{m}")
        lines.append(f"  total spikes: {a.total_spikes} vs {b.total_spikes}")
        na, nb = a.counts_per_neuron(), b.counts_per_neuron()
        if na.std() > 0 and nb.std() > 0:
            lines.append(f"  per-neuron count correlation: {np.corrcoef(na, nb)[0,1]:.4f}")
    return "\n".join(lines)


def assert_statistical(a, b, name_a="A", name_b="B",
                       min_divergence_step=25, rate_tol=0.10, corr_min=0.95):
    """Gate for the float baseline chain, where bit-exactness is not available:
    BLAS sums in whatever order is fastest, which need not match a sequential
    loop, and chaotic dynamics amplify the difference.

    Never use this in the exact chain. If a comparison that should be exact
    only passes here, that is a bug being tolerated.
    """
    import numpy as np

    d = first_divergence(a, b)
    if d is not None and d.step < min_divergence_step:
        raise AssertionError(
            f"diverges too early to be float reassociation: {d}\n"
            + compare(a, b, name_a, name_b))

    ra, rb = a.mean_rate_hz(), b.mean_rate_hz()
    if abs(rb - ra) > rate_tol * max(ra, 1e-9):
        raise AssertionError(f"{name_a} {ra:.2f}Hz vs {name_b} {rb:.2f}Hz "
                             f"exceeds {100*rate_tol:.0f}% tolerance")

    na, nb = a.counts_per_neuron(), b.counts_per_neuron()
    if na.std() == 0 or nb.std() == 0:
        raise AssertionError("degenerate spike counts")
    corr = float(np.corrcoef(na, nb)[0, 1])
    if corr < corr_min:
        raise AssertionError(f"per-neuron count correlation {corr:.3f} < {corr_min}")