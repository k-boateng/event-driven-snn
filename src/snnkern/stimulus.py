"""Frozen external drive.
"""

from __future__ import annotations

import numpy as np

from .spec import StimParams, F32


class Stimulus:
    """Sparse external drive, frozen. Treat as immutable."""

    def __init__(self, n, n_steps, k, ids, amp):
        self.n = int(n)
        self.n_steps = int(n_steps)
        self.k = int(k)             # kicks per step, constant
        self.ids = ids              # int32 [n_steps * k]
        self.amp = amp              # f32   [n_steps * k]

    def at(self, t: int):
        """(neuron_ids, amplitudes) kicked at step t. Constant k per step makes
        the offsets arithmetic, same as the network's fixed fan-out."""
        lo = t * self.k
        return self.ids[lo:lo + self.k], self.amp[lo:lo + self.k]

    def dense_step(self, t: int) -> np.ndarray:
        """Scattered into a length-n vector, for the dense implementations.
        Uses bincount, so a neuron drawn twice in one step gets the amplitude
        twice and matches what the sparse path accumulates."""
        ids, amp = self.at(t)
        return np.bincount(ids, weights=amp, minlength=self.n).astype(F32)

    def save(self, path) -> None:
        np.savez(path, n=self.n, n_steps=self.n_steps, k=self.k,
                 ids=self.ids, amp=self.amp)

    @staticmethod
    def load(path) -> "Stimulus":
        d = np.load(path)
        return Stimulus(int(d["n"]), int(d["n_steps"]), int(d["k"]),
                        d["ids"], d["amp"])


def build(n: int, p: StimParams) -> Stimulus:
    rng = np.random.default_rng(p.seed)
    k = max(1, int(round(p.kick_frac * n)))

    # With replacement, both within a step and across steps. A neuron drawn
    # twice in one step receives the amplitude twice; that is a legitimate
    # event stream, and rejecting duplicates would need a per-step uniqueness
    # pass that does not vectorise.
    ids = rng.integers(0, n, size=p.n_steps * k, dtype=np.int32)
    amp = np.full(p.n_steps * k, p.kick_amp, dtype=F32)
    return Stimulus(n, p.n_steps, k, ids, amp)