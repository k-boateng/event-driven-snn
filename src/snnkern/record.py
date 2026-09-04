"""Spike trains, stored ragged.

Ids are sorted within each step. The Triton kernel appends to its spike queue
with atomic_add on a counter, so queue order depends on thread scheduling and
differs run to run. Sorting makes comparison order-independent.
"""

from __future__ import annotations

import hashlib
import numpy as np


class SpikeRecord:
    def __init__(self, n: int, n_steps: int, per_step: list[np.ndarray]):
        self.n = int(n)
        self.n_steps = int(n_steps)
        counts = np.fromiter((a.size for a in per_step), dtype=np.int64,
                             count=len(per_step))
        self.ptr = np.zeros(n_steps + 1, dtype=np.int64)
        np.cumsum(counts, out=self.ptr[1:])
        self.ids = (np.concatenate([np.sort(a) for a in per_step]).astype(np.int32)
                    if per_step else np.zeros(0, dtype=np.int32))

    def at(self, t: int) -> np.ndarray:
        return self.ids[self.ptr[t]:self.ptr[t + 1]]

    @property
    def total_spikes(self) -> int:
        return int(self.ids.size)

    @property
    def counts_per_step(self) -> np.ndarray:
        return np.diff(self.ptr)

    def mean_rate_hz(self, dt_ms: float = 1.0) -> float:
        """Mean firing rate."""
        return 1000.0 * self.total_spikes / (self.n * self.n_steps * dt_ms)

    def counts_per_neuron(self) -> np.ndarray:
        return np.bincount(self.ids, minlength=self.n)

    def isi_cv(self) -> float:
        """Coefficient of variation of inter-spike intervals, pooled over
        neurons with at least 3 spikes."""
        order = np.lexsort((self._step_of_spike(), self.ids))
        nid = self.ids[order]
        step = self._step_of_spike()[order]
        isi = np.diff(step)
        same = nid[1:] == nid[:-1]
        isi = isi[same]
        if isi.size < 10:
            return float("nan")
        return float(isi.std() / isi.mean())

    def _step_of_spike(self) -> np.ndarray:
        """Step index for each entry of ids."""
        return np.repeat(np.arange(self.n_steps, dtype=np.int32),
                         self.counts_per_step)

    def fingerprint(self) -> str:
        """Short hash of the full spike train."""
        h = hashlib.blake2b(digest_size=8)
        h.update(self.ptr.tobytes())
        h.update(self.ids.tobytes())
        return h.hexdigest()

    def summary(self) -> str:
        return (f"spikes={self.total_spikes} rate={self.mean_rate_hz():.2f}Hz "
                f"cv={self.isi_cv():.2f} fp={self.fingerprint()}")