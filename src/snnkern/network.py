"""Fixed random sparse connectivity, generated and frozen to disk.
"""

from __future__ import annotations

import numpy as np

from .fixedpoint import quantize, assert_headroom
from .spec import NetParams, F32


class Network:
    """Frozen connectivity. Treat as immutable after construction."""

    def __init__(self, n, fan_out, indices, w_fixed, w_float, is_exc):
        self.n = int(n)
        self.fan_out = int(fan_out)
        self.indices = indices      # int32 [n*K]  targets, CSR order by source
        self.w_fixed = w_fixed      # int32 [n*K]  Q16.16,
        self.w_float = w_float      # f32   [n*K]  dense baselines
        self.is_exc = is_exc        # bool  [n]

    def row(self, i: int) -> slice:
        """Edge slice for source neuron i."""
        return slice(i * self.fan_out, (i + 1) * self.fan_out)

    @property
    def n_edges(self) -> int:
        return int(self.indices.size)

    @property
    def density(self) -> float:
        return self.n_edges / (self.n * self.n)

    def in_degrees(self) -> np.ndarray:
        """Per-neuron fan-in. Out-degree is fixed at K; in-degree is not and
        is Binomial(n*K, 1/n), so it has a tail."""
        return np.bincount(self.indices, minlength=self.n)

    def dense_float(self) -> np.ndarray:
        """W[i, j] for the dense implementations. Only viable below n ~ 30k.

        Multi-edges are summed and not overwritten. Assignment would silently drop
        the duplicate and make the dense and sparse paths disagree on a handful
        of synapses which a specification mismatch that looks exactly like a bug.
        """
        need = 4 * self.n * self.n
        if need > 8e9:
            raise MemoryError(
                f"dense W at n={self.n} needs {need/1e9:.1f} GB; "
                "dense implementations do not scale past ~30k neurons"
            )
        W = np.zeros((self.n, self.n), dtype=F32)
        src = np.repeat(np.arange(self.n, dtype=np.int32), self.fan_out)
        np.add.at(W, (src, self.indices), self.w_float)
        return W

    def save(self, path) -> None:
        np.savez(path, n=self.n, fan_out=self.fan_out, indices=self.indices,
                 w_fixed=self.w_fixed, w_float=self.w_float, is_exc=self.is_exc)

    @staticmethod
    def load(path) -> "Network":
        d = np.load(path)
        return Network(int(d["n"]), int(d["fan_out"]), d["indices"],
                       d["w_fixed"], d["w_float"], d["is_exc"])


def build(p: NetParams) -> Network:
    rng = np.random.default_rng(p.seed)
    n, k = p.n, p.fan_out
    if k >= n:
        raise ValueError(f"fan_out {k} must be < n {n}")

    is_exc = np.zeros(n, dtype=bool)
    is_exc[rng.permutation(n)[:int(round(p.exc_frac * n))]] = True

    # Draw from [0, n-1), then shift any draw >= i up by one. Skips i exactly,
    # uniform over the other n-1 neurons, no rejection loop, one vectorised
    # pass over the whole matrix. Multi-edges within a row are permitted.
        # Generate in chunks: the (n, K) int64 temporary plus its boolean
    # broadcast peak at ~2.5x the final array size, which is what OOMs at
    # n=1e6. Chunking caps the transient at CHUNK*K elements.
    indices = np.empty(n * k, dtype=np.int32)
    CHUNK = max(1, 10_000_000 // k)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        t = rng.integers(0, n - 1, size=(hi - lo, k), dtype=np.int64)
        t += (t >= np.arange(lo, hi, dtype=np.int64)[:, None])
        indices[lo * k:hi * k] = t.astype(np.int32).ravel()

    # Dale's law: sign is a property of the presynaptic neuron, so all of a
    # neuron's out-edges carry the same weight.
    w_row = np.where(is_exc, p.w_exc, p.w_inh)
    w_float = np.repeat(w_row, k).astype(F32)
    w_fixed = quantize(np.repeat(w_row, k))

    net = Network(n, k, indices, w_fixed, w_float, is_exc)
    assert_headroom(w_fixed, int(net.in_degrees().max()))
    return net