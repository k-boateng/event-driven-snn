"""Simulation parameters which are read by every implementation
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace, field
import numpy as np

F32 = np.float32
I32 = np.int32


@dataclass(frozen=True)
class LIFParams:
    tau_m: float = 20.0        # membrane time constant, ms
    dt: float = 1.0            # timestep, ms
    v_thresh: float = 1.0
    v_reset: float = 0.0
    refrac_steps: int = 2

    @property
    def decay(self) -> np.float32:
        """exp(-dt/tau) in float32 evaluated once so every backend uses the same bits."""
        return F32(np.exp(-self.dt / self.tau_m))


@dataclass(frozen=True)
class NetParams:
    n: int = 10_000
    fan_out: int = 100         # fixed out-degree K
    exc_frac: float = 0.8
    w_exc: float = 0.05
    w_inh: float = -0.20       # placeholder
    seed: int = 0


@dataclass(frozen=True)
class StimParams:
    n_steps: int = 1000
    kick_frac: float = 0.002   # fraction of neurons kicked per step
    kick_amp: float = 1.5      # supra-threshold: a kick alone fires the neuron
    seed: int = 1


@dataclass(frozen=True)
class Config:
    lif: LIFParams = field(default_factory=LIFParams)
    net: NetParams = field(default_factory=NetParams)
    stim: StimParams = field(default_factory=StimParams)

    def scaled(self, n: int, n_steps: int | None = None) -> "Config":
        """Same config at a different size. Used by the benchmark sweep."""
        return replace(
            self,
            net=replace(self.net, n=n),
            stim=self.stim if n_steps is None else replace(self.stim, n_steps=n_steps),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        """Identifier for frozen data filenames. Every field that changes the
        generated network or stimulus must appear here, or you will silently
        load the wrong .npz and benchmark against the wrong experiment."""
        return (f"n{self.net.n}_k{self.net.fan_out}_s{self.net.seed}"
                f"_T{self.stim.n_steps}_kf{self.stim.kick_frac}_ss{self.stim.seed}")


def test_config(n: int = 500, n_steps: int = 300) -> Config:
        """Small network where recurrent propagation is necessary.
        """
        return Config(
            net=NetParams(n=n, fan_out=100, w_exc=0.10, w_inh=-0.40, seed=0),
            stim=StimParams(n_steps=n_steps, kick_frac=0.01, seed=1),
        )


#Benchmark configurations 
BENCH_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
 
 
def bench_config(n: int, n_steps: int = 1000, kick_frac: float = 0.002) -> Config:
    """Benchmark network at size n. kick_frac*1000 Hz is the stimulus rate;
    measured rate at or below that means the network contributed nothing."""
    return Config(
        net=NetParams(n=n, fan_out=100, w_exc=0.10, w_inh=-0.40, seed=0),
        stim=StimParams(n_steps=n_steps, kick_frac=kick_frac, kick_amp=1.5, seed=1),
    )

def decay_table(decay, size: int = 512) -> np.ndarray:
    """decay^k for k in [0, size), built by repeated float32 multiplication.

    Exists because decay^elapsed must be computed identically on every backend.
    """
    t = np.empty(size, dtype=F32)
    t[0] = F32(1.0)
    for k in range(1, size):
        t[k] = F32(t[k - 1] * decay)
    return t


DECAY_TABLE_SIZE = 512