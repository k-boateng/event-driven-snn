"""Meta-tests: the fixture must be capable of failing the other tests.

A network where every spike is externally injected cannot detect a bug in
synaptic propagation -- the spike train would be identical with the scatter
deleted entirely. These assert the fixture is not in that state.
"""

import numpy as np

from snnkern.impls import event_sparse_cpu


def test_network_is_alive(net, stim, cfg):
    r = event_sparse_cpu.simulate(net, stim, cfg.lif)
    assert r.total_spikes > 0, "network is silent"
    assert r.mean_rate_hz() > 1.0, f"rate {r.mean_rate_hz():.2f}Hz is near-dead"


def test_propagation_is_load_bearing(net, stim, cfg):
    """Most spikes must be caused by other neurons, not by the stimulus."""
    r = event_sparse_cpu.simulate(net, stim, cfg.lif)
    kicked = {(t, int(i)) for t in range(stim.n_steps) for i in stim.at(t)[0]}
    fired = {(t, int(i)) for t in range(r.n_steps) for i in r.at(t)}
    network_driven = len(fired - kicked)
    assert network_driven > len(fired) * 0.5, (
        f"only {network_driven}/{len(fired)} spikes are network-driven; "
        "a bug in the scatter would not change the spike train"
    )


def test_lazy_decay_is_exercised(net, stim, cfg):
    """Some neurons must go untouched for more than one step, or the
    decay**elapsed path never runs with elapsed > 1."""
    r = event_sparse_cpu.simulate(net, stim, cfg.lif)
    last = np.full(net.n, -1, dtype=np.int64)
    multi = total = 0
    for t in range(stim.n_steps):
        touched = set(stim.at(t)[0].tolist())
        if t > 0:
            for src in r.at(t - 1):
                touched.update(net.indices[net.row(int(src))].tolist())
        for i in touched:
            total += 1
            if t - last[i] > 1:
                multi += 1
            last[i] = t
    assert total > 0
    assert multi / total > 0.05, f"only {100*multi/total:.1f}% of touches have elapsed>1"