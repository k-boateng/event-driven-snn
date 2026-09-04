"""Connectivity and stimulus generation."""

import numpy as np
import pytest

from snnkern.spec import NetParams, StimParams
from snnkern.network import build as build_net
from snnkern.stimulus import build as build_stim


def test_no_self_edges(net):
    src = np.repeat(np.arange(net.n, dtype=np.int32), net.fan_out)
    assert int((src == net.indices).sum()) == 0


def test_fixed_out_degree(net):
    assert net.n_edges == net.n * net.fan_out
    assert net.indices.dtype == np.int32


def test_targets_in_range(net):
    assert net.indices.min() >= 0
    assert net.indices.max() < net.n


def test_dale_law(net):
    """All of a neuron's out-edges share one weight, and its sign matches
    its excitatory/inhibitory type."""
    w2d = net.w_float.reshape(net.n, net.fan_out)
    assert np.all(w2d == w2d[:, :1]), "a neuron's out-edges do not share a weight"
    assert np.all(w2d[net.is_exc] > 0)
    assert np.all(w2d[~net.is_exc] < 0)


def test_dense_sums_multi_edges(net):
    """dense_float must SUM duplicate edges, not overwrite them. Assignment
    would drop them and make the dense and sparse paths disagree on a handful
    of synapses -- a spec mismatch that looks exactly like a kernel bug."""
    W = net.dense_float()
    row_sums = W.sum(axis=1)
    sparse_sums = net.w_float.reshape(net.n, net.fan_out).sum(axis=1)
    assert np.allclose(row_sums, sparse_sums, rtol=1e-5)


def test_multi_edges_exist(net):
    """If duplicates ever stop occurring, test_dense_sums_multi_edges silently
    stops testing anything."""
    dup = sum(net.fan_out - np.unique(net.indices[net.row(i)]).size
              for i in range(net.n))
    assert dup > 0, "no duplicate edges: the summing test is now vacuous"


def test_build_is_deterministic():
    a = build_net(NetParams(n=200, fan_out=20, seed=7))
    b = build_net(NetParams(n=200, fan_out=20, seed=7))
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.w_fixed, b.w_fixed)
    assert np.array_equal(a.is_exc, b.is_exc)


def test_different_seeds_differ():
    a = build_net(NetParams(n=200, fan_out=20, seed=0))
    b = build_net(NetParams(n=200, fan_out=20, seed=1))
    assert not np.array_equal(a.indices, b.indices)


def test_fan_out_must_fit():
    with pytest.raises(ValueError):
        build_net(NetParams(n=10, fan_out=10))


def test_stimulus_shape_and_determinism():
    p = StimParams(n_steps=50, kick_frac=0.01, seed=3)
    a = build_stim(400, p)
    b = build_stim(400, p)
    assert a.k == 4 and a.ids.size == 50 * 4
    assert np.array_equal(a.ids, b.ids)


def test_stimulus_dense_matches_events(stim):
    """dense_step must accumulate duplicates, matching what the sparse path
    adds into the accumulator."""
    for t in (0, 1, 17):
        ids, amp = stim.at(t)
        assert np.isclose(stim.dense_step(t).sum(), amp.sum())