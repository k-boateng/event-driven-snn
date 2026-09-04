"""Every implementation must reproduce itself exactly.

Trivial on CPU and the highest-value test in the suite once the Triton kernel
lands: races in atomic queue construction produce wrong answers rather than
crashes, and this is the only test that catches them.
"""

import pytest

from snnkern.impls import naive_dense, vectorized_dense, event_sparse_cpu


def test_naive_dense_float(net, stim, cfg):
    a = naive_dense.simulate(net, stim, cfg.lif, arith="float")
    b = naive_dense.simulate(net, stim, cfg.lif, arith="float")
    assert a.fingerprint() == b.fingerprint()


def test_naive_dense_fixed(net, stim, cfg):
    a = naive_dense.simulate(net, stim, cfg.lif, arith="fixed")
    b = naive_dense.simulate(net, stim, cfg.lif, arith="fixed")
    assert a.fingerprint() == b.fingerprint()


def test_vectorized_dense(net, stim, cfg):
    a = vectorized_dense.simulate(net, stim, cfg.lif)
    b = vectorized_dense.simulate(net, stim, cfg.lif)
    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize("forced_dense", [True, False])
def test_event_sparse_cpu(net, stim, cfg, forced_dense):
    a = event_sparse_cpu.simulate(net, stim, cfg.lif, forced_dense=forced_dense)
    b = event_sparse_cpu.simulate(net, stim, cfg.lif, forced_dense=forced_dense)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_is_sensitive(net, stim, cfg):
    """A fingerprint that never changes would make every test above vacuous."""
    from dataclasses import replace
    a = event_sparse_cpu.simulate(net, stim, cfg.lif)
    b = event_sparse_cpu.simulate(net, stim, replace(cfg.lif, v_thresh=1.05))
    assert a.fingerprint() != b.fingerprint()