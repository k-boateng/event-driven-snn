"""The correctness gate.

Two chains, per notes/SEMANTICS.md:

  EXACT (fixed-point): naive_dense(fixed) == event_sparse_cpu(forced_dense)
                       == event_sparse_cpu(production)
  BASELINE (float32):  naive_dense(float) ~ vectorized_dense
                       -- agreement in practice, not a guarantee: BLAS sums in
                       whatever order is fastest, which need not match the
                       sequential loop.

Every equivalence is also run with v_reset != 0 and with a longer refractory.
v_reset == 0 hides an entire class of initialisation bug: a wrong last_update
initial value is invisible because zero times anything is zero. That bug was
found this way and the parametrisation exists to keep it found.
"""

from dataclasses import replace

import pytest

from snnkern import check
from snnkern.spec import LIFParams
from snnkern.impls import naive_dense, vectorized_dense, event_sparse_cpu


LIF_VARIANTS = {
    "default": LIFParams(),
    "reset_nonzero": LIFParams(v_reset=0.1),
    "long_refrac": LIFParams(refrac_steps=5),
    "no_refrac": LIFParams(refrac_steps=0),
}


@pytest.fixture(params=sorted(LIF_VARIANTS), scope="session")
def lif(request):
    return LIF_VARIANTS[request.param]


# --- exact chain ---------------------------------------------------------

def test_forced_dense_matches_naive(net, stim, lif):
    """Validates the sparse data structures independently of lazy decay."""
    ref = naive_dense.simulate(net, stim, lif, arith="fixed")
    got = event_sparse_cpu.simulate(net, stim, lif, forced_dense=True)
    assert check.first_divergence(ref, got) is None, \
        check.compare(ref, got, "naive_dense(fixed)", "forced_dense")


def test_production_matches_naive(net, stim, lif):
    """The full claim: active set + lazy decay change nothing observable."""
    ref = naive_dense.simulate(net, stim, lif, arith="fixed")
    got = event_sparse_cpu.simulate(net, stim, lif)
    assert check.first_divergence(ref, got) is None, \
        check.compare(ref, got, "naive_dense(fixed)", "event_sparse_cpu")


def test_production_matches_forced_dense(net, stim, lif):
    """Isolates lazy-decay bookkeeping from CSR indexing: if this fails while
    test_forced_dense_matches_naive passes, the bug is in last_update."""
    a = event_sparse_cpu.simulate(net, stim, lif, forced_dense=True)
    b = event_sparse_cpu.simulate(net, stim, lif)
    assert check.first_divergence(a, b) is None, \
        check.compare(a, b, "forced_dense", "production")


# --- baseline chain ------------------------------------------------------

def test_vectorized_dense_matches_naive_float(net, stim, lif):
    """Statistical, not exact: BLAS sums in whatever order is fastest, which
    need not match the sequential loop, and chaos amplifies the difference.
    Measured here: divergence at step 92, one neuron, per-neuron count
    correlation 0.987.

    The gate still requires exact agreement for the first 25 steps, so an early
    systematic bug -- a wrong sign, a dropped term -- is caught. Only late
    single-neuron drift is tolerated.
    """
    ref = naive_dense.simulate(net, stim, lif, arith="float")
    got = vectorized_dense.simulate(net, stim, lif)
    check.assert_statistical(ref, got, "naive_dense(float)", "vectorized_dense")


# --- the chains are genuinely different ----------------------------------

def test_float_and_fixed_chains_diverge(net, stim):
    """Not a bug: a ~3e-6 weight perturbation is amplified by chaotic dynamics.
    Asserting it stays visible keeps the motivation for the fixed-point chain
    honest -- if these ever agreed, the exactness guarantee would be untested.
    """
    lif = LIFParams()
    f = naive_dense.simulate(net, stim, lif, arith="float")
    q = naive_dense.simulate(net, stim, lif, arith="fixed")
    assert check.first_divergence(f, q) is not None