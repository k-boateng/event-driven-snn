"""Shared fixtures. Built once per session -- network generation and the naive
reference are the slow parts, and rebuilding them per test would make the suite
too slow to run on every change."""

import pytest

from snnkern.spec import test_config
from snnkern.network import build as build_net
from snnkern.stimulus import build as build_stim


@pytest.fixture(scope="session")
def cfg():
    # Smaller than the canonical test_config() so the suite stays fast; still
    # in the regime where recurrent propagation dominates -- asserted in
    # test_fixture.py rather than assumed.
    return test_config(n=400, n_steps=150)


@pytest.fixture(scope="session")
def net(cfg):
    return build_net(cfg.net)


@pytest.fixture(scope="session")
def stim(cfg):
    return build_stim(cfg.net.n, cfg.stim)