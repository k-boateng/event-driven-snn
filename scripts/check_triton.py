#!/usr/bin/env python
"""Triton kernel verification
"""

import sys

import torch

from snnkern import check
from snnkern.data import load_or_build
from snnkern.impls import event_sparse_cpu, event_sparse_triton as tri
from snnkern.spec import test_config, bench_config, LIFParams

p = torch.cuda.get_device_properties(0)
print(f"{p.name}  sm_{p.major}{p.minor}\n")

fail = 0

from snnkern.network import build as build_net
from snnkern.stimulus import build as build_stim

VARIANTS = {
    "default": LIFParams(),
    "reset_nonzero": LIFParams(v_reset=0.1),
    "long_refrac": LIFParams(refrac_steps=5),
    "no_refrac": LIFParams(refrac_steps=0),
}

cfg = test_config()
net, stim = build_net(cfg.net), build_stim(cfg.net.n, cfg.stim)
for name, lif in VARIANTS.items():
    ref = event_sparse_cpu.simulate(net, stim, lif)
    got = tri.simulate(net, stim, lif)
    d = check.first_divergence(ref, got)
    ok = d is None
    fail += not ok
    print(f"test_config / {name:14s} {'OK' if ok else 'FAIL'}  "
          f"cpu={ref.fingerprint()} gpu={got.fingerprint()}")
    if not ok:
        print(check.compare(ref, got, "cpu", "triton"))

for n in (10_000, 100_000):
    c = bench_config(n, n_steps=200)
    net, stim = load_or_build(c)
    ref = event_sparse_cpu.simulate(net, stim, c.lif)
    got = tri.simulate(net, stim, c.lif)
    d = check.first_divergence(ref, got)
    fail += d is not None
    print(f"bench n={n:<8}         {'OK' if d is None else 'FAIL'}  "
          f"rate {ref.mean_rate_hz():.2f}Hz  fp {ref.fingerprint()}/{got.fingerprint()}")
    if d is not None:
        print(check.compare(ref, got, "cpu", "triton"))

#determinism check
c = bench_config(10_000, n_steps=200)
net, stim = load_or_build(c)
fps = {tri.simulate(net, stim, c.lif).fingerprint() for _ in range(5)}
print(f"determinism (5 runs)     {'OK' if len(fps) == 1 else 'FAIL'}  "
      f"{len(fps)} distinct fingerprint(s)")
fail += len(fps) != 1

print("\nALL PASS" if not fail else f"\n{fail} FAILURES")
sys.exit(1 if fail else 0)