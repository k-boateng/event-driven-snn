#!/usr/bin/env python
"""Verify CUDA graph replay against eager launches, then measure what it bought.
"""

import sys

import torch

from snnkern import check
from snnkern.data import load_or_build
from snnkern.impls import event_sparse_cpu
from snnkern.impls import event_sparse_triton as tri
from snnkern.spec import bench_config
from snnkern.timing import time_gpu

T = 200
SIZES = [10_000, 100_000, 1_000_000]


def main(sizes):
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  NUM_PROGRAMS={tri.NUM_PROGRAMS}\n")

    for n in sizes:
        cfg = bench_config(n, n_steps=T)
        net, stim = load_or_build(cfg)

        sim = tri.TritonSim(net, stim, cfg.lif)

        # eager
        sim.run(use_graph=False)
        eager_rec = sim.record()
        t_eager = time_gpu(lambda: sim.run(use_graph=False), warmup=2, repeats=5)

        # captured
        sim.capture()
        sim.run(use_graph=True)
        graph_rec = sim.record()
        t_graph = time_gpu(lambda: sim.run(use_graph=True), warmup=2, repeats=5)

        d = check.first_divergence(eager_rec, graph_rec)
        ok = d is None

        e_ms, g_ms = t_eager.per_step_ms(T), t_graph.per_step_ms(T)
        print(f"n = {n}")
        print(f"  eager vs graph: {'BIT-EXACT' if ok else 'MISMATCH ' + str(d)}")
        print(f"  eager   {e_ms:.4f} ms/step")
        print(f"  graph   {g_ms:.4f} ms/step   {e_ms/g_ms:.2f}x")
        if n <= 100_000:
            r = event_sparse_cpu.simulate(net, stim, cfg.lif)
            print(f"  (cpu rate {r.mean_rate_hz():.2f}Hz, fp {r.fingerprint()})")
        print()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SIZES)