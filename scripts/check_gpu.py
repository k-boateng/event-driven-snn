#!/usr/bin/env python
"""Verify GPU against the CPU float chain, then sweep the dense baseline.
"""

import gc

import torch

from snnkern import check
from snnkern.spec import test_config, bench_config
from snnkern.network import build as build_net
from snnkern.stimulus import build as build_stim
from snnkern.data import load_or_build
from snnkern.impls import naive_dense, dense_gpu, event_sparse_cpu
from snnkern.timing import time_gpu, time_cpu

p = torch.cuda.get_device_properties(0)
TOTAL_GB = p.total_memory / 1e9
print(f"{p.name}  sm_{p.major}{p.minor}  {TOTAL_GB:.0f} GB\n")

#correctness: float baseline chain
cfg = test_config()
net, stim = build_net(cfg.net), build_stim(cfg.net.n, cfg.stim)
cpu = naive_dense.simulate(net, stim, cfg.lif, arith="float")
gpu = dense_gpu.simulate(net, stim, cfg.lif)
check.assert_statistical(cpu, gpu, "cpu", "gpu")
print("naive_dense(float) vs dense_gpu: passes assert_statistical")
d = check.first_divergence(cpu, gpu)
print(f"  {'identical' if d is None else d}")
print("  (identical is luck at n=500, not a property: cuBLAS reassociates)\n")

#dense baseline sweep
T = 200
print(f"{'n':>8} {'W':>8} {'rate':>8} {'touched':>8} "
      f"{'denseGPU':>10} {'GB/s':>7} {'%peak':>6} {'eventCPU':>10} {'CPU/GPU':>8}")

for n in (10_000, 30_000, 100_000, 130_000):
    gb = 4 * n * n / 1e9
    if gb > 0.8 * TOTAL_GB:
        print(f"{n:8d} {gb:7.1f}GB  -- exceeds device memory, dense does not exist")
        continue

    c = bench_config(n, n_steps=T)
    net, stim = load_or_build(c)

    W = dense_gpu.build_dense(net, "cuda")          # OUTSIDE the timed region
    r = dense_gpu.simulate(net, stim, c.lif, W=W)
    tg = time_gpu(lambda: dense_gpu.simulate(net, stim, c.lif, W=W, record=False))
    del W
    gc.collect(); torch.cuda.empty_cache()

    tc = time_cpu(lambda: event_sparse_cpu.simulate(net, stim, c.lif),
                  warmup=0, repeats=3)

    ms = tg.per_step_ms(T)
    bw = (4 * n * n / 1e9) / (ms / 1000)
    f = r.mean_rate_hz() / 1000.0
    touched = 1.0 - pow(2.718281828, -f * net.fan_out)
    print(f"{n:8d} {gb:7.1f}GB {r.mean_rate_hz():7.2f}Hz {100*touched:7.1f}% "
          f"{ms:9.3f}ms {bw:7.0f} {100*bw/3350:5.0f}% "
          f"{tc.per_step_ms(T):9.3f}ms {tc.median_s/tg.median_s:7.2f}x")

print("\nCPU/GPU < 1 means the NumPy event-driven implementation is FASTER "
      "than the dense H100 baseline.")