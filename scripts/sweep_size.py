#!/usr/bin/env python
"""Size sweep: every implementation, dense vs sparse, one CSV.

    python scripts/sweep_size.py
    python scripts/sweep_size.py 10000 100000     # subset
"""

import gc
import math
import sys

import torch

from dataclasses import replace

from snnkern import check, results
from snnkern.data import load_or_build
from snnkern.impls import dense_gpu, event_sparse_cpu
from snnkern.impls import event_sparse_triton as tri
from snnkern.spec import bench_config
from snnkern.stimulus import build as build_stim
from snnkern.timing import time_cpu, time_gpu

T = 100
SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
REGIMES = {"sparse": 0.002, "live": 0.005}
OUT = "results/size_sweep.csv"

# Above this the CPU reference is too slow to be worth running in a sweep;
# its scaling is already established and linear.
CPU_MAX_N = 300_000


def fmt(x, unit="ms", w=9):
    return f"{x:{w}.4f}{unit}" if x == x else f"{'--':>{w+len(unit)}}"


def main(sizes):
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1e9
    print(f"{p.name}  {total_gb:.0f} GB  NUM_PROGRAMS={tri.NUM_PROGRAMS}  "
          f"T={T} steps\n")

    rows = []
    for regime, kf in REGIMES.items():
        print(f"--- {regime} regime (kick_frac={kf}) " + "-" * 40)
        print(f"{'n':>9} {'rate':>8} {'touch':>6} {'denseGPU':>11} "
              f"{'eventCPU':>11} {'eager':>11} {'graph':>11} "
              f"{'vs dense':>9} {'vs cpu':>8} {'graph gain':>11}")

        for n in sizes:
            base = bench_config(n, n_steps=T)
            net, _ = load_or_build(base)
            cfg = replace(base, stim=replace(base.stim, kick_frac=kf))
            stim = build_stim(n, cfg.stim)

            # eager and captured
            sim = tri.TritonSim(net, stim, cfg.lif, max_spikes=n)
            sim.run(use_graph=False)
            rec = sim.record()
            t_eager = time_gpu(lambda: sim.run(use_graph=False))

            sim.capture()
            sim.run()
            rec_g = sim.record()
            if check.first_divergence(rec, rec_g) is not None:
                raise AssertionError(f"graph replay differs from eager at n={n}")
            t_graph = time_gpu(lambda: sim.run())

            eager_ms = t_eager.per_step_ms(T)
            graph_ms = t_graph.per_step_ms(T)
            f = rec.mean_rate_hz() / 1000.0
            touched = 1.0 - math.exp(-f * net.fan_out)

            rows.append(results.row("event_sparse_triton", cfg, rec, t_graph,
                                    note=f"{regime};cuda_graph"))
            rows.append(results.row("event_sparse_triton", cfg, rec, t_eager,
                                    note=f"{regime};eager"))

            #Dense GPU
            dense_gb = 4 * n * n / 1e9
            if dense_gb < 0.8 * total_gb:
                W = dense_gpu.build_dense(net, "cuda")
                t_den = time_gpu(lambda: dense_gpu.simulate(
                    net, stim, cfg.lif, W=W, record=False))
                den_ms = t_den.per_step_ms(T)
                rows.append(results.row("dense_gpu", cfg, rec, t_den,
                                        gb_per_step=dense_gb, note=regime))
                del W
                gc.collect(); torch.cuda.empty_cache()
            else:
                den_ms = float("nan")
                rows.append(results.row("dense_gpu", cfg, rec, t_graph,
                                        note=f"{regime};SKIPPED W={dense_gb:.0f}GB"))

            #Sparse CPU
            if n <= CPU_MAX_N:
                r_cpu = event_sparse_cpu.simulate(net, stim, cfg.lif)
                t_cpu = time_cpu(lambda: event_sparse_cpu.simulate(
                    net, stim, cfg.lif), warmup=0, repeats=3)
                cpu_ms = t_cpu.per_step_ms(T)
                rows.append(results.row("event_sparse_cpu", cfg, r_cpu, t_cpu,
                                        note=regime))
                if check.first_divergence(r_cpu, rec) is not None:
                    raise AssertionError(f"EXACT CHAIN BROKEN at n={n}, {regime}")
            else:
                cpu_ms = float("nan")

            print(f"{n:9d} {rec.mean_rate_hz():7.1f}Hz {100*touched:5.0f}% "
                  f"{fmt(den_ms)} {fmt(cpu_ms)} {fmt(eager_ms)} {fmt(graph_ms)} "
                  f"{fmt(den_ms/graph_ms, 'x', 8)} {fmt(cpu_ms/graph_ms, 'x', 7)} "
                  f"{fmt(eager_ms/graph_ms, 'x', 10)}")

            del sim
            gc.collect(); torch.cuda.empty_cache()
        print()

    path = results.append(OUT, rows)
    print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SIZES)