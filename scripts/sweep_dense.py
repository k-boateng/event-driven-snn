#!/usr/bin/env python
"""Dense GPU baseline vs event-driven CPU, written to CSV.
"""

import gc
import sys

import torch

from snnkern import results
from snnkern.data import load_or_build
from snnkern.impls import dense_gpu, event_sparse_cpu
from snnkern.spec import bench_config
from snnkern.timing import time_cpu, time_gpu

T = 200
SIZES = [10_000, 30_000, 100_000, 130_000]
OUT = "results/dense_baseline.csv"


def main(sizes):
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1e9
    print(f"{p.name}  sm_{p.major}{p.minor}  {total_gb:.0f} GB")

    rows = []
    for n in sizes:
        cfg = bench_config(n, n_steps=T)
        net, stim = load_or_build(cfg, verbose=True)
        gb = 4 * n * n / 1e9

        # Event-driven CPU (runs at every size)
        r_cpu = event_sparse_cpu.simulate(net, stim, cfg.lif)
        t_cpu = time_cpu(lambda: event_sparse_cpu.simulate(net, stim, cfg.lif),
                         warmup=0, repeats=3)
        rows.append(results.row("event_sparse_cpu", cfg, r_cpu, t_cpu))
        print(f"n={n:<8} event CPU  {t_cpu.per_step_ms(T):8.3f}ms  "
              f"rate {r_cpu.mean_rate_hz():.2f}Hz")

        #Dense GPU (only while the matrix fits)
        if gb > 0.8 * total_gb:
            print(f"n={n:<8} dense GPU  -- {gb:.1f}GB exceeds device memory")
            rows.append(results.row("dense_gpu", cfg, r_cpu, t_cpu,
                                    note=f"SKIPPED: W={gb:.1f}GB > device"))
            continue

        # Built outside the timed region: rebuilding a 68 GB matrix per repeat
        # both contaminates the measurement and fragments the allocator.
        W = dense_gpu.build_dense(net, "cuda")
        r_gpu = dense_gpu.simulate(net, stim, cfg.lif, W=W)
        t_gpu = time_gpu(lambda: dense_gpu.simulate(net, stim, cfg.lif,
                                                    W=W, record=False))
        del W
        gc.collect()
        torch.cuda.empty_cache()

        rows.append(results.row("dense_gpu", cfg, r_gpu, t_gpu, gb_per_step=gb))
        ms = t_gpu.per_step_ms(T)
        print(f"n={n:<8} dense GPU  {ms:8.3f}ms  {gb/(ms/1000):6.0f} GB/s  "
              f"CPU/GPU {t_cpu.median_s/t_gpu.median_s:.2f}x")

    path = results.append(OUT, rows)
    print(f"\nwrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SIZES)