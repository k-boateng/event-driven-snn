#!/usr/bin/env python
"""See how does each implementation scales with firing rate

    python scripts/sweep_activity.py
    python scripts/sweep_activity.py 30000        # smaller n, faster
"""

import gc
import math
import sys

import torch

from snnkern import check, results
from snnkern.data import load_or_build
from snnkern.impls import dense_gpu, event_sparse_cpu
from snnkern.impls import event_sparse_triton as tri
from snnkern.network import build as build_net
from snnkern.spec import bench_config
from snnkern.stimulus import build as build_stim
from snnkern.timing import time_cpu, time_gpu

from dataclasses import replace

T = 100
KICK_FRACS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
OUT = "results/activity_sweep.csv"


CPU_MAX_KICK = 0.05


def main(n: int):
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1e9
    dense_gb = 4 * n * n / 1e9
    dense_fits = dense_gb < 0.8 * total_gb
    print(f"{p.name}  n={n}  dense W = {dense_gb:.1f} GB "
          f"({'fits' if dense_fits else 'DOES NOT FIT'})\n")

    # One network, many stimuli. load_or_build keys on the full tag, so calling
    # it per kick_frac would write a fresh 120 MB copy of an identical network.
    base = bench_config(n, n_steps=T)
    net, _ = load_or_build(base)

    W = dense_gpu.build_dense(net, "cuda") if dense_fits else None

    print(f"{'kick_f':>7} {'rate':>8} {'touched':>8} {'spk/step':>9} "
          f"{'denseGPU':>10} {'eventCPU':>10} {'triton':>10} "
          f"{'vs dense':>9} {'vs cpu':>8}")

    rows = []
    for kf in KICK_FRACS:
        cfg = replace(base, stim=replace(base.stim, kick_frac=kf))
        stim = build_stim(n, cfg.stim)

        #Triton
        sim = tri.TritonSim(net, stim, cfg.lif, max_spikes=n)
        sim.capture()
        sim.run()
        rec = sim.record()                       # raises on overflow
        t_tri = time_gpu(lambda: sim.run(), warmup=2, repeats=5)
        tri_ms = t_tri.per_step_ms(T)

        f = rec.mean_rate_hz() / 1000.0
        touched_pred = 1.0 - math.exp(-f * net.fan_out)
        spk = rec.total_spikes / T

        rows.append(results.row("event_sparse_triton", cfg, rec, t_tri,
                                note="cuda_graph"))

        #GPU
        if dense_fits:
            t_den = time_gpu(lambda: dense_gpu.simulate(net, stim, cfg.lif,
                                                        W=W, record=False))
            den_ms = t_den.per_step_ms(T)
            rows.append(results.row("dense_gpu", cfg, rec, t_den,
                                    gb_per_step=dense_gb))
        else:
            den_ms = float("nan")

        #CPU
        if kf <= CPU_MAX_KICK:
            r_cpu = event_sparse_cpu.simulate(net, stim, cfg.lif)
            t_cpu = time_cpu(lambda: event_sparse_cpu.simulate(net, stim, cfg.lif),
                             warmup=0, repeats=3)
            cpu_ms = t_cpu.per_step_ms(T)
            rows.append(results.row("event_sparse_cpu", cfg, r_cpu, t_cpu))
            # Cheap cross-check: the exact chain must still hold at every
            # activity level, not just the one it was verified at.
            if check.first_divergence(r_cpu, rec) is not None:
                raise AssertionError(f"EXACT CHAIN BROKEN at kick_frac={kf}")
        else:
            cpu_ms = float("nan")

        print(f"{kf:7.3f} {rec.mean_rate_hz():7.1f}Hz {100*touched_pred:7.1f}% "
              f"{spk:9.0f} {den_ms:9.4f}ms {cpu_ms:9.4f}ms {tri_ms:9.4f}ms "
              f"{den_ms/tri_ms:8.1f}x {cpu_ms/tri_ms:7.1f}x")

        del sim
        gc.collect()
        torch.cuda.empty_cache()

    path = results.append(OUT, rows)
    print(f"\nwrote {len(rows)} rows -> {path}")
    print("touched% is the prediction 1-exp(-f*K), not a measurement")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)