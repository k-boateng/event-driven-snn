#!/usr/bin/env python
""" Measures end-to-end per-step time, per-kernel breakdown and the empty loop floor
"""

import sys
import time

import torch
import triton
import triton.language as tl

from snnkern.data import load_or_build
from snnkern.impls import event_sparse_cpu
from snnkern.impls import event_sparse_triton as tri
from snnkern.spec import bench_config
from snnkern.timing import time_cpu, time_gpu

T = 200
SIZES = [10_000, 100_000]


@triton.jit
def k_noop(x_ptr):
    """Launched with the same grid as the real kernels. Does nothing, so the
    loop measures launch overhead and Python-side cost only."""
    pid = tl.program_id(0)
    if pid < 0:                      # never true; keeps the kernel from being
        tl.store(x_ptr, 0)           # optimised away entirely
    return


def empty_loop(sim, n_steps, launches_per_step=3):
    grid = (tri.NUM_PROGRAMS,)
    for _ in range(n_steps):
        sim.touched_count.zero_()
        for _ in range(launches_per_step):
            k_noop[grid](sim.touched_count)


def ms(ev_pair):
    start, end = ev_pair
    return start.elapsed_time(end)


def time_one_kernel(fn, repeats=20):
    """Isolated timing: synchronise, launch, synchronise. Inflated by the
    syncs, so only comparable against other numbers measured the same way."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    return samples[len(samples) // 2]


def main(sizes):
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")
    print(f"NUM_PROGRAMS={tri.NUM_PROGRAMS}  BLOCK={tri.BLOCK}\n")

    for n in sizes:
        cfg = bench_config(n, n_steps=T)
        net, stim = load_or_build(cfg)
        K, k = net.fan_out, stim.k

        sim = tri.TritonSim(net, stim, cfg.lif)

        #end to end
        t_gpu = time_gpu(lambda: sim.run(), warmup=2, repeats=5)
        gpu_ms = t_gpu.per_step_ms(T)

        rec = sim.record()
        r_cpu = event_sparse_cpu.simulate(net, stim, cfg.lif)
        t_cpu = time_cpu(lambda: event_sparse_cpu.simulate(net, stim, cfg.lif),
                         warmup=0, repeats=3)
        cpu_ms = t_cpu.per_step_ms(T)

        #floor
        t_floor = time_gpu(lambda: empty_loop(sim, T), warmup=2, repeats=5)
        floor_ms = t_floor.per_step_ms(T)

        #per-kernel
        # Rebuild state, then run one representative step (t=100, mid-run, so
        # the frontier is at steady state rather than empty).
        sim.reset_state()
        sim.run()
        t_mid = 100
        grid = (tri.NUM_PROGRAMS,)
        prev_count = sim.counts[t_mid - 1:t_mid]

        prop = time_one_kernel(lambda: tri.k_propagate[grid](
            sim.log[t_mid - 1], prev_count, sim.indices, sim.w,
            sim.acc, sim.flag, sim.touched, sim.touched_count,
            K=K, MAX_TOUCHED=sim.MAX_TOUCHED, BLOCK=tri.BLOCK))

        stimk = time_one_kernel(lambda: tri.k_stimulus[grid](
            sim.stim_ids[t_mid * k:(t_mid + 1) * k], k,
            sim.acc, sim.flag, sim.touched, sim.touched_count,
            sim.ext_amp, sim.MAX_TOUCHED, BLOCK=tri.BLOCK))

        upd = time_one_kernel(lambda: tri.k_update[grid](
            sim.touched, sim.touched_count, sim.v, sim.refrac, sim.last,
            sim.acc, sim.flag, sim.log, sim.counts,
            t_mid, float(cfg.lif.decay), float(cfg.lif.v_thresh),
            float(cfg.lif.v_reset), float(tri.INV_SCALE),
            cfg.lif.refrac_steps, sim.MAX_SPIKES, BLOCK=tri.BLOCK))

        spikes = rec.total_spikes / T
        edges = spikes * K
        touched = sim.touched_count.item()

        print(f"n = {n}")
        print(f"  rate {rec.mean_rate_hz():.2f}Hz   {spikes:.0f} spikes/step   "
              f"{edges:.0f} edges/step   ~{touched} touched")
        print(f"  event CPU (rung 2)      {cpu_ms:9.4f} ms/step")
        print(f"  event Triton (rung 3)   {gpu_ms:9.4f} ms/step   "
              f"speedup {cpu_ms/gpu_ms:6.1f}x")
        print(f"  launch-overhead floor   {floor_ms:9.4f} ms/step   "
              f"({100*floor_ms/gpu_ms:.0f}% of runtime)")
        print(f"  per-kernel (sync-inflated, for balance only):")
        print(f"      propagate {prop:7.4f} ms   stimulus {stimk:7.4f} ms   "
              f"update {upd:7.4f} ms")
        headroom = gpu_ms - floor_ms
        print(f"  addressable by kernel work: {headroom:.4f} ms/step "
              f"({100*headroom/gpu_ms:.0f}%)\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SIZES)