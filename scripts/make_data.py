#!/usr/bin/env python
"""Generate and freeze the benchmark networks
"""

import sys
import time

from snnkern.data import load_or_build, paths
from snnkern.spec import bench_config, BENCH_SIZES


def main(sizes):
    for n in sizes:
        cfg = bench_config(n)
        net_p, _ = paths(cfg)
        if net_p.exists():
            print(f"n={n:<9} exists   {net_p.name}")
            continue
        t0 = time.perf_counter()
        net, stim = load_or_build(cfg)
        mb = (net.indices.nbytes + net.w_fixed.nbytes + net.w_float.nbytes) / 1e6
        print(f"n={n:<9} built in {time.perf_counter()-t0:6.1f}s  {mb:8.1f} MB  {net_p.name}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or BENCH_SIZES)