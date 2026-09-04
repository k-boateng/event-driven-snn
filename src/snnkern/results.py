"""Benchmark results as CSV.
"""

from __future__ import annotations

import csv
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp", "impl", "n", "fan_out", "n_steps", "kick_frac",
    "rate_hz", "touched_frac", "ms_per_step", "iqr_ms", "repeats",
    "gb_per_step", "gb_per_s", "device", "host", "slurm_cpus",
    "omp_threads", "numpy", "torch", "note",
]


def env() -> dict:
    import numpy as np

    device = "cpu"
    torch_ver = ""
    try:
        import torch
        torch_ver = torch.__version__
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "device": device,
        "host": socket.gethostname(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "numpy": np.__version__,
        "torch": torch_ver,
    }


def append(path, rows: list[dict]) -> Path:
    """Append rows, writing a header if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    e = env()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"timestamp": ts, **e, **r})
    return path


def row(impl: str, cfg, record, timing, *, gb_per_step: float = 0.0,
        note: str = "") -> dict:
    """Build one row from a Config, a SpikeRecord and a Timing."""
    import math

    n_steps = cfg.stim.n_steps
    ms = timing.per_step_ms(n_steps)
    f = record.mean_rate_hz() / 1000.0
    return {
        "impl": impl,
        "n": cfg.net.n,
        "fan_out": cfg.net.fan_out,
        "n_steps": n_steps,
        "kick_frac": cfg.stim.kick_frac,
        "rate_hz": round(record.mean_rate_hz(), 4),
        # 1 - exp(-f*K): expected fraction of neurons receiving input in a step.
        "touched_frac": round(1.0 - math.exp(-f * cfg.net.fan_out), 4),
        "ms_per_step": round(ms, 5),
        "iqr_ms": round(1000.0 * timing.iqr_s / n_steps, 5),
        "repeats": timing.n_repeats,
        "gb_per_step": round(gb_per_step, 4),
        "gb_per_s": round(gb_per_step / (ms / 1000.0), 1) if ms > 0 and gb_per_step else "",
        "note": note,
    }