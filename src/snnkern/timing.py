"""Timing harness.

Three errors this exists to prevent:

  1. Timing without synchronising. CUDA launches are asynchronous, so
     time.perf_counter() around a kernel measures queue submission, not
     execution. CUDA events are recorded in the stream and measure the GPU.
  2. No warm-up. The first call pays kernel compilation, autotuning, cuBLAS
     handle creation and allocator growth. Timing it measures none of the
     things you care about.
  3. Reporting the mean, or the minimum. The distribution is right-skewed
     (interference, clock throttling), so the mean is dragged by outliers and
     the minimum is a best case nobody reproduces. Median and IQR describe what
     actually happens.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class Timing:
    median_s: float
    iqr_s: float
    n_repeats: int
    samples: list[float]

    def per_step_ms(self, n_steps: int) -> float:
        return 1000.0 * self.median_s / n_steps

    def __str__(self) -> str:
        return (f"{self.median_s:.4f}s median, IQR {self.iqr_s:.4f}s, "
                f"{self.n_repeats} repeats")


def _summarise(samples: list[float]) -> Timing:
    a = np.asarray(samples, dtype=np.float64)
    q1, q3 = np.percentile(a, [25, 75])
    return Timing(float(np.median(a)), float(q3 - q1), len(samples), samples)


def time_cpu(fn, warmup: int = 1, repeats: int = 5) -> Timing:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _summarise(samples)


def time_gpu(fn, warmup: int = 2, repeats: int = 5) -> Timing:
    """CUDA-event timed. fn must enqueue its work on the current stream and
    must NOT synchronise internally -- a sync inside fn serialises the timing
    region and inflates every sample."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return _summarise(samples)