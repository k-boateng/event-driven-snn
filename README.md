# event-driven-snn

Event-driven sparse spiking neural network simulator. Dynamic work compaction
and GPU spike propagation in Triton, benchmarked bit-exact against a dense GPU
baseline.

A dense simulator multiplies a mostly-zero spike vector by the full weight
matrix every timestep, doing O(n²) work to propagate O(spikes × fan-out) of
actual signal. This closes that gap: only neurons that fired propagate, only
neurons that received input update, and the active set is built on the device
by atomic test-and-set so the host never learns how many neurons spiked.

The sparsity here is **dynamic and unstructured** — the active set is produced
by the computation one step earlier, changes every step, and is never visible
to the host, so nothing can be pre-permuted or scheduled ahead of time.

---

## Results

NVIDIA H100 80GB SXM, fan-out 100, 100 steps, `OMP_NUM_THREADS=8`.
Full data in `results/*.csv`, one row per measurement with the hardware and
thread count it was taken on.

### Per-step cost vs network size

![size](results/fig_size.png)

**Sparse regime** — 1.9 Hz, ~17% of neurons touched per step:

| n | dense GPU | event CPU | event Triton | vs dense | vs CPU |
|---|---|---|---|---|---|
| 10,000 | 0.182 ms | 0.111 ms | 0.0127 ms | 14× | 8.7× |
| 30,000 | 1.288 ms | 0.261 ms | 0.0130 ms | 99× | 20× |
| 100,000 | 13.83 ms | 0.800 ms | 0.0146 ms | 945× | 55× |
| 300,000 | — | 2.315 ms | 0.0212 ms | — | 109× |
| 1,000,000 | — | — | 0.0401 ms | — | — |

**Live regime** — 22 Hz, ~90% touched, 90% of spikes network-driven rather than
externally injected:

| n | dense GPU | event CPU | event Triton | vs dense | vs CPU |
|---|---|---|---|---|---|
| 10,000 | 0.182 ms | 0.277 ms | 0.0142 ms | 13× | 19× |
| 30,000 | 1.289 ms | 0.726 ms | 0.0169 ms | 76× | 43× |
| 100,000 | 13.82 ms | 2.388 ms | 0.0290 ms | 476× | 82× |
| 300,000 | — | 8.060 ms | 0.0655 ms | — | 123× |
| 1,000,000 | — | — | 0.1961 ms | — | — |

A million-neuron recurrent network at 22 Hz runs at **0.196 ms/timestep**,
about 5,000 simulated steps per second.

The dense rows stop at 100,000 because the weight matrix is 4n² bytes: 40 GB at
n=10⁵, 68 GB at 1.3×10⁵, and past that no dense implementation exists on an
80 GB card at any speed.

The dense baseline is not a strawman. It runs at 2,950–3,050 GB/s, or 88–91% of
this card's HBM3 peak — cuBLAS is at the hardware limit and cannot be
meaningfully improved. The algorithm is what differs, not the implementation.
That is also why single-threaded NumPy beats an H100 by 23× at n=1.3×10⁵.

### No crossover with firing rate

![activity](results/fig_activity.png)

Across a 170× range in firing rate, event-driven stays 133–992× faster than
dense. Even at 100% touched, where skipping inactive neurons saves nothing, it
is still 133× ahead.

The arithmetic says why. Dense costs n² per step regardless of activity;
event-driven costs f·n·K. Those are equal at f = n/K, which exceeds 1 for
n > K and is therefore unreachable — so event-driven wins on operation count at
every attainable rate. The win comes from the scatter, not from the active set:
the touched fraction is `1 − exp(−f·K)`, which saturates by ~40 Hz and stops
contributing, while the scatter advantage never saturates.

Any crossover that did appear would be a constant-factor effect — dense is a
perfectly coalesced matvec near peak bandwidth, scatter does atomics and random
access — rather than a work-ratio one.

### The kernel is latency-bound until it isn't

![graphs](results/fig_graph_gain.png)

CUDA graphs remove launch overhead and nothing else, so the speedup they give
measures how much of the runtime was launch-bound. It falls from 3.3× at
n=10⁴ to 1.02× at n=10⁶: the transition from latency-bound to work-bound,
measured rather than asserted.

At n=10⁴ three kernel launches per step at ~5 µs each dominate a few
microseconds of real work. By n=10⁶ the work dominates and graphs are nearly
free.

---

## Correctness

Every speedup above is at matched output. Spike trains from the Triton kernel
are **bit-identical** to the CPU reference — the same neurons firing at the same
timesteps, not agreement within a tolerance.

That is only possible because synaptic current accumulates in **Q16.16
fixed-point integers** rather than float. Integer addition is associative, so
`atomic_add` gives the same total regardless of the order the hardware schedules
it. With float atomics the same kernel would not reproduce itself run to run,
and the correctness gate could not exist: a 3×10⁻⁶ perturbation in the weights
survives 67 timesteps before it changes a spike, and shifts 10% of total
activity by step 300.

| Comparison | Guarantee |
|---|---|
| any implementation, run twice | bit-exact |
| naive dense (fixed-point) ↔ event-driven CPU | bit-exact |
| event-driven CPU ↔ event-driven Triton | bit-exact |
| float32 dense implementations | statistical: exact for 25 steps, then rates and per-neuron counts |

Checked across four LIF variants (default, non-zero reset, longer refractory,
no refractory) and at every point in both sweeps.

```bash
python -m pytest tests/ -q          # 44 tests
python scripts/check_triton.py      # Triton vs CPU reference
python scripts/check_graph.py       # graph replay vs eager
```

The test suite includes meta-tests asserting the fixture can actually fail:
that most spikes are network-driven rather than externally injected, and that
the lazy-decay path runs with elapsed > 1. Without those, a network where every
spike is a stimulus kick would pass the whole equivalence suite while testing
nothing about propagation.

---

## Model

Leaky integrate-and-fire, fixed random sparse connectivity, fixed weights.

- Membrane potential decays by `exp(-dt/τ)` each step; spikes emitted at `t`
  arrive at `t+1` with uniform one-step delay.
- Hard reset to `v_reset` on firing, then an absolute refractory period during
  which input is discarded rather than integrated.
- Synaptic current accumulates in Q16.16 int32 and converts to float32 once per
  neuron per step. Weights are rounded to fixed-point once, at network
  generation.
- Lazy decay: an untouched neuron's potential is reconstructed as
  `v · decay^elapsed` on its next update. Exact for LIF, because decay only
  moves potential toward rest, so an untouched neuron cannot cross threshold.
- Dale's law, 80/20 excitatory/inhibitory, inhibition 4× stronger per synapse.
- Multi-edges permitted (sampled with replacement, acting as doubled synapses);
  self-edges excluded.

External drive is a **sparse event stream**: a fraction `kick_frac` of neurons
receive a supra-threshold kick each step. This is deliberately not the
independent-Poisson-per-neuron drive of standard benchmark networks — under
dense drive every neuron receives input every step, every neuron is active every
step, and the event-driven formulation degenerates into the dense one. The
sparsity of the external input is load-bearing for every number above, and
`kick_frac` is reported with each.

Between 4 and 5 Hz of drive the network crosses a first-order transition into
self-amplification: at 4 Hz of kicks, 11% of spikes are network-driven; at 5 Hz,
89%, with the measured rate jumping from 4 Hz to 27 Hz. The amplifying regime is
not sparse — at 22 Hz roughly 90% of neurons are touched per step. You get
network-driven dynamics or a sparse active set, not both.

---

## Implementations

`src/snnkern/impls/`

| File | What | Cost per step |
|---|---|---|
| `naive_dense.py` | Python loops over a dense matrix, float or fixed-point | O(spikes·n), interpreted |
| `vectorized_dense.py` | NumPy BLAS matvec | 4n² bytes |
| `event_sparse_cpu.py` | scatter + active set + lazy decay | O(spikes·K + touched), plus an O(n) sweep |
| `dense_gpu.py` | PyTorch fp32 dense matvec | 4n² bytes |
| `event_sparse_triton.py` | Triton, device-side queues, optional CUDA graph | O(spikes·K + touched) |

`naive_dense` runs in both arithmetic modes and is the pivot between the two
correctness chains. `event_sparse_cpu` has a `forced_dense` mode that keeps the
sparse data structures but disables the active set and lazy decay, so a mismatch
can be localised to one or the other.

The NumPy event-driven implementation still spends about 45% of its step on O(n)
work — `bincount` allocating a length-n accumulator and `flatnonzero` scanning
it. The Triton version's device-side touched queue removes that, so its
advantage over the CPU version is partly algorithmic rather than only
parallelism.

### Kernel structure

Each timestep is a breadth-first frontier expansion: spikes are the frontier,
synapses are edges, the next frontier is built by traversal.

```
k_propagate   grid-stride over (spiking neuron × edge-block) pairs
                atomic_add(acc + target, w_fixed)          # int32: associative
                atomic_xchg(flag + target, 1) → winner appends to touched queue
k_stimulus    same, for external events
k_update      grid-stride over the touched queue
                lazy decay, threshold, reset, refractory
                emit into the next frontier; clear acc and flag
```

Three decisions that matter:

**Grid-stride loops, not a sized grid.** The spike count is known only on the
device. Copying it to the host each step costs one synchronisation per timestep,
which is the entire time budget; sizing the grid for the worst case is dense work
again. A fixed program count that reads the count from device memory avoids
both — and is also what makes CUDA graph capture possible at all.

**The spike log is the queue.** Step `t`'s update writes into `log[t]`; step
`t+1`'s propagate reads it. No copying, no double buffering, and the whole
history is already on device for a single transfer at the end.

**(source, edge) flattened into one index space.** One program per spiking
neuron would idle lanes for any source with fewer edges than the block size and
serialise K edges inside one program. Flattened, work divides evenly however
many neurons fired.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu126

export SNNKERN_DATA=/path/to/scratch/data     # networks are up to 1.2 GB
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/make_data.py                   # freeze the benchmark networks
python -m pytest tests/ -q
python scripts/sweep_size.py
python scripts/sweep_activity.py
python scripts/plots.py
```

Networks and stimuli are generated once from fixed seeds and frozen to `.npz`,
named by a tag covering every parameter that affects generation. Both the reason
and the requirement are the same: the GPU and CPU implementations have to be
provably reading the same input, and "same seed" is a weaker guarantee than
"same bytes" — NumPy's bounded-integer generator consumes its stream differently
depending on how generation is chunked.

Timings use CUDA events with warm-up and repeats, report median and IQR, and
exclude setup. Every CSV row carries its device, host, `SLURM_CPUS_PER_TASK` and
thread count; a timing without those cannot be compared against a later one.

---

## Non-goals

No plasticity, learning, or STDP — weights are fixed for the lifetime of a run.
No neuron model other than LIF. No per-synapse or distributed axonal delays. No
conductance-based or filtered synapses. Not a general-purpose library: one
topology made fast, not a configurable framework. No claims of biological
plausibility — this is a systems project on a neuroscience substrate.

## Possible extensions

- CUDA port of the frontier kernel, for a same-algorithm comparison against
  hand-written CUDA
- Persistent cooperative kernel with grid-wide synchronisation, which Triton
  cannot express
- Log-normal degree distribution, where fixed fan-out stops making load
  balancing trivial and warp-level balancing becomes a real contribution
- Comparison against GeNN, the established GPU SNN simulator, at 10⁵–10⁶ neurons