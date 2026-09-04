"""Event-driven sparse GPU kernel (Triton).

A timestep is a breadth-first frontier expansion. Spikes are the frontier,
synapses are edges, the next frontier is built by traversal.

THE FOUR DESIGN DECISIONS

1. GRID-STRIDE LOOPS, NOT A SIZED GRID.

2. THE SPIKE LOG IS THE QUEUE.

3. FIXED-POINT ACCUMULATION MAKES THIS DETERMINISTIC.

4. THE DECAY TABLE, NOT exp(elapsed*log(decay)).


If more than MAX_SPIKES neurons fire in one step, atomic_add still increments
the count but the write is masked off, so counts[t] > MAX_SPIKES afterwards and
check_overflow() raises.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ..fixedpoint import quantize, INV_SCALE
from ..record import SpikeRecord
from ..spec import LIFParams, decay_table, DECAY_TABLE_SIZE

NUM_PROGRAMS = 2048
BLOCK = 128


@triton.jit
def k_propagate(
    src_ptr, src_count_ptr,          # frontier: spikes emitted at t-1
    indices_ptr, w_ptr,              # CSR out-edges (fixed fan-out)
    acc_ptr, flag_ptr,               # per-neuron accumulator + touched flag
    touched_ptr, touched_count_ptr,  # touched queue + its counter
    K: tl.constexpr, MAX_TOUCHED,
    BLOCK: tl.constexpr,
):
    """Scatter from the frontier. One program handles a strided subset of
    (source, edge-block) pairs.

    Flattening (source, edge) into a single index space matters: iterating
    source-major with one program per source would idle every program whose
    source has fewer edges than the block size, and would serialise K edges
    inside one program. Flattened, the work divides evenly regardless of how
    many neurons fired.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_src = tl.load(src_count_ptr)

    blocks_per_src = (K + BLOCK - 1) // BLOCK
    total_blocks = n_src * blocks_per_src

    for b in range(pid, total_blocks, nprog):
        s = b // blocks_per_src
        eb = b % blocks_per_src

        src = tl.load(src_ptr + s)
        off = eb * BLOCK + tl.arange(0, BLOCK)
        mask = off < K

        # Out-edges of one source are contiguous in CSR, so consecutive lanes
        # read consecutive addresses: fully coalesced.
        e = src * K + off
        tgt = tl.load(indices_ptr + e, mask=mask, other=0)
        w = tl.load(w_ptr + e, mask=mask, other=0)

        # Integer atomic: associative, therefore order-independent.
        tl.atomic_add(acc_ptr + tgt, w, mask=mask)

        # Test-and-set. Exactly one lane per neuron sees the old value 0 and
        # wins the right to append it, so the queue holds no duplicates.
        prev = tl.atomic_xchg(flag_ptr + tgt, 1, mask=mask)
        first = mask & (prev == 0)
        slot = tl.atomic_add(touched_count_ptr, tl.sum(first.to(tl.int32)))
        rank = tl.cumsum(first.to(tl.int32)) - 1
        tl.store(touched_ptr + slot + rank, tgt,
                 mask=first & (slot + rank < MAX_TOUCHED))


@triton.jit
def k_stimulus(
    ids_ptr, n_ids,
    acc_ptr, flag_ptr,
    touched_ptr, touched_count_ptr,
    ext_amp, MAX_TOUCHED,
    BLOCK: tl.constexpr,
):
    """External drive into the same accumulator and the same touched queue."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    for b in range(pid, (n_ids + BLOCK - 1) // BLOCK, nprog):
        off = b * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_ids
        tgt = tl.load(ids_ptr + off, mask=mask, other=0)

        tl.atomic_add(acc_ptr + tgt, ext_amp, mask=mask)

        prev = tl.atomic_xchg(flag_ptr + tgt, 1, mask=mask)
        first = mask & (prev == 0)
        slot = tl.atomic_add(touched_count_ptr, tl.sum(first.to(tl.int32)))
        rank = tl.cumsum(first.to(tl.int32)) - 1
        tl.store(touched_ptr + slot + rank, tgt,
                 mask=first & (slot + rank < MAX_TOUCHED))


@triton.jit
def k_update(
    touched_ptr, touched_count_ptr,
    v_ptr, refrac_ptr, last_ptr,
    acc_ptr, flag_ptr,
    log_ptr, counts_ptr,
    decay_ptr,                       # decay^k lookup table, float32
    t, thresh, reset, inv_scale,
    refrac_steps, MAX_SPIKES, TABLE_MAX,
    BLOCK: tl.constexpr,
):
    """Update only the touched set, then emit the next frontier.

    Clears acc and flag for the neurons it touched -- O(touched), not O(n).
    Zeroing the whole accumulator each step would reintroduce exactly the dense
    per-step cost this kernel exists to remove.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_touched = tl.load(touched_count_ptr)

    for b in range(pid, (n_touched + BLOCK - 1) // BLOCK, nprog):
        off = b * BLOCK + tl.arange(0, BLOCK)
        mask = off < n_touched
        j = tl.load(touched_ptr + off, mask=mask, other=0)

        refrac = tl.load(refrac_ptr + j, mask=mask, other=0)
        free = mask & (t >= refrac)

        # Lazy decay: reconstruct v from when it was last correct. Table
        # lookup, not exp/log
        last = tl.load(last_ptr + j, mask=mask, other=0)
        elapsed = tl.minimum(t - last, TABLE_MAX)
        v = tl.load(v_ptr + j, mask=mask, other=0.0)
        v = v * tl.load(decay_ptr + elapsed, mask=mask, other=0.0)

        cur = tl.load(acc_ptr + j, mask=mask, other=0).to(tl.float32) * inv_scale
        v = v + cur

        fired = free & (v >= thresh)
        v_new = tl.where(fired | (~free), reset, v)

        tl.store(v_ptr + j, v_new, mask=mask)
        # Refractory neurons keep their old last_update: v is pinned at reset
        # through the whole window, and the spike branch below already records
        # validity as of the last clamped step.
        tl.store(last_ptr + j, t, mask=free)
        tl.store(refrac_ptr + j, t + refrac_steps, mask=fired)
        tl.store(last_ptr + j, t + refrac_steps - 1, mask=fired)

        # Clear for the next step.
        tl.store(acc_ptr + j, 0, mask=mask)
        tl.store(flag_ptr + j, 0, mask=mask)

        # Emit into the next frontier.
        slot = tl.atomic_add(counts_ptr + t, tl.sum(fired.to(tl.int32)))
        rank = tl.cumsum(fired.to(tl.int32)) - 1
        tl.store(log_ptr + t * MAX_SPIKES + slot + rank, j,
                 mask=fired & (slot + rank < MAX_SPIKES))


class TritonSim:
    """Holds the device state so that setup stays outside any timed region."""

    def __init__(self, net, stim, lif: LIFParams, max_spikes: int | None = None,
                 device: str = "cuda"):
        self.net, self.stim, self.lif = net, stim, lif
        n = net.n
        dev = torch.device(device)
        self.dev = dev

        # Headroom over the expected spikes/step. Overflow is detected, not
        # tolerated: see check_overflow(). Pass max_spikes=n to make overflow
        # impossible at any activity level.
        expected = max(64, int(0.05 * n))
        self.MAX_SPIKES = max_spikes or min(n, max(4096, 4 * expected))
        self.MAX_TOUCHED = n  # a neuron can be touched at most once per step

        self.indices = torch.from_numpy(net.indices.astype(np.int32)).to(dev)
        self.w = torch.from_numpy(net.w_fixed.astype(np.int32)).to(dev)
        self.stim_ids = torch.from_numpy(stim.ids.astype(np.int32)).to(dev)

        amps = np.unique(stim.amp)
        assert amps.size == 1, "non-uniform kick amplitude not supported"
        self.ext_amp = int(quantize(np.array([amps[0]]))[0])

        # Same table the CPU implementation uses. Built on the host so there is
        # exactly one definition of decay^k in the project.
        self.dtable = torch.from_numpy(decay_table(lif.decay)).to(dev)

        self.v = torch.empty(n, dtype=torch.float32, device=dev)
        self.refrac = torch.empty(n, dtype=torch.int32, device=dev)
        self.last = torch.empty(n, dtype=torch.int32, device=dev)
        self.acc = torch.empty(n, dtype=torch.int32, device=dev)
        self.flag = torch.empty(n, dtype=torch.int32, device=dev)
        self.touched = torch.empty(n, dtype=torch.int32, device=dev)
        self.touched_count = torch.empty(1, dtype=torch.int32, device=dev)
        self.log = torch.empty((stim.n_steps, self.MAX_SPIKES),
                               dtype=torch.int32, device=dev)
        self.counts = torch.empty(stim.n_steps, dtype=torch.int32, device=dev)

        # Placeholder for step 0, where there is no previous frontier. Must be
        # a persistent tensor: a fresh torch.zeros() inside a captured region
        # would allocate during capture, which capture forbids.
        self._zero_count = torch.zeros(1, dtype=torch.int32, device=dev)
        self._graph = None

    def reset_state(self):
        self.v.fill_(float(self.lif.v_reset))
        self.refrac.zero_()
        # -1, not 0: at t=0 the dense path applies one decay to the initial
        # potential, so elapsed must be 1 on the first touch. Invisible when
        # v_reset == 0 (zero times anything is zero) and wrong otherwise.
        self.last.fill_(-1)
        self.acc.zero_()
        self.flag.zero_()
        self.counts.zero_()

    def _enqueue(self):
        """The whole simulation is enqueued"""
        lif, stim = self.lif, self.stim
        grid = (NUM_PROGRAMS,)
        K = self.net.fan_out
        k = stim.k

        prev_count = self._zero_count

        for t in range(stim.n_steps):
            self.touched_count.zero_()

            if t > 0:
                k_propagate[grid](
                    self.log[t - 1], prev_count,
                    self.indices, self.w,
                    self.acc, self.flag,
                    self.touched, self.touched_count,
                    K=K, MAX_TOUCHED=self.MAX_TOUCHED, BLOCK=BLOCK,
                )

            k_stimulus[grid](
                self.stim_ids[t * k:(t + 1) * k], k,
                self.acc, self.flag,
                self.touched, self.touched_count,
                self.ext_amp, self.MAX_TOUCHED, BLOCK=BLOCK,
            )

            k_update[grid](
                self.touched, self.touched_count,
                self.v, self.refrac, self.last,
                self.acc, self.flag,
                self.log, self.counts,
                self.dtable,
                t, float(lif.v_thresh), float(lif.v_reset),
                float(INV_SCALE), lif.refrac_steps, self.MAX_SPIKES,
                DECAY_TABLE_SIZE - 1,
                BLOCK=BLOCK,
            )
            prev_count = self.counts[t:t + 1]

    def capture(self):
        """Record the run as a CUDA graph.

        The loop is UNROLLED into the graph at capture time: all 3*n_steps
        launches become one DAG with arguments and dependencies baked in.
        Replay submits it once instead of launching 3*n_steps times, taking
        per-launch overhead from ~5-6us to ~1-2us.

        The graph is tied to these tensors, this n_steps and this stimulus.
        Capture per configuration; do not share graphs across configs.
        """
        self.reset_state()

        # Warm up on a side stream first. The first launch of each kernel
        # triggers Triton JIT compilation, which allocates and synchronises --
        # both illegal inside a capture region.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._enqueue()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # reset_state() is deliberately OUTSIDE the capture. Capturing it would
        # make every replay restart from initial conditions which happens to
        # be what we want, but only by accident.
        self.reset_state()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._enqueue()
        return self

    def run(self, use_graph: bool = True):
        """Run the simulation. Falls back to eager launches when no graph has
        been captured, so correctness testing does not depend on capture."""
        self.reset_state()
        if use_graph and self._graph is not None:
            self._graph.replay()
        else:
            self._enqueue()

    def check_overflow(self):
        c = self.counts.max().item()
        if c > self.MAX_SPIKES:
            raise OverflowError(
                f"spike log overflow: {c} spikes in one step > "
                f"MAX_SPIKES={self.MAX_SPIKES}. Results are truncated and wrong."
            )

    def record(self) -> SpikeRecord:
        self.check_overflow()
        log = self.log.cpu().numpy()
        counts = self.counts.cpu().numpy()
        per_step = [np.sort(log[t, :counts[t]]).astype(np.int32)
                    for t in range(self.stim.n_steps)]
        return SpikeRecord(self.net.n, self.stim.n_steps, per_step)


def simulate(net, stim, lif: LIFParams, device: str = "cuda",
             use_graph: bool = False, max_spikes: int | None = None
             ) -> SpikeRecord:
    """use_graph defaults to False: correctness checks should exercise the
    plain launch path, so a capture bug cannot hide behind a passing test."""
    sim = TritonSim(net, stim, lif, max_spikes=max_spikes, device=device)
    if use_graph:
        sim.capture()
    sim.run(use_graph=use_graph)
    return sim.record()