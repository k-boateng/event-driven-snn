#!/usr/bin/env python
"""Plots from results/*.csv"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path("results")
LABEL = {
    "dense_gpu": "dense GPU (PyTorch)",
    "event_sparse_cpu": "event-driven CPU (NumPy)",
    "event_sparse_triton": "event-driven GPU (Triton)",
}
COLOR = {"dense_gpu": "#c44e52", "event_sparse_cpu": "#4c72b0",
         "event_sparse_triton": "#55a868"}


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(RES / name)
    df["note"] = df["note"].fillna("")

    df = df[~df["note"].str.contains("SKIPPED")]
    
    df = (df.sort_values("timestamp")
            .groupby(["impl", "n", "kick_frac", "note"], as_index=False)
            .last())
    df["regime"] = df["note"].str.split(";").str[0]
    df["variant"] = df["note"].apply(
        lambda s: "cuda_graph" if "cuda_graph" in s else
                  ("eager" if "eager" in s else ""))
    return df


def fig_size(df):
    """Per-step time vs network size, one panel per activity regime.
    """
    regimes = [r for r in ("sparse", "live") if r in set(df["regime"])]
    fig, axes = plt.subplots(1, len(regimes), figsize=(11, 4.4), sharey=True)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = df[df["regime"] == regime]
        rate = sub["rate_hz"].median()
        touch = 100 * sub["touched_frac"].median()

        for impl in ("dense_gpu", "event_sparse_cpu", "event_sparse_triton"):
            s = sub[sub["impl"] == impl]
            if impl == "event_sparse_triton":
                s = s[s["variant"] == "cuda_graph"]
            s = s.sort_values("n")
            if s.empty:
                continue
            ax.plot(s["n"], s["ms_per_step"], "o-", color=COLOR[impl],
                    label=LABEL[impl], lw=1.8, ms=5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("neurons")
        ax.set_title(f"{regime}  (~{rate:.0f} Hz, ~{touch:.0f}% touched)",
                     fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.axvspan(1.3e5, 1.2e6, color="0.92", zorder=0)
        lo, hi = ax.get_ylim()
        ax.text(4e5, lo * 1.4, "dense exceeds\n80 GB here",
                fontsize=7.5, color="0.45", ha="center")

    axes[0].set_ylabel("ms per timestep")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-step cost vs network size", y=0.99, fontsize=12)
    fig.tight_layout()
    out = RES / "fig_size.png"
    fig.savefig(out, dpi=160)
    return out


def fig_activity(df):
    """Speedup vs firing rate, with the touched fraction on a second axis.
    """
    piv = (df.pivot_table(index="rate_hz", columns="impl",
                          values="ms_per_step", aggfunc="last")
             .sort_index())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    tri = "event_sparse_triton"
    for other, style in (("dense_gpu", "o-"), ("event_sparse_cpu", "s-")):
        if other not in piv or tri not in piv:
            continue
        r = (piv[other] / piv[tri]).dropna()
        ax.plot(r.index, r.values, style, color=COLOR[other], lw=1.8, ms=5,
                label=f"vs {LABEL[other]}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean firing rate (Hz)")
    ax.set_ylabel("Triton speedup (x)")
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.text(piv.index.max() * 0.45, 1.12, "parity", fontsize=8, color="0.3")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")

    ax2 = ax.twinx()
    rates = piv.index.values
    ax2.plot(rates, [100 * (1 - math.exp(-r / 1000 * 100)) for r in rates],
             ":", color="0.45", lw=1.5)
    ax2.set_ylabel("touched neurons per step (%), predicted", color="0.45",
                   fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors="0.45")

    ax.set_title("No crossover: event-driven wins at every attainable rate\n"
                 "n = 100,000", fontsize=11)
    fig.tight_layout()
    out = RES / "fig_activity.png"
    fig.savefig(out, dpi=160)
    return out


def fig_graph_gain(df):
    """CUDA graph speedup vs size.
    """
    t = df[df["impl"] == "event_sparse_triton"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for regime, mark in (("sparse", "o-"), ("live", "s-")):
        s = t[t["regime"] == regime]
        eager = s[s["variant"] == "eager"].set_index("n")["ms_per_step"]
        graph = s[s["variant"] == "cuda_graph"].set_index("n")["ms_per_step"]
        gain = (eager / graph).dropna().sort_index()
        if gain.empty:
            continue
        ax.plot(gain.index, gain.values, mark, lw=1.8, ms=5, label=regime)

    ax.set_xscale("log")
    ax.set_xlabel("neurons")
    ax.set_ylabel("eager / CUDA graph  (x)")
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.text(1.1e4, 1.03, "no benefit: fully work-bound", fontsize=8, color="0.3")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="regime", title_fontsize=8)
    ax.set_title("CUDA graphs help in proportion to launch overhead", fontsize=11)
    fig.tight_layout()
    out = RES / "fig_graph_gain.png"
    fig.savefig(out, dpi=160)
    return out


def main():
    made = []
    if (RES / "size_sweep.csv").exists():
        df = load("size_sweep.csv")
        made += [fig_size(df), fig_graph_gain(df)]
    if (RES / "activity_sweep.csv").exists():
        made.append(fig_activity(load("activity_sweep.csv")))
    for p in made:
        print("wrote", p)
    if not made:
        print("no CSVs in results/ -- run the sweeps first")


if __name__ == "__main__":
    main()