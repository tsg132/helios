#!/usr/bin/env python3
"""
tools/plot.py — Helios benchmark visualization suite.

Produces publication-quality plots suitable for systems/ML conference papers
(NeurIPS, ICML, OSDI style).

Usage:
    python3 tools/plot.py [--results-dir bench/results]
    python3 tools/plot.py bench/results          # legacy positional form
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams

# ─── Publication-Quality rcParams ─────────────────────────────────────────────
# Target: single-column ~3.5in, double-column ~7in.
# Fonts match IEEE/ACM/NeurIPS body text; STIX mathtext ≈ Computer Modern.

rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size":            10,
    "text.usetex":          False,
    "mathtext.fontset":     "stix",

    "axes.facecolor":       "white",
    "axes.edgecolor":       "#333333",
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "axes.axisbelow":       True,
    "axes.titlesize":       10,
    "axes.titleweight":     "bold",
    "axes.labelsize":       9,
    "axes.labelweight":     "normal",

    "grid.alpha":           0.3,
    "grid.linestyle":       "-",
    "grid.linewidth":       0.5,
    "grid.color":           "#999999",

    "lines.linewidth":      1.2,
    "lines.markersize":     5,
    "lines.markeredgewidth": 0.6,

    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "xtick.direction":      "in",
    "ytick.direction":      "in",

    "legend.fontsize":      8,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#cccccc",
    "legend.fancybox":      False,
    "legend.borderpad":     0.4,
    "legend.handlelength":  1.8,
    "legend.handleheight":  0.8,

    "figure.dpi":           150,
    "figure.facecolor":     "white",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
})

# ─── Okabe-Ito Colorblind-Safe Palette ────────────────────────────────────────
_OI = {
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
    "black":      "#000000",
}

SOLVER_COLORS = {
    "Jacobi":         _OI["blue"],
    "GaussSeidel":    _OI["vermillion"],
    "Async_Static":   _OI["green"],
    "Async_Shuffled": _OI["orange"],
    "Async_TopKGS":   _OI["pink"],
    "Async_CATopKGS": _OI["black"],
    "Async_ResBuck":  _OI["sky_blue"],
    "Plan_Static":    _OI["orange"],
    "Plan_Colored":   _OI["sky_blue"],
    "Plan_Priority":  _OI["pink"],
    "AT_Static":      _OI["orange"],
}

SOLVER_MARKERS = {
    "Jacobi":         "o",
    "GaussSeidel":    "s",
    "Async_Static":   "^",
    "Async_Shuffled": "D",
    "Async_TopKGS":   "v",
    "Async_CATopKGS": "P",
    "Async_ResBuck":  "h",
    "Plan_Static":    "D",
    "Plan_Colored":   "p",
    "Plan_Priority":  "*",
    "AT_Static":      "X",
}

SOLVER_HATCHES = {
    "Jacobi":         "",
    "GaussSeidel":    "///",
    "Async_Static":   "...",
    "Async_Shuffled": "xxx",
    "Async_TopKGS":   "+++",
    "Async_CATopKGS": "OOO",
    "Async_ResBuck":  "---",
    "Plan_Static":    "\\\\\\",
    "Plan_Colored":   "|||",
    "Plan_Priority":  "**",
    "AT_Static":      "\\\\\\",
}

SOLVER_LABELS = {
    "Jacobi":         "Jacobi",
    "GaussSeidel":    "Gauss-Seidel",
    "Async_Static":   "Async (Static)",
    "Async_Shuffled": "Async (Shuffled)",
    "Async_TopKGS":   r"Async (TopK-GS)",
    "Async_CATopKGS": "Async (CA-TopK)",
    "Async_ResBuck":  "Async (ResBucket)",
    "Plan_Static":    "Plan (Static)",
    "Plan_Colored":   "Plan (Colored)",
    "Plan_Priority":  "Plan (Priority)",
    "AT_Static":      "AutoTune (Static)",
}

SOLVER_ORDER = [
    "Jacobi", "GaussSeidel",
    "Plan_Static", "Plan_Colored", "Plan_Priority",
    "Async_Static", "Async_Shuffled", "Async_TopKGS",
    "Async_CATopKGS", "Async_ResBuck",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _match(solver, d, default=None):
    if solver in d:
        return d[solver]
    for key in sorted(d.keys(), key=len, reverse=True):
        if solver.startswith(key):
            return d[key]
    return default

def clr(solver):   return _match(solver, SOLVER_COLORS,  "#555555")
def mkr(solver):   return _match(solver, SOLVER_MARKERS, "x")
def hatch(solver): return _match(solver, SOLVER_HATCHES, "")
def lbl(solver):   return _match(solver, SOLVER_LABELS,  solver)

def solver_sort_key(solver):
    for i, s in enumerate(SOLVER_ORDER):
        if solver.startswith(s):
            return i
    return 99

def is_converged(df):
    c = df["converged"]
    if c.dtype == bool:
        return c
    return c.astype(str).str.lower() == "true"

def fmt_n(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)

def _save(fig, outdir, stem):
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(plots_dir, stem + ".pdf"))
    plt.close(fig)
    print(f"  [ok] plots/{stem}.{{png,pdf}}")

def _apply_spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


# ─── Plot 1: Convergence ──────────────────────────────────────────────────────

def plot_convergence(traces, outdir):
    if traces.empty:
        print("  [skip] convergence traces empty")
        return

    mdps = sorted(traces["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        df = traces[traces["mdp"] == mdp_name]
        solvers = sorted(df["solver"].unique(), key=solver_sort_key)

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        _apply_spine_style(ax)

        for solver in solvers:
            sd = df[df["solver"] == solver].sort_values("time_sec")
            sd = sd[sd["residual"] > 0]
            if sd.empty:
                continue
            every = max(1, len(sd) // 8)
            ax.semilogy(
                sd["time_sec"], sd["residual"],
                color=clr(solver), marker=mkr(solver),
                linewidth=1.2, markersize=4, markevery=every,
                markeredgewidth=0.5, markeredgecolor="white",
                alpha=0.92, label=lbl(solver),
            )

        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel(r"$\|F(V)-V\|_\infty$")
        ax.set_title(f"Convergence: {mdp_name}", fontsize=9, fontweight="bold")
        ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.yaxis.grid(False, which="minor")
        n_solvers = len(solvers)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9,
                  ncol=1 if n_solvers <= 6 else 2)

        fig.tight_layout()
        _save(fig, outdir, f"conv_{mdp_name.replace(' ', '_')}")
        count += 1

    print(f"  => convergence: {count} figures")


# ─── Plot 2: Solver Ranking ───────────────────────────────────────────────────

def plot_solver_ranking(summary, outdir):
    if summary.empty:
        return
    conv = summary[is_converged(summary)].copy()
    if conv.empty:
        return
    conv = conv[~conv["mdp"].str.contains(r"_b\d|_pb\d", regex=True)]
    mdps = sorted(conv["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        mdf = conv[conv["mdp"] == mdp_name]
        if mdf.empty:
            continue
        best_idx = mdf.groupby("solver")["wall_sec"].idxmin()
        df = mdf.loc[best_idx].sort_values("wall_sec")

        n_s = len(df)
        fig, ax = plt.subplots(figsize=(3.5, max(2.0, n_s * 0.38 + 0.6)))
        _apply_spine_style(ax)
        ax.spines["left"].set_visible(False)

        xmax = df["wall_sec"].max() * 1.45
        for idx, (_, row) in enumerate(df.iterrows()):
            solver = row["solver"]
            val = row["wall_sec"]
            ax.barh(idx, val, height=0.6,
                    color=clr(solver), hatch=hatch(solver),
                    edgecolor="white", linewidth=0.5, alpha=0.85)
            txt = f"{val:.3f} s" if val < 10 else f"{val:.2f} s"
            ax.text(val + xmax * 0.02, idx, txt,
                    va="center", ha="left", fontsize=7, color="#333333")

        labels = []
        for _, row in df.iterrows():
            name = lbl(row["solver"])
            t = int(row["threads"]) if "threads" in row and row["threads"] > 1 else 0
            labels.append(f"{name} ({t}T)" if t > 1 else name)

        ax.set_yticks(range(n_s))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Wall time (s)", fontsize=8)
        ax.set_title(f"Solver ranking: {mdp_name}", fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.yaxis.grid(False)
        ax.tick_params(left=False)

        fig.tight_layout()
        _save(fig, outdir, f"ranking_{mdp_name.replace(' ', '_')}")
        count += 1

    print(f"  => ranking: {count} figures")


# ─── Plot 3: Throughput ───────────────────────────────────────────────────────

def plot_throughput(summary, outdir):
    if summary.empty:
        return
    conv = summary[is_converged(summary)].copy()
    if conv.empty:
        return
    conv = conv[~conv["mdp"].str.contains(r"_b\d|_pb\d", regex=True)]
    mdps = sorted(conv["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        mdf = conv[conv["mdp"] == mdp_name]
        if mdf.empty:
            continue
        best_idx = mdf.groupby("solver")["updates_per_sec"].idxmax()
        df = mdf.loc[best_idx].sort_values("updates_per_sec", ascending=True)

        n_s = len(df)
        fig, ax = plt.subplots(figsize=(3.5, max(2.0, n_s * 0.38 + 0.6)))
        _apply_spine_style(ax)
        ax.spines["left"].set_visible(False)

        xmax = df["updates_per_sec"].max() * 1.35
        for idx, (_, row) in enumerate(df.iterrows()):
            solver = row["solver"]
            ax.barh(idx, row["updates_per_sec"], height=0.6,
                    color=clr(solver), hatch=hatch(solver),
                    edgecolor="white", linewidth=0.5, alpha=0.85)

        labels = []
        for _, row in df.iterrows():
            name = lbl(row["solver"])
            t = int(row["threads"]) if "threads" in row and row["threads"] > 1 else 0
            labels.append(f"{name} ({t}T)" if t > 1 else name)

        ax.set_yticks(range(n_s))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(ticker.EngFormatter(unit="", sep=" "))
        ax.set_xlabel("Updates / s", fontsize=8)
        ax.set_title(f"Throughput: {mdp_name}", fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.yaxis.grid(False)
        ax.tick_params(left=False)

        fig.tight_layout()
        _save(fig, outdir, f"throughput_{mdp_name.replace(' ', '_')}")
        count += 1

    print(f"  => throughput: {count} figures")


# ─── Plot 4: Thread Scaling ───────────────────────────────────────────────────

def plot_thread_scaling(datadir, outdir):
    path = os.path.join(datadir, "thread_scaling.csv")
    if not os.path.exists(path):
        print("  [skip] no thread_scaling.csv")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    mdps = sorted(df["mdp"].unique())
    thread_ticks = sorted(df["threads"].unique())
    max_t = int(df["threads"].max())

    for mdp_name in mdps:
        md = df[df["mdp"] == mdp_name]
        n_label = fmt_n(md["n"].values[0])
        beta_val = md["beta"].values[0]
        solvers = sorted(md["solver"].unique(), key=solver_sort_key)

        fig, (ax_scale, ax_abs) = plt.subplots(1, 2, figsize=(7.0, 2.8))

        for solver in solvers:
            sd = md[md["solver"] == solver].sort_values("threads")
            conv = sd[is_converged(sd)]
            if conv.empty:
                continue
            base1 = conv[conv["threads"] == 1]
            if base1.empty:
                continue
            base_ups = base1["updates_per_sec"].values[0]
            threads = conv["threads"].values
            scaling = conv["updates_per_sec"].values / base_ups
            abs_ups = conv["updates_per_sec"].values / 1e6

            kw = dict(color=clr(solver), marker=mkr(solver), linewidth=1.2,
                      markersize=5, markeredgewidth=0.5, markeredgecolor="white")
            ax_scale.plot(threads, scaling, label=lbl(solver), **kw)
            ax_abs.plot(threads, abs_ups, label=lbl(solver), **kw)

        # Ideal linear reference
        ideal_x = np.array([1, max_t], dtype=float)
        ax_scale.plot(ideal_x, ideal_x, color="#888888", linestyle="--",
                      linewidth=0.9, alpha=0.7, label="Ideal", zorder=0)
        ax_scale.fill_between(ideal_x, ideal_x, alpha=0.04, color="#888888", zorder=0)

        for ax in (ax_scale, ax_abs):
            _apply_spine_style(ax)
            ax.set_xticks(thread_ticks)
            ax.set_xlabel("Threads", fontsize=8)
            ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
            ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")

        ax_scale.set_ylim(0, max_t + 0.5)
        ax_scale.set_ylabel(r"Scaling factor (vs. $T=1$)", fontsize=8)
        ax_scale.set_title("Throughput scaling", fontsize=9, fontweight="bold")
        ax_scale.legend(loc="upper left", fontsize=7, framealpha=0.9)

        ax_abs.set_ylim(bottom=0)
        ax_abs.set_ylabel("Throughput (M updates/s)", fontsize=8)
        ax_abs.set_title("Absolute throughput", fontsize=9, fontweight="bold")
        ax_abs.legend(loc="upper left", fontsize=7, framealpha=0.9)

        fig.suptitle(
            f"Thread scaling: {mdp_name}  ($n={n_label}$, $\\beta={beta_val}$)",
            fontsize=9, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, outdir, f"thread_scaling_{mdp_name.replace(' ', '_')}")

    print("  => thread_scaling: figures saved")


# ─── Plot 5: Size Scaling ─────────────────────────────────────────────────────

def plot_size_scaling(datadir, outdir):
    path = os.path.join(datadir, "size_scaling.csv")
    if not os.path.exists(path):
        print("  [skip] no size_scaling.csv")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    solvers = sorted(df["solver"].unique(), key=solver_sort_key)
    for solver in solvers:
        sd = df[df["solver"] == solver].sort_values("n")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        kw = dict(color=clr(solver), marker=mkr(solver), linewidth=1.2,
                  markersize=5, markeredgewidth=0.5, markeredgecolor="white")
        ax1.loglog(conv["n"], conv["wall_sec"], label=lbl(solver), **kw)
        ax2.loglog(conv["n"], conv["updates_per_sec"], label=lbl(solver), **kw)

    ns = sorted(df["n"].unique())
    if len(ns) >= 2:
        n_arr = np.array(ns, dtype=float)
        jac = df[(df["solver"] == "Jacobi") & is_converged(df)].sort_values("n")
        if len(jac) >= 2:
            ref = jac["wall_sec"].values[0] * (n_arr / n_arr[0])
            ax1.loglog(n_arr, ref, color="#aaaaaa", linestyle=":", linewidth=1.0,
                       alpha=0.7, label=r"$O(n)$ reference", zorder=0)

    for ax in (ax1, ax2):
        _apply_spine_style(ax)
        ax.set_xlabel(r"Problem size $n$", fontsize=8)
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: fmt_n(int(x)) if x >= 1 else str(x)
        ))

    ax1.set_ylabel("Wall time (s)", fontsize=8)
    ax1.set_title("Solve time vs. problem size", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)

    ax2.set_ylabel("Updates / s", fontsize=8)
    ax2.set_title("Throughput vs. problem size", fontsize=9, fontweight="bold")
    ax2.yaxis.set_major_formatter(ticker.EngFormatter(unit="", sep=" "))
    ax2.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle(r"Size scaling (random sparse MDP, $\beta=0.99$)",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "size_scaling")


# ─── Plot 6: Beta Sensitivity ─────────────────────────────────────────────────

def plot_beta_sensitivity(summary, outdir):
    beta_rows = summary[summary["mdp"].str.startswith("Grid_b")]
    if beta_rows.empty:
        print("  [skip] no beta sensitivity data")
        return

    beta_rows = beta_rows.copy()
    beta_rows["beta_val"] = (
        beta_rows["mdp"].str.extract(r"Grid_b(\d+\.\d+)").astype(float)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    solvers = sorted(beta_rows["solver"].unique(), key=solver_sort_key)
    for solver in solvers:
        sd = beta_rows[beta_rows["solver"] == solver].sort_values("beta_val")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        kw = dict(color=clr(solver), marker=mkr(solver), linewidth=1.2,
                  markersize=5, markeredgewidth=0.5, markeredgecolor="white")
        ax1.plot(conv["beta_val"], conv["wall_sec"], label=lbl(solver), **kw)
        ax2.plot(conv["beta_val"], conv["total_updates"], label=lbl(solver), **kw)

    for ax in (ax1, ax2):
        _apply_spine_style(ax)
        ax.set_xlabel(r"Discount factor $\beta$", fontsize=8)
        ax.set_yscale("log")
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    ax1.set_ylabel("Wall time (s)", fontsize=8)
    ax1.set_title(r"Solve time vs. $\beta$", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Total updates", fontsize=8)
    ax2.set_title(r"Iteration count vs. $\beta$", fontsize=9, fontweight="bold")
    ax2.yaxis.set_major_formatter(ticker.EngFormatter(unit="", sep=" "))

    fig.suptitle(r"$\beta$ sensitivity (Grid $30\times30$)",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "beta_sensitivity")


# ─── Plot 7: Difficulty Spectrum ──────────────────────────────────────────────

def plot_difficulty(traces, summary, outdir):
    meta = summary[summary["mdp"].str.startswith("Meta_pb")]
    if meta.empty:
        print("  [skip] no difficulty spectrum data")
        return

    meta = meta.copy()
    meta["pb"] = meta["mdp"].str.extract(r"Meta_pb(\d+\.\d+)").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for solver in sorted(meta["solver"].unique(), key=solver_sort_key):
        sd = meta[meta["solver"] == solver].sort_values("pb")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        ax1.plot(conv["pb"], conv["wall_sec"],
                 color=clr(solver), marker=mkr(solver), linewidth=1.2,
                 markersize=5, markeredgewidth=0.5, markeredgecolor="white",
                 label=lbl(solver))

    _apply_spine_style(ax1)
    ax1.set_xlabel(r"Bridge probability $p_\mathrm{bridge}$ (lower = harder)", fontsize=8)
    ax1.set_ylabel("Wall time (s)", fontsize=8)
    ax1.set_title("Difficulty spectrum: metastable MDP", fontsize=9, fontweight="bold")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
    ax1.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)

    meta_traces = traces[traces["mdp"].str.startswith("Meta_pb")]
    if not meta_traces.empty:
        hardest = sorted(meta_traces["mdp"].unique())[-1]
        td = meta_traces[meta_traces["mdp"] == hardest]
        for solver in sorted(td["solver"].unique(), key=solver_sort_key):
            sd = td[td["solver"] == solver].sort_values("time_sec")
            sd = sd[sd["residual"] > 0]
            if sd.empty:
                continue
            every = max(1, len(sd) // 8)
            ax2.semilogy(sd["time_sec"], sd["residual"],
                         color=clr(solver), marker=mkr(solver), linewidth=1.2,
                         markersize=4, markevery=every,
                         markeredgewidth=0.5, markeredgecolor="white",
                         alpha=0.92, label=lbl(solver))
        pb_label = hardest.replace("Meta_pb", "")
        ax2.set_title(r"$p_\mathrm{bridge}=" + pb_label + r"$ (hardest case)",
                      fontsize=9, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=7, framealpha=0.9)

    _apply_spine_style(ax2)
    ax2.set_xlabel("Wall time (s)", fontsize=8)
    ax2.set_ylabel(r"$\|F(V)-V\|_\infty$", fontsize=8)
    ax2.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
    ax2.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")

    fig.tight_layout()
    _save(fig, outdir, "difficulty_spectrum")


# ─── Plot 8: Summary Heatmap ──────────────────────────────────────────────────

def plot_summary_heatmap(summary, outdir):
    bench1 = summary[~summary["mdp"].str.contains(
        r"_b|_scl|_n|_AT|_pb|MC_k|Rand_n|Rand_500K|Rand_1M|Rand_2M", regex=True
    )]
    conv = bench1[is_converged(bench1)]
    if conv.empty:
        print("  [skip] no convergence data for heatmap")
        return

    pivot = conv.pivot_table(values="wall_sec", index="mdp",
                             columns="solver", aggfunc="min")
    if pivot.empty:
        return

    col_order = pivot.mean().sort_values().index
    pivot = pivot[col_order]

    n_cols = len(pivot.columns)
    n_rows = len(pivot.index)
    fig, ax = plt.subplots(figsize=(max(5.0, n_cols * 1.05 + 1.5),
                                    max(2.5, n_rows * 0.55 + 1.2)))

    vals = pivot.values.copy()
    log_vals = np.where(np.isnan(vals), np.nan,
                        np.log10(np.clip(vals, 1e-6, None)))

    im = ax.imshow(log_vals, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest",
                   vmin=np.nanmin(log_vals), vmax=np.nanmax(log_vals))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([lbl(s) for s in pivot.columns],
                       rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=8)

    median_log = np.nanmedian(log_vals)
    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt = f"{val:.3f}" if val < 10 else f"{val:.1f}"
                lv = log_vals[i, j]
                tc = "white" if (lv > median_log + 0.5 or lv < median_log - 0.5) else "#222222"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=tc, fontweight="bold")

    ax.set_title(r"Wall time heatmap: MDP $\times$ Solver (seconds)",
                 fontsize=9, fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label(r"$\log_{10}(\mathrm{seconds})$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.tick_params(bottom=False, left=False)

    fig.tight_layout()
    _save(fig, outdir, "heatmap")


# ─── Plot 9: AutoTune Summary ─────────────────────────────────────────────────

def plot_autotune(datadir, outdir):
    path = os.path.join(datadir, "autotune.csv")
    if not os.path.exists(path):
        print("  [skip] no autotune.csv")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    palette = [_OI["blue"], _OI["vermillion"], _OI["green"]]
    n_rows = len(df)
    colors_list = [palette[i % len(palette)] for i in range(n_rows)]
    h_list      = ["", "///", "..."][:n_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, max(2.0, n_rows * 0.6 + 1.0)))

    for ax in (ax1, ax2):
        _apply_spine_style(ax)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
        ax.tick_params(left=False)

    bar_h = 0.55
    xmax1 = df["wall_sec"].max() * 1.4
    for i, row in df.iterrows():
        ax1.barh(i, row["wall_sec"], height=bar_h,
                 color=colors_list[i], hatch=h_list[i],
                 edgecolor="white", linewidth=0.5, alpha=0.85)
        ax1.text(row["wall_sec"] + xmax1 * 0.025, i,
                 f'{row["planner"]}, blk={int(row["blk"])}',
                 va="center", ha="left", fontsize=7, color="#333333")
    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels(df["mdp"], fontsize=8)
    ax1.set_xlim(0, xmax1)
    ax1.set_xlabel("Wall time (s)", fontsize=8)
    ax1.set_title("AutoTune: best config wall time", fontsize=9, fontweight="bold")

    xmax2 = df["ups"].max() * 1.3
    for i, row in df.iterrows():
        ax2.barh(i, row["ups"], height=bar_h,
                 color=colors_list[i], hatch=h_list[i],
                 edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_yticks(range(n_rows))
    ax2.set_yticklabels(df["mdp"], fontsize=8)
    ax2.set_xlim(0, xmax2)
    ax2.xaxis.set_major_formatter(ticker.EngFormatter(unit="", sep=" "))
    ax2.set_xlabel("Updates / s", fontsize=8)
    ax2.set_title("AutoTune: throughput", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, outdir, "autotune")


# ─── Plot 10: NEON SIMD Speedup ───────────────────────────────────────────────

def _parse_simd_txt(path):
    sections = {}
    current = None
    rows = []
    headers = None

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip()
            if line.startswith("====== Bench"):
                if current and rows:
                    sections[current] = (headers, rows)
                label = line.strip("= ").replace("Bench ", "").rstrip("=").strip()
                current = label
                rows = []
                headers = None
            elif line.startswith("n ") or line.startswith("Operator"):
                headers = line.split()
            elif current and headers and line and not line.startswith("=") and not line.startswith("ARM"):
                parts = line.split()
                if len(parts) >= 4:
                    rows.append(parts)

    if current and rows:
        sections[current] = (headers, rows)
    return sections

def plot_simd(datadir, outdir):
    path = os.path.join(datadir, "simd_benchmark.txt")
    if not os.path.exists(path):
        print("  [skip] no simd_benchmark.txt")
        return

    sections = _parse_simd_txt(path)
    if not sections:
        print("  [skip] simd_benchmark.txt could not be parsed")
        return

    # Figure 1: Bench A — dot product speedup vs nnz/row
    bench_a_key = next((k for k in sections if "A:" in k), None)
    if bench_a_key:
        _, rows = sections[bench_a_key]
        try:
            nnz_vals = [int(r[1]) for r in rows]
            speedups  = [float(r[3].rstrip("x")) for r in rows]

            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            _apply_spine_style(ax)
            x = np.arange(len(nnz_vals))
            bars = ax.bar(x, speedups, width=0.55, color=_OI["blue"],
                          edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
            for bar, sp in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{sp:.2f}x", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in nnz_vals], fontsize=8)
            ax.set_xlabel("Non-zeros per row", fontsize=8)
            ax.set_ylabel("NEON speedup vs. scalar", fontsize=8)
            ax.set_title("NEON: sparse dot product\n($n=50{,}000$)",
                         fontsize=9, fontweight="bold")
            ax.set_ylim(0, max(speedups) * 1.25)
            ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
            ax.xaxis.grid(False)
            fig.tight_layout()
            _save(fig, outdir, "simd_dot_product")
        except (ValueError, IndexError):
            pass

    # Figure 2: Bench B + E — sweep speedup vs n
    bench_data = []
    for key, label in [
        (next((k for k in sections if "B:" in k), None), "Jacobi sweep"),
        (next((k for k in sections if "E:" in k), None), "Gauss-Seidel sweep"),
    ]:
        if key is None:
            continue
        _, rows = sections[key]
        try:
            bench_data.append((label, [int(r[0]) for r in rows],
                               [float(r[3].rstrip("x")) for r in rows]))
        except (ValueError, IndexError):
            pass

    if bench_data:
        n_panels = len(bench_data)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 2.6))
        if n_panels == 1:
            axes = [axes]
        for ax, (label, ns_, speedups_), color in zip(
            axes, bench_data, [_OI["blue"], _OI["vermillion"]]
        ):
            _apply_spine_style(ax)
            x = np.arange(len(ns_))
            bars = ax.bar(x, speedups_, width=0.55, color=color,
                          edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
            for bar, sp in zip(bars, speedups_):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{sp:.2f}x", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels([fmt_n(v) for v in ns_], fontsize=8)
            ax.set_xlabel(r"Problem size $n$", fontsize=8)
            ax.set_ylabel("NEON speedup vs. scalar", fontsize=8)
            ax.set_title(f"NEON: {label}\n(nnz/row = 20)",
                         fontsize=9, fontweight="bold")
            ax.set_ylim(0, max(speedups_) * 1.3)
            ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
            ax.xaxis.grid(False)
        fig.tight_layout()
        _save(fig, outdir, "simd_sweep_speedup")

    print("  => simd: figures saved")


# ─── Plot 11: Solver Family Comparison ────────────────────────────────────────

def plot_solver_family_comparison(summary, outdir):
    if summary.empty:
        return
    conv = summary[is_converged(summary)].copy()
    conv = conv[~conv["mdp"].str.contains(
        r"_b\d|_pb\d|_n\d|_scl|_AT|Rand_500K|Rand_1M", regex=True
    )]
    if conv.empty:
        return

    families = {
        "Single-thread": ["Jacobi", "GaussSeidel"],
        "Plan":          ["Plan_Static", "Plan_Colored", "Plan_Priority"],
        "Async":         ["Async_Static", "Async_Shuffled", "Async_TopKGS",
                          "Async_CATopKGS", "Async_ResBuck"],
    }
    fam_colors  = {"Single-thread": _OI["blue"], "Plan": _OI["orange"], "Async": _OI["green"]}
    fam_hatches = {"Single-thread": "",          "Plan": "///",          "Async": "..."}

    mdps = sorted(conv["mdp"].unique())
    if not mdps:
        return

    n_mdps = len(mdps)
    n_fams = len(families)
    fig, ax = plt.subplots(figsize=(max(5.0, n_mdps * 1.4), 2.8))
    _apply_spine_style(ax)

    bar_w = 0.75 / n_fams
    x = np.arange(n_mdps)

    for fam_idx, (fam_name, solvers) in enumerate(families.items()):
        best_times = []
        for mdp_name in mdps:
            mdf = conv[conv["mdp"] == mdp_name]
            fam_df = mdf[mdf["solver"].isin(solvers)]
            best_times.append(fam_df["wall_sec"].min() if not fam_df.empty else np.nan)

        offset = (fam_idx - (n_fams - 1) / 2) * bar_w
        ax.bar(x + offset, best_times, width=bar_w * 0.92,
               color=fam_colors[fam_name], hatch=fam_hatches[fam_name],
               edgecolor="white", linewidth=0.5, alpha=0.85, label=fam_name)

    ax.set_xticks(x)
    ax.set_xticklabels(mdps, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Best wall time (s)", fontsize=8)
    ax.set_title("Solver family comparison across core MDPs",
                 fontsize=9, fontweight="bold")
    ax.yaxis.grid(True, which="major", alpha=0.3, linewidth=0.5, color="#999999")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    _save(fig, outdir, "family_comparison")
    print("  => family_comparison saved")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Helios publication-quality plot suite"
    )
    parser.add_argument(
        "--results-dir", "-d",
        default="bench/results",
        metavar="DIR",
        help="Path to bench/results directory (default: bench/results)",
    )
    parser.add_argument("positional", nargs="?", default=None)
    args = parser.parse_args()

    datadir = args.positional if args.positional else args.results_dir

    if not os.path.isdir(datadir):
        print(f"Error: '{datadir}' is not a directory.")
        sys.exit(1)

    plots_dir = os.path.join(datadir, "plots")
    print(f"Helios Plot Suite — reading from {datadir}/")
    print(f"Output directory  : {plots_dir}/\n")

    traces  = pd.DataFrame()
    summary = pd.DataFrame()

    trace_path   = os.path.join(datadir, "convergence_traces.csv")
    summary_path = os.path.join(datadir, "summary.csv")

    if os.path.exists(trace_path):
        traces = pd.read_csv(trace_path)
        print(f"  Loaded {len(traces):,} convergence trace rows")
    else:
        print(f"  [warn] {trace_path} not found")

    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        print(f"  Loaded {len(summary):,} summary rows")
    else:
        print(f"  [warn] {summary_path} not found")

    print("\nGenerating plots...\n")

    plot_convergence(traces, datadir)
    plot_solver_ranking(summary, datadir)
    plot_throughput(summary, datadir)
    plot_thread_scaling(datadir, datadir)
    plot_size_scaling(datadir, datadir)
    plot_beta_sensitivity(summary, datadir)
    plot_difficulty(traces, summary, datadir)
    plot_summary_heatmap(summary, datadir)
    plot_autotune(datadir, datadir)
    plot_simd(datadir, datadir)
    plot_solver_family_comparison(summary, datadir)

    print(f"\nAll plots saved to {plots_dir}/")
    print("Each figure saved as both .png (300 DPI) and .pdf (vector).")


if __name__ == "__main__":
    main()
