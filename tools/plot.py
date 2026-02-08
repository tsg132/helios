#!/usr/bin/env python3
"""
tools/plot.py — Helios benchmark visualization suite.

Reads CSV output from bench/run_bench and generates publication-quality plots.

Usage:
    python3 tools/plot.py bench/results
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ─── Global Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "grid.color":       "#bbbbbb",
    "font.size":        12,
    "font.family":      "sans-serif",
    "axes.titlesize":   16,
    "axes.titleweight":  "bold",
    "axes.labelsize":   13,
    "axes.labelweight":  "medium",
    "legend.fontsize":  10,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#dddddd",
    "legend.fancybox":   True,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.2,
    "lines.linewidth":  2.5,
    "lines.markersize": 8,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
})

# ─── Color Palette (bold, high-contrast, colorblind-safe) ────────────────────

SOLVER_COLORS = {
    "Jacobi":          "#1976D2",   # strong blue
    "GaussSeidel":     "#E65100",   # deep orange
    "Async_Static":    "#2E7D32",   # forest green
    "Async_Shuffled":  "#7B1FA2",   # purple
    "Async_TopKGS":    "#C62828",   # crimson
    "Async_CATopKGS":  "#5D4037",   # brown
    "Async_ResBuck":   "#AD1457",   # magenta
    "Plan_Static":     "#00838F",   # teal
    "Plan_Colored":    "#EF6C00",   # amber
    "Plan_Priority":   "#4527A0",   # indigo
    "AT_Static":       "#00838F",   # teal (same as Plan)
}

SOLVER_MARKERS = {
    "Jacobi":          "o",
    "GaussSeidel":     "s",
    "Async_Static":    "^",
    "Async_Shuffled":  "D",
    "Async_TopKGS":    "v",
    "Async_CATopKGS":  "p",
    "Async_ResBuck":   "h",
    "Plan_Static":     "P",
    "Plan_Colored":    "X",
    "Plan_Priority":   "*",
    "AT_Static":       "P",
}

SOLVER_LABELS = {
    "Jacobi":          "Jacobi",
    "GaussSeidel":     "Gauss-Seidel",
    "Async_Static":    "Async (Static)",
    "Async_Shuffled":  "Async (Shuffled)",
    "Async_TopKGS":    "Async (TopK-GS)",
    "Async_CATopKGS":  "Async (CA-TopK)",
    "Async_ResBuck":   "Async (ResBucket)",
    "Plan_Static":     "Plan (Static)",
    "Plan_Colored":    "Plan (Colored)",
    "Plan_Priority":   "Plan (Priority)",
    "AT_Static":       "AutoTune",
}

# Solver display order (best first in each category)
SOLVER_ORDER = [
    "Jacobi", "GaussSeidel",
    "Plan_Static", "Plan_Colored", "Plan_Priority",
    "Async_Static", "Async_Shuffled", "Async_TopKGS",
    "Async_CATopKGS", "Async_ResBuck",
]

def _match_solver(solver, d):
    """Match solver name to dict, supporting variants like Plan_Static_4T."""
    if solver in d:
        return d[solver]
    for key in sorted(d.keys(), key=len, reverse=True):
        if solver.startswith(key):
            return d[key]
    return None

def clr(solver):  return _match_solver(solver, SOLVER_COLORS) or "#555555"
def mkr(solver):  return _match_solver(solver, SOLVER_MARKERS) or "x"
def lbl(solver):  return _match_solver(solver, SOLVER_LABELS) or solver

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
    """Format problem size nicely: 1000 -> '1K', 1000000 -> '1M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


# ─── Plot 1: Convergence (one figure per MDP) ────────────────────────────────

def plot_convergence(traces, outdir):
    """One convergence figure per MDP — clean, readable, no cramming."""
    if traces.empty:
        print("  [skip] convergence traces empty")
        return

    mdps = sorted(traces["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        df = traces[traces["mdp"] == mdp_name]
        solvers = sorted(df["solver"].unique(), key=solver_sort_key)

        fig, ax = plt.subplots(figsize=(10, 6.5))

        for solver in solvers:
            sd = df[df["solver"] == solver].sort_values("time_sec")
            sd = sd[sd["residual"] > 0]
            if sd.empty:
                continue
            me = max(1, len(sd) // 12)
            ax.semilogy(sd["time_sec"], sd["residual"],
                        color=clr(solver), marker=mkr(solver),
                        markersize=6, markevery=me, alpha=0.9,
                        label=lbl(solver))

        ax.set_xlabel("Wall Time (sec)")
        ax.set_ylabel("Residual")
        ax.set_title(f"Convergence: {mdp_name}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

        fig.tight_layout()
        safe = mdp_name.replace(" ", "_")
        fig.savefig(f"{outdir}/conv_{safe}.png", dpi=150)
        plt.close(fig)
        count += 1

    print(f"  [ok] conv_*.png ({count} figures)")


# ─── Plot 2: Solver Ranking (one figure per MDP) ─────────────────────────────

def plot_solver_ranking(summary, outdir):
    """One wall-time ranking bar chart per MDP. Includes all MDPs."""
    if summary.empty:
        return

    conv = summary[is_converged(summary)].copy()
    if conv.empty:
        return

    # Exclude parameter-sweep variants (beta sweep, difficulty sweep)
    conv = conv[~conv["mdp"].str.contains("_b\\d|_pb\\d", regex=True)]

    mdps = sorted(conv["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        mdf = conv[conv["mdp"] == mdp_name]
        if mdf.empty:
            continue

        # Deduplicate: keep best (min wall_sec) per solver, note thread count
        best_idx = mdf.groupby("solver")["wall_sec"].idxmin()
        df = mdf.loc[best_idx].sort_values("wall_sec")

        n_solvers = len(df)
        fig_h = max(3.5, n_solvers * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))

        colors = [clr(s) for s in df["solver"]]
        bars = ax.barh(range(n_solvers), df["wall_sec"], color=colors, alpha=0.9,
                       edgecolor="white", linewidth=0.8, height=0.65)

        # Label with solver name + thread count if >1
        labels = []
        for _, row in df.iterrows():
            name = lbl(row["solver"])
            t = int(row["threads"]) if "threads" in row and row["threads"] > 1 else 0
            labels.append(f"{name} ({t}T)" if t > 1 else name)

        ax.set_yticks(range(n_solvers))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()

        xmax = df["wall_sec"].max() * 1.3
        for bar, val in zip(bars, df["wall_sec"]):
            txt = f"{val:.3f}s" if val < 10 else f"{val:.1f}s"
            ax.text(bar.get_width() + xmax * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    txt, va="center", fontsize=10, color="#333",
                    fontweight="medium")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Wall Time (sec)")
        ax.set_title(f"Solver Ranking: {mdp_name}", fontsize=14, fontweight="bold")

        fig.tight_layout()
        safe = mdp_name.replace(" ", "_")
        fig.savefig(f"{outdir}/ranking_{safe}.png", dpi=150)
        plt.close(fig)
        count += 1

    print(f"  [ok] ranking_*.png ({count} figures)")


def plot_throughput(summary, outdir):
    """One throughput bar chart per MDP."""
    if summary.empty:
        return

    conv = summary[is_converged(summary)].copy()
    if conv.empty:
        return

    conv = conv[~conv["mdp"].str.contains("_b\\d|_pb\\d", regex=True)]

    mdps = sorted(conv["mdp"].unique())
    count = 0

    for mdp_name in mdps:
        mdf = conv[conv["mdp"] == mdp_name]
        if mdf.empty:
            continue

        # Deduplicate: keep best (max throughput) per solver
        best_idx = mdf.groupby("solver")["updates_per_sec"].idxmax()
        df = mdf.loc[best_idx].sort_values("updates_per_sec", ascending=True)

        n_solvers = len(df)
        fig_h = max(3.5, n_solvers * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))

        colors = [clr(s) for s in df["solver"]]
        bars = ax.barh(range(n_solvers), df["updates_per_sec"], color=colors, alpha=0.9,
                       edgecolor="white", linewidth=0.8, height=0.65)

        labels = []
        for _, row in df.iterrows():
            name = lbl(row["solver"])
            t = int(row["threads"]) if "threads" in row and row["threads"] > 1 else 0
            labels.append(f"{name} ({t}T)" if t > 1 else name)

        ax.set_yticks(range(n_solvers))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        ax.set_xlabel("Updates / sec")
        ax.set_title(f"Throughput: {mdp_name}", fontsize=14, fontweight="bold")

        fig.tight_layout()
        safe = mdp_name.replace(" ", "_")
        fig.savefig(f"{outdir}/throughput_{safe}.png", dpi=150)
        plt.close(fig)
        count += 1

    print(f"  [ok] throughput_*.png ({count} figures)")


# ─── Plot 3: Thread Scaling (one figure per MDP) ─────────────────────────────

def plot_thread_scaling(outdir):
    """One figure per MDP, each with 2 panels: throughput scaling + absolute."""
    path = os.path.join(outdir, "thread_scaling.csv")
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
        n_val = md["n"].values[0]
        n_label = fmt_n(n_val)
        solvers = sorted(md["solver"].unique(), key=solver_sort_key)

        fig, (ax_scale, ax_abs) = plt.subplots(1, 2, figsize=(14, 6))

        for solver in solvers:
            sd = md[md["solver"] == solver].sort_values("threads")
            conv = sd[is_converged(sd)]
            if conv.empty:
                continue

            base = conv[conv["threads"] == 1]
            if base.empty:
                continue
            base_ups = base["updates_per_sec"].values[0]

            scaling = conv["updates_per_sec"].values / base_ups
            threads = conv["threads"].values
            abs_ups = conv["updates_per_sec"].values / 1e6

            ax_scale.plot(threads, scaling,
                          color=clr(solver), marker=mkr(solver),
                          markersize=10, label=lbl(solver), linewidth=2.8)
            ax_abs.plot(threads, abs_ups,
                        color=clr(solver), marker=mkr(solver),
                        markersize=10, label=lbl(solver), linewidth=2.8)

            # Annotate each multi-thread point with scaling value
            for t, s in zip(threads, scaling):
                if t > 1:
                    ax_scale.annotate(f"{s:.2f}x",
                                      (t, s), textcoords="offset points",
                                      xytext=(10, 5), fontsize=11,
                                      color=clr(solver), fontweight="bold")

            # Annotate absolute throughput values
            for t, u in zip(threads, abs_ups):
                ax_abs.annotate(f"{u:.0f}M",
                                (t, u), textcoords="offset points",
                                xytext=(10, 5), fontsize=10,
                                color=clr(solver), fontweight="medium")

        # Ideal scaling line
        ax_scale.plot([1, max_t], [1, max_t], color="#aaaaaa", linestyle="--",
                      linewidth=1.5, alpha=0.5, label="Ideal linear", zorder=0)
        ax_scale.fill_between([1, max_t], [1, max_t], alpha=0.03, color="#999999")

        ax_scale.set_xticks(thread_ticks)
        ax_scale.set_ylim(0, max_t + 0.5)
        ax_scale.set_xlabel("Threads")
        ax_scale.set_ylabel("Throughput Scaling (vs T=1)")
        ax_scale.set_title("Throughput Scaling", fontsize=14, fontweight="bold")
        ax_scale.legend(fontsize=10, loc="upper left")

        ax_abs.set_xticks(thread_ticks)
        ax_abs.set_ylim(bottom=0)
        ax_abs.set_xlabel("Threads")
        ax_abs.set_ylabel("Throughput (M updates/sec)")
        ax_abs.set_title("Absolute Throughput", fontsize=14, fontweight="bold")
        ax_abs.legend(fontsize=10, loc="upper left")

        fig.suptitle(f"Thread Scaling: {mdp_name} (n={n_label}, beta={md['beta'].values[0]})",
                     fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        safe_name = mdp_name.replace(" ", "_")
        fig.savefig(f"{outdir}/thread_scaling_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  [ok] thread_scaling_{safe_name}.png")


# ─── Plot 4: Size Scaling ────────────────────────────────────────────────────

def plot_size_scaling(outdir):
    """Log-log: wall time and throughput vs problem size."""
    path = os.path.join(outdir, "size_scaling.csv")
    if not os.path.exists(path):
        print("  [skip] no size_scaling.csv")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    solvers = sorted(df["solver"].unique(), key=solver_sort_key)
    for solver in solvers:
        sd = df[df["solver"] == solver].sort_values("n")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        ax1.loglog(conv["n"], conv["wall_sec"],
                   color=clr(solver), marker=mkr(solver),
                   markersize=8, label=lbl(solver))
        ax2.loglog(conv["n"], conv["updates_per_sec"],
                   color=clr(solver), marker=mkr(solver),
                   markersize=8, label=lbl(solver))

    # O(n) reference line on wall time
    ns = sorted(df["n"].unique())
    if len(ns) >= 2:
        n_arr = np.array(ns, dtype=float)
        jac = df[(df["solver"] == "Jacobi") & is_converged(df)].sort_values("n")
        if len(jac) >= 2:
            ref = jac["wall_sec"].values[0] * (n_arr / n_arr[0])
            ax1.loglog(n_arr, ref, color="#999999", linestyle=":", linewidth=1.5,
                       alpha=0.5, label="O(n) reference")

    ax1.set_xlabel("Problem Size (n)")
    ax1.set_ylabel("Wall Time (sec)")
    ax1.set_title("Size Scaling: Solve Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Problem Size (n)")
    ax2.set_ylabel("Updates / sec")
    ax2.set_title("Size Scaling: Throughput", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(ticker.EngFormatter())

    fig.tight_layout()
    fig.savefig(f"{outdir}/size_scaling.png", dpi=150)
    plt.close(fig)
    print("  [ok] size_scaling.png")


# ─── Plot 5: Beta Sensitivity ────────────────────────────────────────────────

def plot_beta_sensitivity(summary, outdir):
    """How convergence time grows with beta approaching 1."""
    beta_rows = summary[summary["mdp"].str.startswith("Grid_b")]
    if beta_rows.empty:
        print("  [skip] no beta data")
        return

    beta_rows = beta_rows.copy()
    beta_rows["beta_val"] = beta_rows["mdp"].str.extract(r"Grid_b(\d+\.\d+)").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    solvers = sorted(beta_rows["solver"].unique(), key=solver_sort_key)
    for solver in solvers:
        sd = beta_rows[beta_rows["solver"] == solver].sort_values("beta_val")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        ax1.plot(conv["beta_val"], conv["wall_sec"],
                 color=clr(solver), marker=mkr(solver),
                 markersize=8, label=lbl(solver))
        ax2.plot(conv["beta_val"], conv["total_updates"],
                 color=clr(solver), marker=mkr(solver),
                 markersize=8, label=lbl(solver))

    ax1.set_xlabel("Discount Factor (beta)")
    ax1.set_ylabel("Wall Time (sec)")
    ax1.set_title("Beta Sensitivity: Time to Converge", fontsize=14, fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Discount Factor (beta)")
    ax2.set_ylabel("Total Updates")
    ax2.set_title("Beta Sensitivity: Iterations Required", fontsize=14, fontweight="bold")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.EngFormatter())
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{outdir}/beta_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  [ok] beta_sensitivity.png")


# ─── Plot 6: Difficulty Spectrum ──────────────────────────────────────────────

def plot_difficulty(traces, summary, outdir):
    """Metastable bridge probability: wall time + convergence of hardest case."""
    meta = summary[summary["mdp"].str.startswith("Meta_pb")]
    if meta.empty:
        print("  [skip] no difficulty data")
        return

    meta = meta.copy()
    meta["pb"] = meta["mdp"].str.extract(r"Meta_pb(\d+\.\d+)").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: wall time vs bridge probability
    solvers = sorted(meta["solver"].unique(), key=solver_sort_key)
    for solver in solvers:
        sd = meta[meta["solver"] == solver].sort_values("pb")
        conv = sd[is_converged(sd)]
        if conv.empty:
            continue
        ax1.plot(conv["pb"], conv["wall_sec"],
                 color=clr(solver), marker=mkr(solver),
                 markersize=8, label=lbl(solver))

    ax1.set_xlabel("Bridge Probability (lower = harder)")
    ax1.set_ylabel("Wall Time (sec)")
    ax1.set_title("Difficulty Spectrum: Metastable MDP", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.invert_xaxis()

    # Right: convergence traces for hardest case
    meta_traces = traces[traces["mdp"].str.startswith("Meta_pb")]
    if not meta_traces.empty:
        hardest = sorted(meta_traces["mdp"].unique())[-1]
        td = meta_traces[meta_traces["mdp"] == hardest]
        solvers_t = sorted(td["solver"].unique(), key=solver_sort_key)
        for solver in solvers_t:
            sd = td[td["solver"] == solver].sort_values("time_sec")
            sd = sd[sd["residual"] > 0]
            if sd.empty:
                continue
            me = max(1, len(sd) // 10)
            ax2.semilogy(sd["time_sec"], sd["residual"],
                         color=clr(solver), marker=mkr(solver),
                         markersize=4, markevery=me, alpha=0.9,
                         label=lbl(solver))
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Residual")
        ax2.set_title(f"Convergence: {hardest} (hardest)", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/difficulty_spectrum.png", dpi=150)
    plt.close(fig)
    print("  [ok] difficulty_spectrum.png")


# ─── Plot 7: Summary Heatmap ─────────────────────────────────────────────────

def plot_summary_heatmap(summary, outdir):
    """Heatmap: MDP x Solver -> wall time."""
    bench1 = summary[~summary["mdp"].str.contains(
        "_b|_scl|_n|_AT|_pb|MC_k|Rand_n|Rand_500K|Rand_1M|Rand_2M", regex=True)]
    conv = bench1[is_converged(bench1)]
    if conv.empty:
        print("  [skip] no convergence data for heatmap")
        return

    pivot = conv.pivot_table(values="wall_sec", index="mdp",
                             columns="solver", aggfunc="min")
    if pivot.empty:
        return

    # Reorder columns by mean time
    col_order = pivot.mean().sort_values().index
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.4),
                                     max(4, len(pivot) * 1.0)))

    # Log-scaled colors
    vals = pivot.values.copy()
    log_vals = np.where(np.isnan(vals), np.nan,
                        np.log10(np.clip(vals, 1e-6, None)))

    im = ax.imshow(log_vals, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([lbl(s) for s in pivot.columns],
                       rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Annotate cells
    median_log = np.nanmedian(log_vals)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt = f"{val:.3f}s" if val < 10 else f"{val:.1f}s"
                text_color = "white" if log_vals[i, j] > median_log else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("Wall Time Heatmap: MDP x Solver",
                 fontsize=15, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("log10(seconds)", fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{outdir}/heatmap.png", dpi=150)
    plt.close(fig)
    print("  [ok] heatmap.png")


# ─── Plot 8: Autotune Summary ────────────────────────────────────────────────

def plot_autotune(outdir):
    path = os.path.join(outdir, "autotune.csv")
    if not os.path.exists(path):
        print("  [skip] no autotune.csv")
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    colors_list = ["#00838F", "#2E7D32", "#E65100"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.barh(df["mdp"], df["wall_sec"],
             color=colors_list[:len(df)], alpha=0.9,
             edgecolor="white", linewidth=0.8, height=0.5)
    for i, row in df.iterrows():
        ax1.text(row["wall_sec"] + ax1.get_xlim()[1] * 0.03, i,
                 f'{row["planner"]} (blk={int(row["blk"])})',
                 fontsize=10, va="center")
    ax1.set_xlabel("Wall Time (sec)")
    ax1.set_title("AutoTune: Best Config Performance", fontsize=14, fontweight="bold")

    ax2.barh(df["mdp"], df["ups"],
             color=colors_list[:len(df)], alpha=0.9,
             edgecolor="white", linewidth=0.8, height=0.5)
    ax2.set_xlabel("Updates / sec")
    ax2.set_title("AutoTune: Throughput", fontsize=14, fontweight="bold")
    ax2.xaxis.set_major_formatter(ticker.EngFormatter())

    fig.tight_layout()
    fig.savefig(f"{outdir}/autotune.png", dpi=150)
    plt.close(fig)
    print("  [ok] autotune.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        datadir = "bench/results"
    else:
        datadir = sys.argv[1]

    if not os.path.isdir(datadir):
        print(f"Error: {datadir} is not a directory")
        sys.exit(1)

    print(f"Helios Plot Suite — reading from {datadir}/\n")

    traces = pd.DataFrame()
    summary = pd.DataFrame()

    trace_path = os.path.join(datadir, "convergence_traces.csv")
    summary_path = os.path.join(datadir, "summary.csv")

    if os.path.exists(trace_path):
        traces = pd.read_csv(trace_path)
        print(f"  Loaded {len(traces):,} trace rows")
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
    plot_beta_sensitivity(summary, datadir)
    plot_thread_scaling(datadir)
    plot_size_scaling(datadir)
    plot_difficulty(traces, summary, datadir)
    plot_summary_heatmap(summary, datadir)
    plot_autotune(datadir)

    print(f"\nAll plots saved to {datadir}/")

if __name__ == "__main__":
    main()
