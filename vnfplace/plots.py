"""Publication figures (vector PDF + PNG) from benchmark results.

Colour follows the method (fixed order, never re-assigned when a method is
missing), every series also has its own marker so identity never relies on
colour alone, and there is never a second y-axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]
GREY = "#8a8985"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
    "lines.markersize": 6, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def style_map(methods):
    """Learned methods first (slot 1 = proposed), then baselines; >8 methods fall back to grey."""
    order = sorted(methods, key=lambda m: ("PPO" not in m and not m.startswith("GNN"), m))
    return {m: (PALETTE[i] if i < len(PALETTE) else GREY, MARKERS[i % len(MARKERS)]) for i, m in enumerate(order)}


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def _ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def fig_vs_load(df: pd.DataFrame, metric: str, ylabel: str, out: Path, name: str):
    sm = style_map(df.method.unique())
    for topo, dt in df.groupby("topology"):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for m, g in dt.groupby("method"):
            agg = g.groupby("load")[metric].apply(lambda s: _ci(s)).apply(pd.Series)
            c, mk = sm[m]
            ax.errorbar(agg.index, agg[0], yerr=agg[1], label=m, color=c, marker=mk, capsize=2)
        ax.set_xlabel("Offered load (edge+fog CPU)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=2)
        _save(fig, out, f"{name}_{topo}")


def fig_pareto(df: pd.DataFrame, out: Path, x="avg_power_w", y="acceptance"):
    sm = style_map(df.method.unique())
    for topo, dt in df.groupby("topology"):
        loads = sorted(dt.load.unique())
        fig, axes = plt.subplots(1, len(loads), figsize=(2.4 * len(loads), 2.6), squeeze=False)
        for ax, ld in zip(axes[0], loads):
            for m, g in dt[dt.load == ld].groupby("method"):
                (mx, ex), (my, ey) = _ci(g[x]), _ci(g[y])
                c, mk = sm[m]
                ax.errorbar(mx, my, xerr=ex, yerr=ey, color=c, marker=mk, capsize=2, label=m, linestyle="none")
            ax.set_title(f"load = {ld:g}")
            ax.set_xlabel("Average power (W)")
        axes[0][0].set_ylabel("Acceptance ratio")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=min(4, len(lab)), frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        _save(fig, out, f"pareto_{topo}")


def fig_latency_cdf(req: pd.DataFrame, out: Path, load: float):
    sm = style_map(req.method.unique())
    d = req[(req.load == load) & req.accepted]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m, g in d.groupby("method"):
        v = np.sort(g.latency_ms.values)
        c, mk = sm[m]
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=m, marker=mk, markevery=max(1, len(v) // 6))
    ax.set_xlabel("End-to-end latency of accepted requests (ms)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False)
    _save(fig, out, f"latency_cdf_load{load:g}")


def fig_cd(fr: dict, out: Path):
    """Critical-difference diagram (Demsar 2006) from stats.friedman_nemenyi output."""
    ranks = fr["avg_ranks"]
    k, cd = len(ranks), fr["nemenyi_cd"]
    fig, ax = plt.subplots(figsize=(5.0, 0.4 + 0.22 * k))
    ax.set_xlim(0.8, k + 0.2)
    ax.set_ylim(0, k + 1)
    ax.axis("off")
    ax.hlines(k + 0.5, 1, k, color="#52514e", linewidth=1)
    for r in range(1, k + 1):
        ax.vlines(r, k + 0.4, k + 0.6, color="#52514e", linewidth=1)
        ax.text(r, k + 0.75, str(r), ha="center", fontsize=8)
    for i, (m, r) in enumerate(sorted(ranks.items(), key=lambda t: t[1])):
        y = k - 0.2 - i * 0.9
        ax.plot([r, r], [y, k + 0.5], color="#52514e", linewidth=0.8)
        ax.text(r, y - 0.15, f"{m} ({r:.2f})", ha="center", va="top", fontsize=7)
    ax.hlines(0.3, 1, 1 + cd, color="#0b0b0b", linewidth=2)
    ax.text(1 + cd / 2, 0.45, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(f"{fr['metric']}: Friedman p = {fr['friedman_p']:.2g}, {fr['n_blocks']} blocks", fontsize=8)
    _save(fig, out, f"cd_{fr['metric']}")


def fig_training(run_dirs, out: Path, name="training"):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    groups = {}
    for d in run_dirs:
        d = Path(d)
        log = pd.read_csv(d / "train_log.csv")
        groups.setdefault(d.name.rsplit("_s", 1)[0], []).append(log.dropna(subset=["val_score"]))
    sm = style_map(list(groups))
    for label, logs in groups.items():
        steps = logs[0].steps.values
        vals = np.stack([lg.val_score.values[:len(steps)] for lg in logs if len(lg) >= len(steps)])
        mean, sd = vals.mean(0), vals.std(0)
        c, mk = sm[label]
        ax.plot(steps, mean, color=c, marker=mk, markevery=max(1, len(steps) // 6), label=f"{label} (n={len(vals)})")
        ax.fill_between(steps, mean - sd, mean + sd, color=c, alpha=0.15, linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Validation objective")
    ax.legend(frameon=False)
    _save(fig, out, name)


def make_all(results: Path, out: Path):
    df = pd.read_csv(results / "summary.csv")
    fig_vs_load(df, "acceptance", "Acceptance ratio", out, "acceptance_vs_load")
    fig_vs_load(df, "energy_per_accepted_j", "Energy per accepted request (J)", out, "energy_vs_load")
    fig_vs_load(df, "avg_power_w", "Average power (W)", out, "power_vs_load")
    fig_vs_load(df, "latency_p95_ms", "P95 latency (ms)", out, "p95_vs_load")
    fig_pareto(df, out)
    rq = results / "requests.csv.gz"
    if rq.exists():
        req = pd.read_csv(rq)
        for ld in sorted(req.load.unique()):
            fig_latency_cdf(req, out, ld)
    fn = results / "friedman_nemenyi.json"
    if fn.exists():
        for fr in json.loads(fn.read_text()):
            fig_cd(fr, out)
