#!/usr/bin/env python3
"""Merge benchmark outputs and generate the LaTeX tables, macros and figures of the paper.

    python scripts/make_paper_tables.py --results results --runs runs --out ../paper

Writes <out>/gen/*.tex and <out>/figs/*.pdf. Every number quoted in the
results section of the manuscript is produced here from the CSV files.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vnfplace import stats as S  # noqa: E402
from vnfplace.topology import build_infrastructure  # noqa: E402

RENAME = {"GNN-PPO": "GNN-MAPPO", "MLP-PPO": "MLP-MAPPO", "ILP-PerRequest": "ILP-PerRequest",
          "GCN-PPO": "GCN-MAPPO", "GATv2-PPO": "GATv2-MAPPO", "GNN-basicfeat": "GINE, basic features",
          "GCN-basicfeat-sep": "GCN, basic feat., separate actors", "GNN-resmask": "GINE, trained w/ resource mask",
          "GNN-PPO-zs": "GNN-MAPPO (zero-shot)", "MLP-PPO-zs": "MLP-MAPPO (zero-shot)"}
REF = "GNN-MAPPO"
ORDER = ["GNN-MAPPO", "MLP-MAPPO", "ILP-PerRequest", "Greedy-Latency", "Least-Loaded", "Greedy-Energy",
         "Best-Fit", "First-Fit", "Random"]
TOPO_LABEL = {"fog-12e6f-s1": "IoT fog (19 nodes)", "real-nobel_germany": "Nobel-Germany",
              "real-geant2012": "GÉANT", "real-janos_us": "Janos-US", "real-germany50": "Germany50",
              "real-uninett2011": "Uninett", "fog-40e12f-s1": "IoT fog (53 nodes)"}
TOPO_ORDER = list(TOPO_LABEL)
HIB = {"acceptance": True, "avg_power_w": False, "energy_per_accepted_j": False, "latency_mean_ms": False,
       "latency_p95_ms": False, "avg_active_nodes": False, "decision_ms_mean": False}


def load_dir(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["method"] = df["method"].replace(RENAME)
    return df


def load_requests(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["method"] = df["method"].replace(RENAME)
    return df


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    return S.bootstrap_ci(x, n_boot=2000)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:,.{d}f}"


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def order_methods(ms):
    ms = list(ms)
    known = [m for m in ORDER if m in ms]
    return known + sorted(m for m in ms if m not in known)


# ----------------------------------------------------------------------------- tables
def tab_topologies(out: Path):
    rows = []
    specs = [("fog", "train/test"), ("real:nobel_germany", "train/test"), ("real:geant2012", "train/test"),
             ("real:janos_us", "zero-shot"), ("real:germany50", "zero-shot"), ("real:uninett2011", "zero-shot"),
             ("fog:40x12", "zero-shot")]
    for spec, role in specs:
        s = build_infrastructure(spec, seed=1).summary()
        rows.append(f"{TOPO_LABEL[s['name']]} & {s['nodes']} & {s['links']} & {s['edge_nodes']}/{s['fog_nodes']}/{s['cloud_nodes']}"
                    f" & {s['edge_fog_vcpu']:.0f} & {s['mean_link_delay_ms']:.2f} & {s['diameter_delay_ms']:.1f} & {role} \\\\")
    body = "\n".join(rows)
    (out / "tab_topologies.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Substrates. Real backbones from SNDlib/Topology Zoo; every PoP hosts a fog site and one cloud DC is attached to the most central PoP. Delay figures include propagation, switching and serialisation.}
\label{tab:topologies}
\small
\begin{tabular}{lrrcrrrl}
\toprule
Topology & Nodes & Links & Edge/fog/cloud & Edge+fog vCPU & Mean link delay (ms) & Max path delay (ms) & Use \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_main(df: pd.DataFrame, out: Path, load: float, macros: dict):
    """One table: methods x topologies at a given load."""
    topos = [t for t in TOPO_ORDER if t in df.topology.unique()]
    lines = []
    for t in topos:
        d = df[(df.topology == t) & (np.isclose(df.load, load))]
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{8}}{{l}}{{\textit{{{TOPO_LABEL[t]}}}}} \\")
        best_pw_ok = None
        stats = {}
        for m in order_methods(d.method.unique()):
            g = d[d.method == m]
            stats[m] = {k: ci(g[k]) for k in ["acceptance", "avg_power_w", "energy_per_accepted_j",
                                               "latency_mean_ms", "latency_p95_ms", "avg_active_nodes",
                                               "decision_ms_mean"]}
            stats[m]["n"] = len(g)
        # bold: highest acceptance; lowest power among methods within 1 pp of the best acceptance
        acc_best = max(v["acceptance"][0] for v in stats.values())
        ok = [m for m, v in stats.items() if v["acceptance"][0] >= acc_best - 0.01]
        best_pw_ok = min(ok, key=lambda m: stats[m]["avg_power_w"][0])
        for m, v in stats.items():
            a = v["acceptance"]
            accs = f"{100 * a[0]:.1f} [{100 * a[1]:.1f}, {100 * a[2]:.1f}]"
            if a[0] >= acc_best - 1e-9:
                accs = r"\textbf{" + accs + "}"
            p = v["avg_power_w"]
            pws = f"{p[0]:,.0f} [{p[1]:,.0f}, {p[2]:,.0f}]"
            if m == best_pw_ok:
                pws = r"\textbf{" + pws + "}"
            name = tex_escape(m) + (r"$^\dagger$" if m == REF else "")
            lines.append(f"{name} & {accs} & {pws} & {v['energy_per_accepted_j'][0] / 1000:.2f} & "
                         f"{v['latency_mean_ms'][0]:.1f} & {v['latency_p95_ms'][0]:.1f} & "
                         f"{v['avg_active_nodes'][0]:.1f} & {v['decision_ms_mean'][0]:.1f} \\\\")
    tag = f"{load:g}".replace(".", "")
    (out / f"tab_main_load{tag}.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Main benchmark at offered load $\rho=""" + f"{load:g}" + r"""$ (shield on for all methods; 10 seeds $\times$ 1700 measured requests). Acceptance (\%) and average power (W) as mean [95\% bootstrap CI]. Bold: highest acceptance, and lowest power among methods within 1~pp of it. $^\dagger$proposed.}
\label{tab:main_load""" + tag + r"""}
\small
\begin{tabular}{lccrrrrr}
\toprule
Method & Acceptance (\%) & Avg.\ power (W) & $E$/acc.\ (kJ) & Lat.\ (ms) & P95 (ms) & Active nodes & Dec.\ (ms) \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_loads(df: pd.DataFrame, out: Path):
    """Compact acceptance / power across all loads for the three main topologies."""
    topos = [t for t in TOPO_ORDER if t in df.topology.unique()]
    loads = sorted(df.load.unique())
    lines = []
    for t in topos:
        d = df[df.topology == t]
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{{1 + 2 * len(loads)}}}{{l}}{{\textit{{{TOPO_LABEL[t]}}}}} \\")
        for m in order_methods(d.method.unique()):
            cells = []
            for ld in loads:
                g = d[(d.method == m) & np.isclose(d.load, ld)]
                cells += [f"{100 * g.acceptance.mean():.1f}", f"{g.avg_power_w.mean():,.0f}"]
            lines.append(tex_escape(m) + " & " + " & ".join(cells) + r" \\")
    head = " & ".join([rf"\multicolumn{{2}}{{c}}{{$\rho={ld:g}$}}" for ld in loads])
    cmid = " ".join([rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(loads))])
    sub = " & ".join(["Acc.\\,\\% & $\\bar P$ (W)"] * len(loads))
    (out / "tab_loads.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Acceptance ratio (\%) and average power (W) versus offered load; means over 10 seeds.}
\label{tab:loads}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{l""" + "rr" * len(loads) + r"""}
\toprule
 & """ + head + r""" \\
""" + cmid + "\n & " + sub + r""" \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def paired(df: pd.DataFrame, ref: str, others, metric: str):
    """Paired differences ref - other over (topology, load, seed) blocks."""
    rows = []
    a = df[df.method == ref].set_index(["topology", "load", "seed"])[metric]
    for o in others:
        b = df[df.method == o].set_index(["topology", "load", "seed"])[metric]
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if len(j) < 2:
            continue
        d = j.a - j.b
        try:
            p = S.st.wilcoxon(j.a, j.b).pvalue if (d != 0).any() else 1.0
        except ValueError:
            p = 1.0
        better = (d > 0) if HIB[metric] else (d < 0)
        rel = (d / j.b.replace(0, np.nan)).mean()
        rows.append(dict(other=o, n=len(j), mean_diff=d.mean(), rel=rel, p=p, wins=better.mean(),
                         cliff=S.cliffs_delta(j.a.values, j.b.values)))
    r = pd.DataFrame(rows)
    if len(r):
        from statsmodels.stats.multitest import multipletests
        r["p_holm"] = multipletests(r.p, method="holm")[1]
    return r


def tab_paired(df: pd.DataFrame, out: Path, macros: dict):
    others = [m for m in order_methods(df.method.unique()) if m != REF]
    acc = paired(df, REF, others, "acceptance").set_index("other")
    pw = paired(df, REF, others, "avg_power_w").set_index("other")
    epa = paired(df, REF, others, "energy_per_accepted_j").set_index("other")
    lines = []
    for o in others:
        if o not in acc.index:
            continue
        a, p, e = acc.loc[o], pw.loc[o], epa.loc[o]
        lines.append(f"{tex_escape(o)} & {int(a.n)} & {100 * a.mean_diff:+.2f} & {100 * a.wins:.0f} & {a.p_holm:.1e} & "
                     f"{100 * p.rel:+.1f} & {100 * p.wins:.0f} & {p.p_holm:.1e} & {p.cliff:+.2f} & "
                     f"{100 * e.rel:+.1f} & {100 * e.wins:.0f} \\\\")
    (out / "tab_paired.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Paired comparison of GNN-MAPPO with every other method over all (topology, load, seed) blocks of the main benchmark. $\Delta$: GNN-MAPPO minus the other method (acceptance in pp, power and energy per accepted request in \% of the other method). Wins: share of blocks in which GNN-MAPPO is better. $p$: Wilcoxon signed-rank, Holm-corrected per metric. $\delta$: Cliff's delta of the power (negative: GNN-MAPPO lower).}
\label{tab:paired}
\small
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & & \multicolumn{3}{c}{Acceptance} & \multicolumn{4}{c}{Average power} & \multicolumn{2}{c}{Energy / accepted} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}
Other method & Blocks & $\Delta$ (pp) & Wins \% & $p_{\mathrm{Holm}}$ & $\Delta$ (\%) & Wins \% & $p_{\mathrm{Holm}}$ & $\delta$ & $\Delta$ (\%) & Wins \% \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    return acc, pw, epa


def pareto_share(df: pd.DataFrame):
    """Share of blocks in which GNN-MAPPO is not Pareto-dominated (acceptance up, power down)."""
    res = {}
    for key, g in df.groupby(["topology", "load", "seed"]):
        r = g[g.method == REF]
        if r.empty:
            continue
        a0, p0 = r.acceptance.iloc[0], r.avg_power_w.iloc[0]
        others = g[g.method != REF]
        dom = ((others.acceptance >= a0) & (others.avg_power_w <= p0) &
               ((others.acceptance > a0) | (others.avg_power_w < p0))).any()
        res[key] = not dom
    return float(np.mean(list(res.values()))) if res else np.nan


def tab_ablation(df: pd.DataFrame, out: Path, label: str, caption: str, fname: str, ref=REF):
    loads = sorted(df.load.unique())
    ms = [ref] + [m for m in df.method.unique() if m != ref]
    lines = []
    for m in ms:
        cells = []
        for ld in loads:
            g = df[(df.method == m) & np.isclose(df.load, ld)]
            cells += [f"{100 * g.acceptance.mean():.1f}", f"{g.avg_power_w.mean():,.0f}"]
        if m != ref:
            pa = paired(df, ref, [m], "avg_power_w")
            pacc = paired(df, ref, [m], "acceptance")
            extra = (f"{-100 * pa.rel.iloc[0]:+.1f} & {100 * (1 - pa.wins.iloc[0]):.0f} & {pa.p.iloc[0]:.1e}"
                     f" & {-100 * pacc.mean_diff.iloc[0]:+.2f}") if len(pa) else "-- & -- & -- & --"
        else:
            extra = r"\multicolumn{4}{c}{reference}"
        lines.append(tex_escape(m) + " & " + " & ".join(cells) + " & " + extra + r" \\")
    head = " & ".join([rf"\multicolumn{{2}}{{c}}{{$\rho={ld:g}$}}" for ld in loads])
    sub = " & ".join(["Acc.\\,\\% & $\\bar P$ (W)"] * len(loads))
    (out / fname).write_text(r"""\begin{table*}[t]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "rr" * len(loads) + r"""rrrr}
\toprule
 & """ + head + r""" & \multicolumn{4}{c}{vs.\ reference (all loads)} \\
 & """ + sub + r""" & $\Delta\bar P$ \% & Lower \% & $p$ & $\Delta$acc pp \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_shield(on: pd.DataFrame, off: pd.DataFrame, out: Path):
    loads = sorted(set(off.load.unique()) & set(on.load.unique()))
    on = on[on.load.isin(loads) & on.seed.isin(off.seed.unique())]
    ms = order_methods(set(off.method.unique()) & set(on.method.unique()))
    lines = []
    for m in ms:
        a = off[off.method == m]
        b = on[on.method == m]
        de = a["rej_dead_end"].fillna(0).mean() if "rej_dead_end" in a else 0.0
        deb = b["rej_dead_end"].fillna(0).mean() if "rej_dead_end" in b else 0.0
        lines.append(f"{tex_escape(m)} & {100 * a.acceptance.mean():.1f} & {100 * b.acceptance.mean():.1f} & "
                     f"{100 * (b.acceptance.mean() - a.acceptance.mean()):+.1f} & {100 * de:.2f} & {100 * deb:.2f} & "
                     f"{a.avg_power_w.mean():,.0f} & {b.avg_power_w.mean():,.0f} & "
                     f"{a.decision_ms_mean.mean():.1f} & {b.decision_ms_mean.mean():.1f} \\\\")
    (out / "tab_shield.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Effect of the completion shield on the IoT fog network ($\rho\in\{""" + ",".join(f"{l:g}" for l in loads) + r"""\}$, 10 seeds): acceptance, dead-end rejection rate, average power and decision time without and with the shield. The learned policies are the same checkpoints in both columns.}
\label{tab:shield}
\small
\begin{tabular}{lrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{Acceptance (\%)} & \multicolumn{2}{c}{Dead ends (\%)} & \multicolumn{2}{c}{$\bar P$ (W)} & \multicolumn{2}{c}{Decision (ms)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Method & off & on & $\Delta$ & off & on & off & on & off & on \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_zeroshot(df: pd.DataFrame, indom: pd.DataFrame, out: Path):
    topos = [t for t in TOPO_ORDER if t in df.topology.unique()]
    lines = []
    for t in topos:
        d = df[df.topology == t]
        ms = order_methods(d.method.unique())
        extra = indom[(indom.topology == t) & indom.load.isin(d.load.unique()) & indom.method.isin(["GNN-MAPPO", "MLP-MAPPO"])]
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{5}}{{l}}{{\textit{{{TOPO_LABEL[t]}}}}} \\")
        for m, g in list(extra.groupby("method")) + [(m, d[d.method == m]) for m in ms]:
            label = tex_escape(m) + (" (trained here)" if m in ("GNN-MAPPO", "MLP-MAPPO") else "")
            lines.append(f"{label} & {100 * g.acceptance.mean():.1f} & {g.avg_power_w.mean():,.0f} & "
                         f"{g.energy_per_accepted_j.mean() / 1000:.2f} & {g.decision_ms_mean.mean():.1f} \\\\")
    (out / "tab_zeroshot.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Zero-shot transfer of policies trained only on the 19-node IoT fog network (means over 10 seeds and $\rho\in\{0.75,1,1.25\}$; shield on). For Nobel-Germany and GÉANT the policies trained on that topology are shown for comparison.}
\label{tab:zeroshot}
\small
\begin{tabular}{lrrrr}
\toprule
Method & Acc.\ (\%) & $\bar P$ (W) & $E$/acc.\ (kJ) & Dec.\ (ms) \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ----------------------------------------------------------------------------- figures
def style_map(methods):
    from vnfplace.plots import PALETTE, MARKERS, GREY
    ms = list(dict.fromkeys(methods))
    first = [m for m in ORDER if m in ms]
    rest = sorted(m for m in ms if m not in first)
    order = first + rest
    return {m: (PALETTE[i] if i < len(PALETTE) else GREY, MARKERS[i % len(MARKERS)]) for i, m in enumerate(order)}


def figures(df: pd.DataFrame, req: pd.DataFrame, figdir: Path, runs: Path, abl: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vnfplace.plots import _save, fig_cd
    figdir.mkdir(parents=True, exist_ok=True)
    sm = style_map(df.method.unique())
    topos = [t for t in TOPO_ORDER if t in df.topology.unique()]
    metrics = [("acceptance", "Acceptance ratio"), ("avg_power_w", "Average power (W)"),
               ("energy_per_accepted_j", "Energy / accepted req. (kJ)")]
    fig, axes = plt.subplots(len(metrics), len(topos), figsize=(3.0 * len(topos), 2.3 * len(metrics)), squeeze=False)
    for j, t in enumerate(topos):
        dt = df[df.topology == t]
        for i, (mt, lab) in enumerate(metrics):
            ax = axes[i][j]
            for m in order_methods(dt.method.unique()):
                g = dt[dt.method == m].groupby("load")[mt]
                mu, se = g.mean(), 1.96 * g.std() / np.sqrt(g.count())
                if mt == "energy_per_accepted_j":
                    mu, se = mu / 1000, se / 1000
                c, mk = sm[m]
                ax.errorbar(mu.index, mu.values, yerr=se.values, color=c, marker=mk, ms=4, lw=1.4, capsize=2, label=m)
            if i == 0:
                ax.set_title(TOPO_LABEL[t])
            if i == len(metrics) - 1:
                ax.set_xlabel(r"Offered load $\rho$")
            if j == 0:
                ax.set_ylabel(lab)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    _save(fig, figdir, "fig_vs_load")

    # Pareto at rho = 1.0
    fig, axes = plt.subplots(1, len(topos), figsize=(3.2 * len(topos), 2.8), squeeze=False)
    for j, t in enumerate(topos):
        ax = axes[0][j]
        d = df[(df.topology == t) & np.isclose(df.load, 1.0)]
        for m in order_methods(d.method.unique()):
            g = d[d.method == m]
            c, mk = sm[m]
            ax.errorbar(g.avg_power_w.mean(), g.acceptance.mean(), xerr=1.96 * g.avg_power_w.std() / np.sqrt(len(g)),
                        yerr=1.96 * g.acceptance.std() / np.sqrt(len(g)), color=c, marker=mk, ms=6, capsize=2,
                        linestyle="none", label=m)
        ax.set_title(TOPO_LABEL[t] + r", $\rho=1$")
        ax.set_xlabel("Average power (W)")
        if j == 0:
            ax.set_ylabel("Acceptance ratio")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    _save(fig, figdir, "fig_pareto")

    # Latency CDF (fog, rho = 1)
    if len(req):
        d = req[(req.topology == topos[0]) & np.isclose(req.load, 1.0) & req.accepted] if "topology" in req else req
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        for m in order_methods(d.method.unique()):
            v = np.sort((d[d.method == m].latency_ms / d[d.method == m].max_latency_ms).values)
            if len(v) == 0:
                continue
            c, mk = sm[m]
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, marker=mk, markevery=max(1, len(v) // 6), ms=4,
                    lw=1.3, label=m)
        ax.set_xlabel(r"Latency / budget $D(\mathbf{x}_r)/D_r$ (accepted requests)")
        ax.set_ylabel("CDF")
        ax.set_xlim(0, 1.0)
        ax.legend(frameon=False, fontsize=6.5)
        _save(fig, figdir, "fig_latency_cdf")

        # acceptance per class at rho = 1.25 (fog)
        d = req[(req.topology == topos[0]) & np.isclose(req.load, 1.25)] if "topology" in req else req
        tab = d.groupby(["method", "cls"]).accepted.mean().unstack()
        tab = tab.loc[[m for m in order_methods(tab.index)]]
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        w = 0.8 / len(tab)
        for i, m in enumerate(tab.index):
            c, mk = sm[m]
            ax.bar(np.arange(tab.shape[1]) + i * w, tab.loc[m].values, width=w, color=c, label=m)
        ax.set_xticks(np.arange(tab.shape[1]) + 0.4 - w / 2)
        ax.set_xticklabels([c.replace("_", " ") for c in tab.columns])
        ax.set_ylim(0.5, 1.01)
        ax.set_ylabel("Acceptance ratio")
        ax.legend(frameon=False, ncol=5, fontsize=6.5, loc="lower left")
        _save(fig, figdir, "fig_class_acceptance")

    # CD diagrams
    for mt in ["acceptance", "avg_power_w", "energy_per_accepted_j"]:
        fr = S.friedman_nemenyi(df[df.method != "ILP-PerRequest"], mt, HIB[mt])
        fr["metric"] = {"acceptance": "acceptance", "avg_power_w": "power",
                        "energy_per_accepted_j": "energy"}[mt]
        fig_cd(fr, figdir)

    # Training curves (fog variants)
    groups = {}
    for d in sorted(glob.glob(str(runs / "fog" / "*_s*"))):
        name = Path(d).name.rsplit("_s", 1)[0]
        lg = pd.read_csv(Path(d) / "train_log.csv").dropna(subset=["val_score"])
        groups.setdefault(name, []).append(lg)
    if groups:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))
        smr = style_map([RENAME.get(k, k) for k in groups])
        for name, logs in groups.items():
            lab = RENAME.get(name, name)
            n = min(len(l) for l in logs)
            steps = logs[0].steps.values[:n]
            c, mk = smr[lab]
            for ax, col, sc in zip(axes, ["val_acceptance", "val_power_w"], [1, 1]):
                v = np.stack([l[col].values[:n] for l in logs]) * sc
                ax.plot(steps, v.mean(0), color=c, marker=mk, markevery=max(1, n // 5), ms=3.5, lw=1.2, label=lab)
                ax.fill_between(steps, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=c, alpha=0.12, lw=0)
            lam = np.stack([l["lambda"].values for l in [pd.read_csv(Path(d) / "train_log.csv")
                                                          for d in sorted(glob.glob(str(runs / "fog" / f"{name}_s*")))]])
            st_all = pd.read_csv(Path(sorted(glob.glob(str(runs / "fog" / f"{name}_s*")))[0]) / "train_log.csv").steps.values
            axes[2].plot(st_all, lam.mean(0), color=c, lw=1.2, marker=mk, markevery=max(1, len(st_all) // 5), ms=3.5)
        axes[0].axhline(0.98, color="#52514e", lw=0.8, ls="--")
        axes[0].set_ylabel("Validation acceptance")
        axes[1].set_ylabel("Validation power (W)")
        axes[2].set_ylabel(r"Lagrange multiplier $\lambda$")
        for ax in axes:
            ax.set_xlabel("Environment steps")
        axes[0].set_ylim(0.9, 1.0)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12), fontsize=7)
        fig.tight_layout()
        _save(fig, figdir, "fig_training")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    gen, figdir = out / "gen", out / "figs"
    gen.mkdir(parents=True, exist_ok=True)
    macros = {}

    e1 = load_dir(str(R / "E1_*" / "*_s*" / "summary.csv"))
    e1.to_csv(R / "E1_merged_summary.csv", index=False)
    req = load_requests(str(R / "E1_fog" / "*_s*" / "requests.csv.gz"))
    if len(req):
        req["topology"] = "fog-12e6f-s1"
    tab_topologies(gen)
    for ld in [1.0, 1.25]:
        tab_main(e1, gen, ld, macros)
    tab_loads(e1, gen)
    acc, pw, epa = tab_paired(e1, gen, macros)

    # Headline numbers
    def rel(t, other, metric="avg_power_w", loads=None):
        d = e1[e1.topology == t] if t else e1
        if loads is not None:
            d = d[d.load.isin(loads)]
        r = paired(d, REF, [other], metric)
        return r.iloc[0] if len(r) else None

    M = {}
    for t, key in [("fog-12e6f-s1", "Fog"), ("real-nobel_germany", "Nobel"), ("real-geant2012", "Geant")]:
        for o, ok in [("Greedy-Latency", "GL"), ("Least-Loaded", "LL"), ("MLP-MAPPO", "MLP"), ("ILP-PerRequest", "ILP"),
                      ("Greedy-Energy", "GE")]:
            r = rel(t, o)
            ra = rel(t, o, "acceptance")
            re_ = rel(t, o, "energy_per_accepted_j")
            if r is None:
                continue
            M[f"res{key}{ok}Pow"] = f"{-100 * r.rel:.1f}"
            M[f"res{key}{ok}Acc"] = f"{100 * ra.mean_diff:+.2f}"
            M[f"res{key}{ok}Epa"] = f"{-100 * re_.rel:.1f}"
            M[f"res{key}{ok}Wins"] = f"{100 * r.wins:.0f}"
        d = e1[e1.topology == t]
        for m, mk in [(REF, "Ours"), ("Greedy-Latency", "GL"), ("MLP-MAPPO", "MLP"), ("ILP-PerRequest", "ILP"),
                      ("Greedy-Energy", "GE"), ("Least-Loaded", "LL"), ("Best-Fit", "BF")]:
            g = d[d.method == m]
            if len(g):
                M[f"res{key}{mk}AccAll"] = f"{100 * g.acceptance.mean():.1f}"
                M[f"res{key}{mk}PowAll"] = f"{g.avg_power_w.mean():,.0f}"
                M[f"res{key}{mk}Dec"] = f"{g.decision_ms_mean.mean():.1f}"
    for o, ok in [("Greedy-Latency", "GL"), ("MLP-MAPPO", "MLP"), ("Least-Loaded", "LL"), ("ILP-PerRequest", "ILP")]:
        if o in pw.index:
            M[f"resAll{ok}Pow"] = f"{-100 * pw.loc[o].rel:.1f}"
            M[f"resAll{ok}PowWins"] = f"{100 * pw.loc[o].wins:.0f}"
            M[f"resAll{ok}PowP"] = f"{pw.loc[o].p_holm:.1e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + ("}" if "e-" in f"{pw.loc[o].p_holm:.1e}" else "")
            M[f"resAll{ok}Acc"] = f"{100 * acc.loc[o].mean_diff:+.2f}"
            M[f"resAll{ok}Epa"] = f"{-100 * epa.loc[o].rel:.1f}"
    M["resParetoShare"] = f"{100 * pareto_share(e1):.0f}"
    M["resNumEvalRequests"] = f"{int(e1.requests.sum()):,}".replace(",", "{,}")
    mins = []
    for d in glob.glob(str(Path(a.runs) / "*" / "GNN-PPO_s*")):
        mins.append(pd.read_csv(Path(d) / "train_log.csv").time_s.iloc[-1] / 60)
    M["resTrainMinutes"] = f"{np.median(mins):.0f}" if mins else "--"
    M["resNumTrainRuns"] = str(len(glob.glob(str(Path(a.runs) / "*" / "*_s*"))))

    # Ablation / shield / zero-shot
    abl = load_dir(str(R / "E5_ablation" / "s*" / "summary.csv"))
    if len(abl):
        tab_ablation(abl, gen, "tab:ablation",
                     r"Ablation on the IoT fog network (shield on, 10 seeds). $\Delta\bar P$: power of the variant relative to GNN-MAPPO, paired over loads and seeds (positive: the variant draws more power); Lower \%: blocks in which the variant draws less power; $p$: Wilcoxon (uncorrected); $\Delta$acc: variant minus GNN-MAPPO.",
                     "tab_ablation.tex")
        for m in abl.method.unique():
            if m == REF:
                continue
            r = paired(abl, REF, [m], "avg_power_w")
            if len(r):
                key = "".join(ch for ch in m if ch.isalnum())
                M[f"resAbl{key}Pow"] = f"{-100 * r.rel.iloc[0]:+.1f}"
                M[f"resAbl{key}P"] = f"{r.p.iloc[0]:.2g}"
    off = load_dir(str(R / "E5_noshield" / "s*" / "summary.csv"))
    if len(off):
        on = e1[(e1.topology == "fog-12e6f-s1")]
        tab_shield(on, off, gen)
        for m, mk in [(REF, "Ours"), ("Greedy-Latency", "GL"), ("Greedy-Energy", "GE"), ("Best-Fit", "BF"), ("MLP-MAPPO", "MLP")]:
            a_, b_ = off[off.method == m], on[(on.method == m) & on.load.isin(off.load.unique())]
            if len(a_) and len(b_):
                M[f"resShield{mk}Gain"] = f"{100 * (b_.acceptance.mean() - a_.acceptance.mean()):.1f}"
                M[f"resShield{mk}DeadOff"] = f"{100 * a_['rej_dead_end'].fillna(0).mean():.2f}"
    rm = load_dir(str(R / "E5_resmask" / "s*" / "summary.csv"))
    if len(rm):
        for m in rm.method.unique():
            g = rm[rm.method == m]
            key = "".join(ch for ch in m if ch.isalnum())
            M[f"resResmask{key}Acc"] = f"{100 * g.acceptance.mean():.1f}"
            M[f"resResmask{key}Pow"] = f"{g.avg_power_w.mean():,.0f}"
    zs = load_dir(str(R / "E4_*" / "s*" / "summary.csv"))
    if len(zs):
        tab_zeroshot(zs, e1, gen)
        for t in zs.topology.unique():
            d = zs[zs.topology == t]
            key = {"fog-12e6f-s1": "FogSmall", "real-nobel_germany": "Nobel", "real-geant2012": "Geant",
                   "real-janos_us": "Janos", "real-germany50": "Germany", "real-uninett2011": "Uninett",
                   "fog-40e12f-s1": "FogLarge"}[t]
            r = paired(d, "GNN-MAPPO (zero-shot)", ["Greedy-Latency"], "avg_power_w")
            ra = paired(d, "GNN-MAPPO (zero-shot)", ["Greedy-Latency"], "acceptance")
            if len(r):
                M[f"resZS{key}Pow"] = f"{-100 * r.rel.iloc[0]:.1f}"
                M[f"resZS{key}Acc"] = f"{100 * ra.mean_diff.iloc[0]:+.2f}"
        zs.to_csv(R / "E4_merged_summary.csv", index=False)

    # Abstract sentence (filled with the headline numbers)
    M["resAbstract"] = (
        "On the IoT fog network and the Nobel-Germany backbone, GNN-MAPPO keeps the acceptance ratio within "
        + M["resFogGLAcc"].lstrip("+-") + " and " + M["resNobelGLAcc"].lstrip("+-") + " percentage points of the "
        "latency-greedy heuristic, the best acceptance baseline, while drawing " + M["resFogGLPow"] + r"\% and "
        + M["resNobelGLPow"] + r"\% less average power (lower in " + M["resFogGLWins"] + r"\% and "
        + M["resNobelGLWins"] + r"\% of the paired blocks); the heuristics and the per-request optimum that draw "
        "less power reject 3--9 percentage points more requests. On the larger GÉANT backbone, where tight-budget "
        "requests can only be served near their sources, a class-aware capacity reservation applied to every method "
        r"keeps the acceptance within \resPairGLAccAbs{} percentage points of the latency-greedy heuristic (without "
        "reservation: " + M["resGeantGLAcc"].lstrip("-") + r") at \resPairGLPow\% lower power. "
        "Ablations show that the constrained objective, parameter sharing across chain positions and the feasibility "
        "layers matter more than the graph encoder, whose advantage over a node-wise MLP is not significant, and "
        "zero-shot transfer to unseen backbones retains the power saving but loses acceptance.")

    for f in gen.glob("tab_*.tex"):
        t = f.read_text()
        if "adjustbox" not in t:
            t = t.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}")
            t = t.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
            f.write_text(t)
    M = {"".join(ch for ch in k if ch.isalpha()): v for k, v in M.items()}
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (gen / "macros.tex").write_text("% generated by scripts/make_paper_tables.py\n" + "\n".join(lines) + "\n")
    (gen / "macros.json").write_text(json.dumps(M, indent=1))
    figures(e1, req, figdir, Path(a.runs), abl)
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
