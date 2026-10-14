#!/usr/bin/env python3
"""Tables and macros for the capacity-reservation experiment (E6, GÉANT).

    python scripts/make_reserve_tables.py --out <paper_dir>

Reads results/E1_real_geant2012 (R = 0), results/C_pilot (R sweep with the
original checkpoints) and results/E6_reserve_real_geant2012 (R = 0.3, policies
fine-tuned with reservation). Writes gen/tab_reserve_sweep.tex,
gen/tab_reserve.tex and gen/macros_reserve.tex.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
REN = {"GNN-PPO": "GNN-MAPPO", "MLP-PPO": "MLP-MAPPO", "GNN-MAPPO-C": "GNN-MAPPO-R"}


def load(pattern, **extra):
    fs = glob.glob(str(ROOT / pattern))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for k, v in extra.items():
        d[k] = v
    return d


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(run, 1.0)
    return adj


def sci(p):
    if p >= 0.01:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def fmt_int(x):
    return f"{x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen = Path(a.out) / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    M = {}

    # ---------------------------------------------------------------- sweep
    e1 = load("results/E1_real_geant2012/main_s*/summary.csv", R=0.0)
    sw = [e1]
    for f in glob.glob(str(ROOT / "results/C_pilot/r*_s*")):
        r = float(Path(f).name.split("_")[0][1:])
        d = pd.read_csv(Path(f) / "summary.csv")
        d["R"] = r
        sw.append(d)
    sw = pd.concat(sw, ignore_index=True)
    sw = sw[sw.load.isin([0.75, 1.0, 1.25]) & sw.seed.isin(range(5))
            & sw.method.isin(["GNN-PPO", "Greedy-Latency", "Greedy-Energy"])]
    g = sw.groupby(["R", "method"])[["acceptance", "avg_power_w", "acc_industrial_control", "acc_ar_offload"]].mean()
    rows = []
    for R in sorted(sw.R.unique()):
        cells = [f"{R:.1f}"]
        for m in ["GNN-PPO", "Greedy-Energy", "Greedy-Latency"]:
            r = g.loc[(R, m)]
            cells += [f"{100 * r.acceptance:.1f}", fmt_int(r.avg_power_w)]
        rows.append(" & ".join(cells) + r" \\")
    tab = r"""\begin{table}[t]
\centering
\caption{Capacity reservation on GÉANT with the original checkpoints (no retraining): acceptance (\%) and average power (W) versus the reserved CPU share $R$ (means over seeds 0--4 and $\rho\in\{0.75,1,1.25\}$; shield on; every method uses the same $R$).}
\label{tab:reserve_sweep}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{2}{c}{GNN-MAPPO} & \multicolumn{2}{c}{Greedy-Energy} & \multicolumn{2}{c}{Greedy-Latency} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
$R$ & Acc. & $\bar P$ & Acc. & $\bar P$ & Acc. & $\bar P$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
"""
    (gen / "tab_reserve_sweep.tex").write_text(tab)
    for R, key in [(0.0, "Zero"), (0.3, "Three"), (0.5, "Five")]:
        for m, mk in [("GNN-PPO", "Ours"), ("Greedy-Energy", "GE"), ("Greedy-Latency", "GL")]:
            r = g.loc[(R, m)]
            M[f"resSw{key}{mk}Acc"] = f"{100 * r.acceptance:.1f}"
            M[f"resSw{key}{mk}Pow"] = fmt_int(r.avg_power_w)
            M[f"resSw{key}{mk}IC"] = f"{100 * r.acc_industrial_control:.1f}"

    # --------------------------------------------------------------- E6 main
    e6 = load("results/E6_reserve_real_geant2012/r0.3_s*/summary.csv")
    base = e1.copy()
    order = ["GNN-MAPPO-C", "GNN-PPO", "MLP-PPO", "Greedy-Latency", "Least-Loaded",
             "Greedy-Energy", "Best-Fit", "First-Fit", "Random"]
    cols = ["acceptance", "avg_power_w", "energy_per_accepted_j", "latency_p95_ms",
            "acc_industrial_control", "acc_ar_offload"]
    G6 = e6.groupby("method")[cols].mean()
    G0 = base.groupby("method")[cols].mean()
    rows = []
    for m in order:
        r = G6.loc[m]
        name = {"GNN-MAPPO-C": r"GNN-MAPPO-R$^\dagger$ (fine-tuned)",
                "GNN-PPO": "GNN-MAPPO (original)", "MLP-PPO": "MLP-MAPPO (original)"}.get(m, m)
        if m == "GNN-MAPPO-C":
            r0 = G0.loc["GNN-PPO"]
        else:
            r0 = G0.loc[m]
        rows.append(" & ".join([name, f"{100 * r0.acceptance:.1f}", fmt_int(r0.avg_power_w),
                                f"{100 * r.acceptance:.1f}", fmt_int(r.avg_power_w),
                                f"{r.energy_per_accepted_j / 1e3:.2f}",
                                f"{100 * r.acc_industrial_control:.1f}", f"{100 * r.acc_ar_offload:.1f}"]) + r" \\")
    tab = r"""\begin{table*}[t]
\centering
\caption{GÉANT without ($R=0$) and with capacity reservation ($R=0.3$, identical for every method): means over 10 seeds and $\rho\in\{0.5,\dots,1.5\}$, shield on. GNN-MAPPO-R is GNN-MAPPO fine-tuned for $3\times10^5$ steps with $R=0.3$; its $R=0$ columns repeat the original policy. IC, AR: acceptance of the industrial-control and AR-offload classes. $^\dagger$proposed.}
\label{tab:reserve}
\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{2}{c}{$R=0$} & \multicolumn{5}{c}{$R=0.3$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-8}
Method & Acc.\ (\%) & $\bar P$ (W) & Acc.\ (\%) & $\bar P$ (W) & $E$/acc.\ (kJ) & IC (\%) & AR (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""
    (gen / "tab_reserve.tex").write_text(tab)

    for m, mk in [("GNN-MAPPO-C", "OursR"), ("GNN-PPO", "OursOrig"), ("Greedy-Latency", "GL"),
                  ("Greedy-Energy", "GE"), ("Least-Loaded", "LL"), ("Best-Fit", "BF")]:
        r = G6.loc[m]
        M[f"resRes{mk}Acc"] = f"{100 * r.acceptance:.1f}"
        M[f"resRes{mk}Pow"] = fmt_int(r.avg_power_w)
        M[f"resRes{mk}IC"] = f"{100 * r.acc_industrial_control:.1f}"
        M[f"resRes{mk}AR"] = f"{100 * r.acc_ar_offload:.1f}"
    for m, mk in [("GNN-PPO", "OursOrig"), ("Greedy-Latency", "GL"), ("Greedy-Energy", "GE")]:
        r = G0.loc[m]
        M[f"resResZero{mk}Acc"] = f"{100 * r.acceptance:.1f}"
        M[f"resResZero{mk}Pow"] = fmt_int(r.avg_power_w)
        M[f"resResZero{mk}IC"] = f"{100 * r.acc_industrial_control:.1f}"
        M[f"resResZero{mk}AR"] = f"{100 * r.acc_ar_offload:.1f}"

    # ------------------------------------------------------- paired tests
    pv = e6.pivot_table(index=["load", "seed"], columns="method", values=["acceptance", "avg_power_w"])
    others = [("Greedy-Latency", "GL"), ("Least-Loaded", "LL"), ("Greedy-Energy", "GE"), ("GNN-PPO", "OursOrig")]
    res, ps = [], []
    for o, ok in others:
        da = 100 * (pv.acceptance["GNN-MAPPO-C"] - pv.acceptance[o])
        dp = 100 * (pv.avg_power_w["GNN-MAPPO-C"] / pv.avg_power_w[o] - 1)
        pa, pp = wilcoxon(da).pvalue, wilcoxon(dp).pvalue
        res.append((ok, da, dp))
        ps += [pa, pp]
    adj = holm(ps)
    for i, (ok, da, dp) in enumerate(res):
        M[f"resPair{ok}Acc"] = f"{da.mean():+.2f}"
        M[f"resPair{ok}AccAbs"] = f"{abs(da.mean()):.2f}"
        M[f"resPair{ok}AccP"] = sci(adj[2 * i])
        M[f"resPair{ok}Pow"] = f"{-dp.mean():.1f}"
        M[f"resPair{ok}PowP"] = sci(adj[2 * i + 1])
        M[f"resPair{ok}PowMore"] = f"{dp.mean():.1f}"
        M[f"resPair{ok}PowWins"] = f"{100 * (dp < 0).mean():.0f}"
        M[f"resPair{ok}Dom"] = f"{100 * ((da >= 0) & (dp <= 0)).mean():.0f}"
    # Gap to Greedy-Latency before reservation (all 10 seeds, all loads).
    pv0 = e1.pivot_table(index=["load", "seed"], columns="method", values=["acceptance", "avg_power_w"])
    M["resPairZeroGLAcc"] = f"{100 * (pv0.acceptance['GNN-PPO'] - pv0.acceptance['Greedy-Latency']).mean():+.2f}"
    M["resPairZeroGLPow"] = f"{-100 * (pv0.avg_power_w['GNN-PPO'] / pv0.avg_power_w['Greedy-Latency'] - 1).mean():.1f}"

    with open(gen / "macros_reserve.tex", "w") as f:
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in M.items():
        print(k, v)


if __name__ == "__main__":
    main()
