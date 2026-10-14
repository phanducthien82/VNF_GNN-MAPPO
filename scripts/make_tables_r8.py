#!/usr/bin/env python3
"""Tables, figures and LaTeX macros of the R8 manuscript (protected protocol).

Reads results_v3/E*/.../summary.csv and writes gen_r8/*.tex and gen_r8/figs/*.pdf.
All statistics are seed-level: a seed's value is its mean over the offered loads,
so that the ten training seeds are the independent replicates; paired tests are
Wilcoxon signed-rank tests over seeds with Holm correction within each table.

    python scripts/make_tables_r8.py --out ../paper_v3/gen_r8
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from frontier import gaps  # noqa: E402

R = ROOT / "results_v3"
TOPO = {"fog-12e6f-s1": "IoT fog", "real-nobel_germany": "Nobel-Germany", "real-geant2012": "GÉANT"}
TOPO_KEY = {"fog-12e6f-s1": "Fog", "real-nobel_germany": "Nobel", "real-geant2012": "Geant"}
RENAME = {"GNN-PPO": "GNN-MAPPO", "MLP-PPO": "MLP-MAPPO", "DQN": "DQN-Lagrangian"}
LEARNED = ["GNN-MAPPO-E", "GNN-MAPPO", "MLP-MAPPO-E", "MLP-MAPPO", "DQN-Lagrangian-E", "DQN-Lagrangian"]
SHORT = {"GNN-MAPPO": "GNN", "MLP-MAPPO": "MLP", "DQN-Lagrangian": "DQN", "GNN-MAPPO-E": "GNNE", "MLP-MAPPO-E": "MLPE",
         "DQN-Lagrangian-E": "DQNE"}
HEUR = ["Greedy-Latency", "Greedy-Energy", "Least-Loaded", "Best-Fit"]
MIX = ["Greedy-Mix:0.1", "Greedy-Mix:0.3", "Greedy-Mix:1", "Greedy-Mix:3"]
ORDER = LEARNED + HEUR + MIX + ["ILP-PerRequest"]
LABEL = {m: m for m in ORDER}
LABEL.update({"GNN-MAPPO-E": "GNN-MAPPO, 10-seed ensemble", "MLP-MAPPO-E": "MLP-MAPPO, 10-seed ensemble",
              "GNN-MAPPO": "GNN-MAPPO, single seed", "MLP-MAPPO": "MLP-MAPPO, single seed",
              "DQN-Lagrangian-E": "DQN-Lagrangian, ensemble", "DQN-Lagrangian": "DQN-Lagrangian, single seed"})
LABEL.update({f"Greedy-Mix:{w}": f"Greedy-Mix ($w={w}$)" for w in ["0.1", "0.3", "1", "3"]})
MACROS: dict = {}


DIGITS = dict(zip("0123456789.", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", ""]))


def mac(name: str, value: str):
    """LaTeX macro names may contain letters only: digits are spelled out, other characters dropped."""
    name = "".join(DIGITS.get(c, c) for c in name if c.isalnum() or c == ".")
    MACROS[name] = value


def load(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(R / pattern)))
    if not fs:
        return pd.DataFrame()
    parts = []
    for f in fs:
        d = pd.read_csv(f)
        ba = Path(f).parent / "bench_args.json"
        if ba.exists():
            d["reserve"] = json.loads(ba.read_text()).get("reserve", 0.0)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["method"] = df["method"].replace(RENAME)
    return df


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        v = float(x[0]) if len(x) else np.nan
        return v, v, v
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def seed_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """[topology, method, seed] -> mean of ``metric`` over loads."""
    return df.groupby(["topology", "method", "seed"])[metric].mean().reset_index()


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def wil(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    t = f"{v:,.{d}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def fmtp(p):
    if p is None or np.isnan(p):
        return "--"
    return f"{p:.3f}" if p >= 0.001 else "$<$0.001"


def pm(m, lo, hi, d=1):
    return f"{fmt(m, d)} {{\\scriptsize [{fmt(lo, d)}, {fmt(hi, d)}]}}"


# ----------------------------------------------------------------------------- main table
def tab_main(e1: pd.DataFrame, out: Path):
    cols = [("acceptance", 100, 2, "Acc. (\\%)"), ("avg_power_w", 1, 0, "Power (W)"),
            ("energy_per_accepted_j", 1e-3, 2, "$E/A$ (kJ)"), ("latency_p95_ms", 1, 1, "p95 lat. (ms)"),
            ("avg_active_nodes", 1, 1, "Active nodes"), ("decision_ms_mean", 1, 1, "Dec. (ms)")]
    lines = []
    for topo in TOPO:
        d = e1[e1.topology == topo]
        if d.empty:
            continue
        lines.append(f"\\midrule\\multicolumn{{7}}{{l}}{{\\textit{{{TOPO[topo]}}}}}\\\\")
        for m in [x for x in ORDER if x in set(d.method)]:
            cells = []
            for c, sc, dg, _ in cols:
                sm = seed_means(d[d.method == m], c)[c] * sc
                mu, lo, hi = boot_ci(sm)
                short = {"acceptance": "Acc", "avg_power_w": "Pow", "energy_per_accepted_j": "Epa",
                         "latency_p95_ms": "Lat", "avg_active_nodes": "Act", "decision_ms_mean": "Dec"}[c]
                mac(f"res{TOPO_KEY[topo]}{m}{short}", fmt(mu, dg))
                cells.append(f"{fmt(mu, dg)}" + (f"{{\\scriptsize$\\pm${fmt((hi - lo) / 2, dg)}}}" if c in
                                                ("acceptance", "avg_power_w") else ""))
            lines.append(f"{LABEL[m]} & " + " & ".join(cells) + "\\\\")
    head = " & ".join(h for *_, h in cols)
    (out / "tab_main.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Main benchmark with protection of running services and the "
        "completion shield for every method. Mean over ten seeds of the per-seed average over the five offered "
        "loads; $\\pm$ is the half-width of the 95\\% bootstrap confidence interval over seeds. Dec.: mean decision "
        "time per request during the benchmark runs, which shared the machine; idle single-core times are given in "
        "Section~\\ref{subsec:runtime}.}\n"
        "\\label{tab:main}\n\\small\n\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & " + head + "\\\\\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ----------------------------------------------------------------------------- frontier and paired table
def tab_frontier(e1: pd.DataFrame, out: Path):
    learned = [m for m in LEARNED if m in set(e1.method)]
    gp = gaps(e1, learned)
    gp.to_csv(out / "frontier_blocks.csv", index=False)
    rows, pvals = [], []
    for topo in TOPO:
        for m in learned:
            g = gp[(gp.topology == topo) & (gp.method == m)]
            if g.empty:
                continue
            # Unit of replication: the training seed for single policies; for an ensemble (one fixed
            # policy) every (load, seed) block is an independent test trace.
            unit = ["load", "seed"] if m.endswith("-E") else ["seed"]
            ps = g.groupby(unit).gap.mean() * 100
            mu, lo, hi = boot_ci(ps)
            rec = dict(topo=topo, m=m, gap=(mu, lo, hi), p_gap=wil(ps.values), better=100 * (g.gap < 0).mean(),
                       n=len(ps), unit="block" if len(unit) == 2 else "seed")
            for ref, tag in (("Greedy-Latency", "GL"), ("Greedy-Energy", "GE")):
                d = e1[(e1.topology == topo) & (e1.method.isin([m, ref]))]
                a = d.groupby(unit + ["method"]).acceptance.mean().unstack().dropna()
                p = d.groupby(unit + ["method"]).avg_power_w.mean().unstack().dropna()
                da = 100 * (a[m] - a[ref])
                dp = 100 * (p[m] - p[ref]) / p[ref]
                rec[f"dacc_{tag}"] = boot_ci(da)
                rec[f"dpow_{tag}"] = boot_ci(dp)
                rec[f"p_acc_{tag}"] = wil(da.values)
                rec[f"p_pow_{tag}"] = wil(dp.values)
            rows.append(rec)
    # Holm correction within each family of tests (one family per column group of the table).
    for fam in (["p_gap"], ["p_acc_GL", "p_pow_GL"], ["p_acc_GE", "p_pow_GE"]):
        flat = [r[k] for r in rows for k in fam]
        ok = [i for i, v in enumerate(flat) if not np.isnan(v)]
        adj = np.full(len(flat), np.nan)
        if ok:
            adj[ok] = holm([flat[i] for i in ok])
        for i, r in enumerate(rows):
            for j, k in enumerate(fam):
                r[k + "_h"] = adj[i * len(fam) + j]
    lines, last = [], None
    for r in rows:
        if r["topo"] != last:
            lines.append(f"\\midrule\\multicolumn{{8}}{{l}}{{\\textit{{{TOPO[r['topo']]}}}}}\\\\")
            last = r["topo"]
        g = r["gap"]
        lines.append(f"{LABEL[r['m']]} & {pm(*g, 1)} & {fmtp(r['p_gap_h'])} & {r['better']:.0f} & "
                     f"{fmt(r['dacc_GL'][0], 2)} & {fmt(r['dpow_GL'][0], 1)} & "
                     f"{fmt(r['dacc_GE'][0], 2)} & {fmt(r['dpow_GE'][0], 1)}\\\\")
        tk, mk = TOPO_KEY[r["topo"]], SHORT[r["m"]]
        mac(f"res{tk}{mk}Gap", fmt(g[0], 1))
        mac(f"res{tk}{mk}GapAbs", fmt(abs(g[0]), 1))
        mac(f"res{tk}{mk}GapLo", fmt(g[1], 1))
        mac(f"res{tk}{mk}GapHi", fmt(g[2], 1))
        mac(f"res{tk}{mk}GapP", fmtp(r["p_gap_h"]))
        mac(f"res{tk}{mk}GapBetter", f"{r['better']:.0f}")
        mac(f"res{tk}{mk}N", str(r["n"]))
        for tag in ("GL", "GE"):
            mac(f"res{tk}{mk}Acc{tag}Lo", fmt(r[f"dacc_{tag}"][1], 2))
            mac(f"res{tk}{mk}Acc{tag}Hi", fmt(r[f"dacc_{tag}"][2], 2))
            mac(f"res{tk}{mk}Pow{tag}Lo", fmt(r[f"dpow_{tag}"][1], 1))
            mac(f"res{tk}{mk}Pow{tag}Hi", fmt(r[f"dpow_{tag}"][2], 1))
            mac(f"res{tk}{mk}Acc{tag}", fmt(r[f"dacc_{tag}"][0], 2))
            mac(f"res{tk}{mk}AccAbs{tag}", fmt(abs(r[f"dacc_{tag}"][0]), 2))
            mac(f"res{tk}{mk}Pow{tag}", fmt(r[f"dpow_{tag}"][0], 1))
            mac(f"res{tk}{mk}PowAbs{tag}", fmt(abs(r[f"dpow_{tag}"][0]), 1))
            mac(f"res{tk}{mk}Acc{tag}P", fmtp(r[f"p_acc_{tag}_h"]))
            mac(f"res{tk}{mk}Pow{tag}P", fmtp(r[f"p_pow_{tag}_h"]))
    (out / "tab_frontier.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Learned policies against the tuned heuristic frontier and "
        "against the two greedy rules. \\emph{Gap}: power relative to the Pareto front of Greedy-Mix, "
        "Greedy-Energy and Greedy-Latency at the same acceptance, mean with 95\\% bootstrap CI over the units of "
        "replication (the ten training seeds for single policies, the 50 independent (load, seed) test traces "
        "for the ensembles); $p$: Wilcoxon signed-rank test over the same units, Holm-corrected over the column; "
        "\\emph{Better}: share of (load, seed) blocks with a negative gap. DQN-Lagrangian is trained on the IoT "
        "fog network only. $\\Delta$Acc in percentage points and $\\Delta P$ "
        "in percent, learned method minus reference.}\n\\label{tab:frontier}\n\\small\n"
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Tuned frontier} & "
        "\\multicolumn{2}{c}{vs Greedy-Latency} & \\multicolumn{2}{c}{vs Greedy-Energy}\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Method & Gap (\\%) & $p$ & Better (\\%) & $\\Delta$Acc & $\\Delta P$ & $\\Delta$Acc & $\\Delta P$\\\\\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    # Paired comparisons between ensembles on the same test traces (block level), Holm within this family.
    fam = []
    for a, b, tag in (("GNN-MAPPO-E", "MLP-MAPPO-E", "GNNEvsMLPE"), ("GNN-MAPPO-E", "DQN-Lagrangian-E", "GNNEvsDQNE")):
        for topo in TOPO:
            x = gp[(gp.topology == topo) & (gp.method == a)].set_index(["load", "seed"])
            y = gp[(gp.topology == topo) & (gp.method == b)].set_index(["load", "seed"])
            d = (100 * (x.gap - y.gap)).dropna()
            if len(d) >= 5:
                k = f"res{TOPO_KEY[topo]}{tag}"
                mac(k, fmt(d.mean(), 1))
                mac(k + "Abs", fmt(abs(d.mean()), 1))
                mac(k + "Better", f"{100 * (d < 0).mean():.0f}")
                mac(k + "N", str(len(d)))
                if "acceptance" in x.columns:
                    da = (100 * (x.acceptance - y.acceptance)).dropna()
                    mac(k + "Acc", fmt(da.mean(), 2))
                fam.append((k + "P", wil(d.values)))
    if fam:
        for (k, _), q in zip(fam, holm(np.nan_to_num(np.array([v for _, v in fam]), nan=1.0))):
            mac(k, fmtp(q))
    return gp


# ----------------------------------------------------------------------------- per-load table
def tab_loads(e1: pd.DataFrame, out: Path):
    ms = [m for m in ["GNN-MAPPO-E", "GNN-MAPPO", "DQN-Lagrangian", "Greedy-Latency", "Greedy-Energy",
                      "Greedy-Mix:0.3", "Greedy-Mix:1"] if m in set(e1.method)]
    loads = sorted(e1.load.unique())
    lines = []
    for topo in TOPO:
        d = e1[e1.topology == topo]
        if d.empty:
            continue
        lines.append(f"\\midrule\\multicolumn{{{1 + 2 * len(loads)}}}{{l}}{{\\textit{{{TOPO[topo]}}}}}\\\\")
        for m in ms:
            if d[d.method == m].empty:
                continue
            g = d[d.method == m].groupby("load")[["acceptance", "avg_power_w"]].mean()
            cells = [f"{100 * g.acceptance.get(l, np.nan):.1f}" for l in loads] + \
                    [f"{g.avg_power_w.get(l, np.nan):,.0f}" for l in loads]
            lines.append(f"{LABEL[m]} & " + " & ".join(cells) + "\\\\")
    hl = " & ".join(f"$\\rho={l:g}$" for l in loads)
    (out / "tab_loads.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Acceptance (\\%) and average power (W) per offered load "
        "(mean over ten seeds).}\n\\label{tab:loads}\n\\small\n\\begin{tabular}{l" + "r" * (2 * len(loads))
        + "}\n\\toprule\n & \\multicolumn{" + str(len(loads)) + "}{c}{Acceptance (\\%)} & \\multicolumn{"
        + str(len(loads)) + "}{c}{Power (W)}\\\\\n\\cmidrule(lr){2-" + str(1 + len(loads)) + "}\\cmidrule(lr){"
        + str(2 + len(loads)) + "-" + str(1 + 2 * len(loads)) + "}\nMethod & " + hl + " & " + hl + "\\\\\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ----------------------------------------------------------------------------- tiers
def tab_tier(e1: pd.DataFrame, out: Path):
    ms = [m for m in ["GNN-MAPPO-E", "GNN-MAPPO", "MLP-MAPPO-E", "MLP-MAPPO", "DQN-Lagrangian", "Greedy-Latency",
                      "Greedy-Energy", "Greedy-Mix:0.3", "Greedy-Mix:1"] if m in set(e1.method)]
    lines = []
    for topo in TOPO:
        d = e1[(e1.topology == topo) & (e1.load == 1.0)]
        if d.empty:
            continue
        lines.append(f"\\midrule\\multicolumn{{8}}{{l}}{{\\textit{{{TOPO[topo]}}}, $\\rho=1$}}\\\\")
        for m in ms:
            g = d[d.method == m]
            if g.empty:
                continue
            T = g.window_s.mean()
            pe, pf, pc = (g[f"energy_{t}_j"].mean() / T for t in ("edge", "fog", "cloud"))
            pl = g.link_energy_j.mean() / T
            nv = g[["vnf_edge", "vnf_fog", "vnf_cloud"]].mean()
            sh = 100 * nv / nv.sum()
            tk = TOPO_KEY[topo]
            mac(f"res{tk}Tier{m}EdgePow", fmt(pe, 0))
            mac(f"res{tk}Tier{m}FogPow", fmt(pf, 0))
            mac(f"res{tk}Tier{m}CloudPow", fmt(pc, 0))
            mac(f"res{tk}Tier{m}EdgeShare", f"{sh.vnf_edge:.0f}")
            mac(f"res{tk}Tier{m}FogShare", f"{sh.vnf_fog:.0f}")
            mac(f"res{tk}Tier{m}CloudShare", f"{sh.vnf_cloud:.0f}")
            lines.append(f"{LABEL[m]} & {pe:,.0f} & {pf:,.0f} & {pc:,.0f} & {pl:,.0f} & "
                         f"{sh.vnf_edge:.0f} & {sh.vnf_fog:.0f} & {sh.vnf_cloud:.0f}\\\\")
    (out / "tab_tier.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Where the power goes at $\\rho=1$: average power per tier "
        "and links (W), and share of placed VNFs per tier (\\%), mean over ten seeds.}\n\\label{tab:tier}\n"
        "\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{4}{c}"
        "{Power (W)} & \\multicolumn{3}{c}{VNFs (\\%)}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\n"
        "Method & Edge & Fog & Cloud & Links & Edge & Fog & Cloud\\\\\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------------------- SLA integrity
def tab_sla(e0: pd.DataFrame, e1: pd.DataFrame, out: Path):
    ms = [m for m in ["GNN-MAPPO", "MLP-MAPPO", "Greedy-Latency", "Greedy-Energy", "Least-Loaded", "Best-Fit",
                      "Greedy-Mix:0.3", "Greedy-Mix:1"] if m in set(e0.method)]
    lines = []
    for topo in TOPO:
        d0 = e0[e0.topology == topo]
        if d0.empty:
            continue
        keys = d0[["load", "seed"]].drop_duplicates()
        d1 = e1[e1.topology == topo].merge(keys, on=["load", "seed"])
        lines.append(f"\\midrule\\multicolumn{{6}}{{l}}{{\\textit{{{TOPO[topo]}}}}}\\\\")
        for m in ms:
            a = d0[d0.method == m]
            b = d1[d1.method == m]
            acc_n = (a.acceptance * a.requests).sum()
            viol = 100 * a.sla_violated.sum() / max(1.0, acc_n)
            lines.append(f"{LABEL[m]} & {viol:.1f} & {100 * a.acceptance.mean():.1f} & {a.avg_power_w.mean():,.0f}"
                         f" & {100 * b.acceptance.mean():.1f} & {b.avg_power_w.mean():,.0f}\\\\")
            mac(f"res{TOPO_KEY[topo]}Viol{m.replace('-', '').replace(':', '').replace('.', '')}", f"{viol:.1f}")
            k = m.replace('-', '').replace(':', '').replace('.', '')
            mac(f"res{TOPO_KEY[topo]}Sla{k}AccOff", f"{100 * a.acceptance.mean():.1f}")
            mac(f"res{TOPO_KEY[topo]}Sla{k}AccOn", f"{100 * b.acceptance.mean():.1f}")
            mac(f"res{TOPO_KEY[topo]}Sla{k}PowOff", f"{a.avg_power_w.mean():,.0f}")
            mac(f"res{TOPO_KEY[topo]}Sla{k}PowOn", f"{b.avg_power_w.mean():,.0f}")
    (out / "tab_sla.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Effect of admissions on running services ($\\rho\\in\\{0.75,1,"
        "1.25\\}$, seeds 0--4). \\emph{Viol.}: admitted services that exceed their latency budget at least once "
        "during their lifetime when protection is off, per 100 admitted services. Acceptance (\\%) and power (W) without and with protection.}\n"
        "\\label{tab:sla}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{No protection} & \\multicolumn{2}{c}{Protection}\\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\nMethod & Viol. & Acc. & Power & Acc. & Power\\\\\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------------------- ablation
def tab_ablation(abl: pd.DataFrame, abl_ens: pd.DataFrame, e1: pd.DataFrame, out: Path, variants: dict):
    """Ablations on the IoT fog network for single seeds (seed level) and ten-seed ensembles (block level)."""
    keys = abl[["load", "seed"]].drop_duplicates()
    base = e1[(e1.topology == "fog-12e6f-s1")].merge(keys, on=["load", "seed"])
    single = pd.concat([abl, base[base.method.isin(["GNN-MAPPO", "MLP-MAPPO"])]], ignore_index=True)
    ens = base[base.method.isin(["GNN-MAPPO-E", "MLP-MAPPO-E"])]
    if not abl_ens.empty:
        ens = pd.concat([ens, abl_ens.merge(keys, on=["load", "seed"])], ignore_index=True)
    heur = base[base.method.isin(HEUR + MIX)]
    gs = gaps(pd.concat([single, heur], ignore_index=True), list(variants))
    ge = gaps(pd.concat([ens, heur], ignore_index=True), [v + "-E" for v in variants])
    rows = []
    for v, lab in variants.items():
        d = single[single.method == v]
        if d.empty:
            continue
        g1 = boot_ci(gs[gs.method == v].groupby("seed").gap.mean() * 100)
        de = ens[ens.method == v + "-E"]
        if not de.empty:
            gb = ge[ge.method == v + "-E"].set_index(["load", "seed"]).gap * 100
            g2 = boot_ci(gb)
            ref = ge[ge.method == "GNN-MAPPO-E"].set_index(["load", "seed"]).gap * 100
            dd = (gb - ref).dropna()
            p2 = wil(dd.values) if v != "GNN-MAPPO" else np.nan
            rows.append((v, lab, d, g1, de, g2, dd.mean() if v != "GNN-MAPPO" else np.nan, p2))
        else:
            rows.append((v, lab, d, g1, de, (np.nan,) * 3, np.nan, np.nan))
    ps = [r[7] for r in rows]
    ok = [i for i, x in enumerate(ps) if not np.isnan(x)]
    adj = np.full(len(ps), np.nan)
    if ok:
        adj[ok] = holm([ps[i] for i in ok])
    lines = []
    for i, (v, lab, d, g1, de, g2, dg, _) in enumerate(rows):
        ce = (f"{100 * de.acceptance.mean():.2f} & {de.avg_power_w.mean():,.0f} & {pm(*g2, 1)} & "
              f"{fmt(dg, 1)} ({fmtp(adj[i])})") if not de.empty else "-- & -- & -- & --"
        if v == "GNN-MAPPO" and not de.empty:
            ce = f"{100 * de.acceptance.mean():.2f} & {de.avg_power_w.mean():,.0f} & {pm(*g2, 1)} & --"
        lines.append(f"{lab} & {100 * d.acceptance.mean():.2f} & {d.avg_power_w.mean():,.0f} & {pm(*g1, 1)} & "
                     + ce + "\\\\")
        k = v.replace("-", "").replace("_", "").replace(":", "")
        mac(f"resAbl{k}Gap", fmt(g1[0], 1))
        mac(f"resAbl{k}EGap", fmt(g2[0], 1))
        mac(f"resAbl{k}EdGap", fmt(dg, 1))
        mac(f"resAbl{k}EdGapP", fmtp(adj[i]))
        mac(f"resAbl{k}Acc", f"{100 * d.acceptance.mean():.2f}")
        if not de.empty:
            mac(f"resAbl{k}EAcc", f"{100 * de.acceptance.mean():.2f}")
    (out / "tab_ablation.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Ablations on the IoT fog network ($\\rho\\in\\{0.75,1,1.25\\}$, "
        "protection and shield on). Acceptance (\\%), average power (W) and frontier gap (\\%, 95\\% bootstrap CI) "
        "of single training seeds (CI over the ten seeds) and of the ten-seed ensemble of each variant (CI over "
        "the " + str(len(abl_ens[["load", "seed"]].drop_duplicates()) if not abl_ens.empty else 0) + " test traces of "
        "seeds " + (f"0--{int(abl_ens.seed.max())}" if not abl_ens.empty else "") + "). $\\Delta$gap: ensemble gap minus that of the GNN-MAPPO ensemble on the same traces, "
        "with the Holm-corrected Wilcoxon $p$ over traces.}\n\\label{tab:ablation}\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Single seeds} "
        "& \\multicolumn{4}{c}{Ten-seed ensemble}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"
        "Variant & Acc. & Power & Gap & Acc. & Power & Gap & $\\Delta$gap ($p$)\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ----------------------------------------------------------------------------- figure
def fig_frontier(e1: pd.DataFrame, figdir: Path, load: float = 1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    topos = [t for t in TOPO if t in set(e1.topology)]
    fig, axes = plt.subplots(1, len(topos), figsize=(4.2 * len(topos), 3.4))
    axes = np.atleast_1d(axes)
    style = {"GNN-MAPPO-E": ("o", "C3"), "GNN-MAPPO": ("o", "C5"), "MLP-MAPPO-E": ("s", "C1"),
             "DQN-Lagrangian-E": ("D", "C4"), "Greedy-Latency": ("^", "C0"), "Greedy-Energy": ("v", "C2"), "Least-Loaded": ("<", "C7"),
             "Best-Fit": (">", "C8")}
    for ax, topo in zip(axes, topos):
        d = e1[(e1.topology == topo) & (e1.load == load)]
        g = d.groupby("method")[["acceptance", "avg_power_w"]].mean()
        fam = [m for m in ["Greedy-Energy"] + MIX + ["Greedy-Latency"] if m in g.index]
        ax.plot(100 * g.loc[fam, "acceptance"], g.loc[fam, "avg_power_w"], "-", color="0.6", lw=1.2,
                label="Greedy-Mix family", zorder=1)
        for m in MIX:
            if m in g.index:
                ax.plot(100 * g.loc[m, "acceptance"], g.loc[m, "avg_power_w"], ".", color="0.4", zorder=2)
                ax.annotate(m.split(":")[1], (100 * g.loc[m, "acceptance"], g.loc[m, "avg_power_w"]),
                            fontsize=7, color="0.35", xytext=(3, -8), textcoords="offset points")
        for m, (mk, c) in style.items():
            if m not in g.index:
                continue
            sd = d[d.method == m]
            ax.errorbar(100 * g.loc[m, "acceptance"], g.loc[m, "avg_power_w"],
                        xerr=100 * 1.96 * sd.acceptance.std() / np.sqrt(len(sd)),
                        yerr=1.96 * sd.avg_power_w.std() / np.sqrt(len(sd)), fmt=mk, color=c, ms=6,
                        label=LABEL.get(m, m).replace("$", ""), zorder=3, capsize=2)
        ax.set_title(f"{TOPO[topo]}, $\\rho={load:g}$")
        ax.set_xlabel("Acceptance (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Average power (W)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "fig_frontier.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- seeds and ensembles
def tab_seeds(sel: pd.DataFrame, gp_test: pd.DataFrame, out: Path):
    """Single seeds vs seed ensembles: selection stream (choice of the rule) and test traces."""
    rl = [m for m in sel.method.unique() if m.startswith("GNN")]
    gs = gaps(sel, rl)
    gs["gap"] *= 100
    lines = []
    for topo in TOPO:
        a = gs[gs.topology == topo]
        if a.empty:
            continue
        per = a[a.method.str.match(r"GNN-s\d")].groupby("method").gap.mean()
        ens = a[a.method.str.startswith("GNN-ens")].groupby("method").gap.mean()
        t = gp_test[(gp_test.topology == topo)]
        ts = t[t.method == "GNN-MAPPO"].groupby("seed").gap.mean() * 100
        te = t[t.method == "GNN-MAPPO-E"].gap.mean() * 100 if "GNN-MAPPO-E" in set(t.method) else np.nan
        best = per.idxmin().split("-s")[1]
        lines.append(f"{TOPO[topo]} & {fmt(per.median(), 1)} & [{fmt(per.min(), 1)}, {fmt(per.max(), 1)}] & "
                     f"{fmt(ens.get('GNN-ens5a', np.nan), 1)} & {fmt(ens.get('GNN-ens5b', np.nan), 1)} & "
                     f"{fmt(ens.get('GNN-ens10', np.nan), 1)} & {fmt(ts.median(), 1)} & "
                     f"[{fmt(ts.min(), 1)}, {fmt(ts.max(), 1)}] & {fmt(te, 1)}\\\\")
        tk = TOPO_KEY[topo]
        mac(f"resSel{tk}SeedMed", fmt(per.median(), 1))
        mac(f"resSel{tk}SeedMin", fmt(per.min(), 1))
        mac(f"resSel{tk}SeedMax", fmt(per.max(), 1))
        mac(f"resSel{tk}BestSeed", best)
        mac(f"resSel{tk}EnsTen", fmt(ens.get("GNN-ens10", np.nan), 1))
        mac(f"resSel{tk}EnsA", fmt(ens.get("GNN-ens5a", np.nan), 1))
        mac(f"resSel{tk}EnsB", fmt(ens.get("GNN-ens5b", np.nan), 1))
        mac(f"resSel{tk}EnsBetter", str(int((per > ens.get("GNN-ens10", np.nan)).sum())))
        from scipy.stats import spearmanr
        sidx = pd.Series(per.values, index=[int(m.split("-s")[1]) for m in per.index]).sort_index()
        common = sidx.index.intersection(ts.index)
        rho = spearmanr(sidx[common], ts[common]).correlation
        mac(f"resSel{tk}Spearman", f"{rho:.2f}")
        mac(f"resSel{tk}TopFive", str(len(set(sidx.nsmallest(5).index) & set(ts.nsmallest(5).index))))
        mac(f"resTest{tk}BestSelSeed", fmt(ts.get(int(best), np.nan), 1))
        mac(f"resTest{tk}SeedMed", fmt(ts.median(), 1))
        mac(f"resTest{tk}SeedMin", fmt(ts.min(), 1))
        mac(f"resTest{tk}SeedMax", fmt(ts.max(), 1))
    (out / "tab_seeds.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Frontier gap (\\%) of single training seeds and of seed "
        "ensembles of GNN-MAPPO. \\emph{Selection}: selection traces (two traces, $\\rho\\in\\{0.5,1,1.5\\}$, "
        "the same traces for every policy), on which the deployment rule was chosen. \\emph{Test}: test traces of "
        "Table~\\ref{tab:frontier}. Ensembles average the action distributions of seeds 0--4 (a), 5--9 (b) or "
        "all ten seeds.}\n\\label{tab:seeds}\n\\small\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{5}{c}{Selection traces} & \\multicolumn{3}{c}{Test traces}\\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\n"
        " & \\multicolumn{2}{c}{Single seeds} & \\multicolumn{3}{c}{Ensembles} & \\multicolumn{2}{c}{Single seeds} & "
        "Ensemble\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
        "Network & median & range & 5 (a) & 5 (b) & 10 & median & range & 10\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ----------------------------------------------------------------------------- shield
def tab_shield(off: pd.DataFrame, e1: pd.DataFrame, out: Path):
    keys = off[["load", "seed"]].drop_duplicates()
    on = e1[e1.topology == "fog-12e6f-s1"].merge(keys, on=["load", "seed"])
    lines = []
    for m in [x for x in ORDER if x in set(off.method) and x in set(on.method)]:
        a, b = off[off.method == m], on[on.method == m]
        dea = 100 * a.rej_dead_end.fillna(0).mean()
        deb = 100 * b.rej_dead_end.fillna(0).mean()
        da = 100 * (b.acceptance.mean() - a.acceptance.mean())
        lines.append(f"{LABEL[m]} & {100 * a.acceptance.mean():.1f} & {100 * b.acceptance.mean():.1f} & "
                     f"{fmt(da, 1)} & {dea:.1f} & {deb:.1f} & {a.avg_power_w.mean():,.0f} & "
                     f"{b.avg_power_w.mean():,.0f}\\\\")
        k = m.replace("-", "").replace(":", "")
        mac(f"resShield{k}DeadOff", f"{dea:.1f}")
        mac(f"resShield{k}DeadOn", f"{deb:.1f}")
        mac(f"resShield{k}Gain", fmt(da, 1))
    (out / "tab_shield.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Completion shield on the IoT fog network ($\\rho\\in\\{0.75,1,"
        "1.25\\}$, ten seeds, protection on): acceptance (\\%), share of requests rejected after a partial "
        "placement (dead ends, \\%) and average power (W) without and with the shield. The learned policies are "
        "the same checkpoints in both columns.}\n\\label{tab:shield}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Acceptance} & \\multicolumn{2}{c}"
        "{Dead ends} & \\multicolumn{2}{c}{Power}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Method & off & on & $\\Delta$ & off & on & off & on\\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------------------- learner and objective
def tab_learner(e1: pd.DataFrame, wtd: pd.DataFrame, out: Path):
    """PPO-Lagrangian vs DQN-Lagrangian vs weighted-reward PPO on the IoT fog network (same traces)."""
    fog = e1[e1.topology == "fog-12e6f-s1"]
    allm = pd.concat([fog, wtd], ignore_index=True) if not wtd.empty else fog
    variants = [("GNN-MAPPO-E", "PPO, Lagrangian, ensemble"), ("DQN-Lagrangian-E", "DQN, Lagrangian, ensemble"),
                ("GNN-MAPPO", "PPO, Lagrangian ($\\alpha=0.98$)"), ("DQN-Lagrangian", "DQN, Lagrangian ($\\alpha=0.98$)")]
    variants += [(f"PPO-w{w}", f"PPO, weighted ($r_{{\\mathrm{{acc}}}}=r_{{\\mathrm{{rej}}}}={w}$)") for w in (2, 7, 20)]
    variants = [(v, l) for v, l in variants if v in set(allm.method)]
    gp = gaps(allm, [v for v, _ in variants])
    loads = sorted(allm.load.unique())
    lines = []
    ref = allm[allm.method == "GNN-MAPPO"]
    for v, lab in variants:
        d = allm[allm.method == v]
        seeds = sorted(d.seed.unique())
        acc_l = d.groupby("load").acceptance.mean() * 100
        unit = ["load", "seed"] if v.endswith("-E") else ["seed"]
        g = gp[gp.method == v].groupby(unit).gap.mean() * 100
        mu, lo, hi = boot_ci(g)
        # Paired with GNN-MAPPO on the seeds that both have.
        r = ref[ref.seed.isin(seeds)]
        da = 100 * (d.groupby("seed").acceptance.mean() - r.groupby("seed").acceptance.mean()).dropna()
        dp = 100 * ((d.groupby("seed").avg_power_w.mean() - r.groupby("seed").avg_power_w.mean())
                    / r.groupby("seed").avg_power_w.mean()).dropna()
        pg = wil(dp.values) if v != "GNN-MAPPO" else np.nan
        cells = " & ".join(f"{acc_l.get(l, np.nan):.1f}" for l in (loads[0], 1.0, loads[-1]))
        nlab = (f"{len(glob.glob(str(ROOT / 'runs_v3' / 'fog' / 'DQN_s*')))}" if v == "DQN-Lagrangian-E" else
                "10" if v == "GNN-MAPPO-E" else str(len(seeds)))
        lines.append(f"{lab} & {nlab} & {cells} & {100 * d.acceptance.mean():.2f} & "
                     f"{d.avg_power_w.mean():,.0f} & {pm(mu, lo, hi, 1)}\\\\")
        k = v.replace("-", "")
        mac(f"resLrn{k}Acc", f"{100 * d.acceptance.mean():.2f}")
        mac(f"resLrn{k}AccLo", f"{acc_l.get(loads[0], np.nan):.1f}")
        mac(f"resLrn{k}AccHi", f"{acc_l.get(loads[-1], np.nan):.1f}")
        mac(f"resLrn{k}Pow", f"{d.avg_power_w.mean():,.0f}")
        mac(f"resLrn{k}Gap", fmt(mu, 1))
        mac(f"resLrn{k}GapLo", fmt(lo, 1))
        mac(f"resLrn{k}GapHi", fmt(hi, 1))
        if v not in ("GNN-MAPPO", "GNN-MAPPO-E") and not v.endswith("-E"):
            mac(f"resLrn{k}dAcc", fmt(da.mean(), 2))
            mac(f"resLrn{k}dPow", fmt(dp.mean(), 1))
            mac(f"resLrn{k}dPowP", fmtp(pg))
            mac(f"resLrn{k}Seeds", str(len(seeds)))
    (out / "tab_learner.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Learner and objective on the IoT fog network, all with the same "
        "encoder, mask, protection rule and shield, on the same test traces. Acceptance (\\%) at the lowest, "
        "middle and highest offered load and averaged over the five loads, average power (W) and frontier gap "
        "(\\%, mean over seeds with 95\\% bootstrap CI). $n$: number of training seeds.}\n\\label{tab:learner}\n"
        "\\small\n\\begin{tabular}{lrrrrrrr}\n\\toprule\nLearner and objective & $n$ & "
        + " & ".join(f"Acc.\\ $\\rho={l:g}$" for l in (loads[0], 1.0, loads[-1]))
        + " & Acc. & Power & Frontier gap\\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def sel_learners(sel: pd.DataFrame):
    """Training-protocol choice on the selection stream (macros only)."""
    sel = sel.replace({"method": {"DQN-Lagrangian": "DQN"}})
    rl = [m for m in ["GNNv2", "EPprot", "CONT", "CONTprot", "DQN"] if m in set(sel.method)]
    gp = gaps(sel, rl)
    for m in rl:
        d = sel[sel.method == m]
        mac(f"resSel{m}Acc", f"{100 * d.acceptance.mean():.2f}")
        mac(f"resSel{m}Pow", f"{d.avg_power_w.mean():,.0f}")
        mac(f"resSel{m}Gap", fmt(100 * gp[gp.method == m].gap.mean(), 1))


# ----------------------------------------------------------------------------- zero-shot
ZS_LABEL = {"fog-40e12f-s1": "IoT fog, 40 edge + 12 fog", "real-janos_us": "Janos-US",
            "real-germany50": "Germany50", "real-uninett2011": "Uninett2011",
            "real-geant2012": "GÉANT", "real-nobel_germany": "Nobel-Germany"}


def tab_zeroshot(e4: pd.DataFrame, e1: pd.DataFrame, out: Path):
    lines = []
    ms = ["GNN-MAPPO-E-zs", "GNN-MAPPO-zs", "MLP-MAPPO-zs", "Greedy-Latency", "Greedy-Energy", "Greedy-Mix:0.3",
          "Greedy-Mix:1"]
    lab = {"GNN-MAPPO-E-zs": "GNN-MAPPO (ensemble), zero-shot", "GNN-MAPPO-zs": "GNN-MAPPO, single seed, zero-shot",
           "MLP-MAPPO-zs": "MLP-MAPPO, single seed, zero-shot", "GNN-MAPPO-E": "GNN-MAPPO (ensemble), trained here"}
    rows_gap = {}
    for topo in ZS_LABEL:
        d = e4[e4.topology == topo]
        if d.empty:
            continue
        keys = d[["load", "seed"]].drop_duplicates()
        here = e1[(e1.topology == topo) & (e1.method == "GNN-MAPPO-E")].merge(keys, on=["load", "seed"])
        dd = pd.concat([d, here], ignore_index=True)
        learned = [m for m in ["GNN-MAPPO-E-zs", "GNN-MAPPO-zs", "MLP-MAPPO-zs", "GNN-MAPPO-E"] if m in set(dd.method)]
        gp = gaps(dd, learned)
        nn = ""
        try:
            sub = sorted(glob.glob(str(R / "E4" / topo.replace("real-", "real_").replace("fog-40e12f-s1", "fog_40x12")
                                       / "s*" / "infrastructure.json")))
            if sub:
                nn = f", $|\\mathcal{{V}}|={json.loads(Path(sub[0]).read_text()).get('nodes', '')}$"
        except Exception:
            pass
        lines.append(f"\\midrule\\multicolumn{{5}}{{l}}{{\\textit{{{ZS_LABEL[topo]}}}{nn}}}\\\\")
        for m in ([x for x in ms if x in set(dd.method)] + (["GNN-MAPPO-E"] if "GNN-MAPPO-E" in set(dd.method) else [])):
            g = dd[dd.method == m]
            gm = gp[gp.method == m].gap.mean() * 100 if m in learned else np.nan
            lines.append(f"{lab.get(m, LABEL.get(m, m))} & {100 * g.acceptance.mean():.1f} & "
                         f"{g.avg_power_w.mean():,.0f} & {fmt(gm, 1)} & {g.decision_ms_mean.mean():.1f}\\\\")
            if m in learned:
                rows_gap[(topo, m)] = gm
                k = ZS_LABEL[topo].split(",")[0].replace(" ", "").replace("-", "").replace("É", "E")
                mac(f"resZs{k}{m.replace('-', '')}Gap", fmt(gm, 1))
                mac(f"resZs{k}{m.replace('-', '')}Acc", f"{100 * g.acceptance.mean():.1f}")
        for ref in ("Greedy-Latency",):
            g = dd[dd.method == ref]
            k = ZS_LABEL[topo].split(",")[0].replace(" ", "").replace("-", "").replace("É", "E")
            mac(f"resZs{k}GLAcc", f"{100 * g.acceptance.mean():.1f}")
    (out / "tab_zeroshot.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Zero-shot transfer of the policies trained only on the 19-node "
        "IoT fog network ($\\rho\\in\\{0.75,1,1.25\\}$, seeds 0--" + str(int(e4.seed.max())) + ", protection and "
        "shield on); GNN-MAPPO (ensemble) averages the ten fog-trained policies. Gap: power relative "
        "to the front of the four heuristics shown, at the same acceptance. For Nobel-Germany and GÉANT the "
        "policy trained on that topology is shown for comparison.}\n\\label{tab:zeroshot}\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Acc. (\\%) & Power (W) & "
        "Gap (\\%) & Dec. (ms)\\\\\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------------------- per-request ILP
def tab_ilp(ilp: pd.DataFrame, e1: pd.DataFrame, out: Path):
    lines = []
    for topo in TOPO:
        d = ilp[ilp.topology == topo]
        if d.empty:
            continue
        keys = d[["load", "seed"]].drop_duplicates()
        b = e1[e1.topology == topo].merge(keys, on=["load", "seed"])
        tk = TOPO_KEY[topo]
        cand = ["ILP-PerRequest", "GNN-MAPPO-E", "GNN-MAPPO"]
        gpi = gaps(pd.concat([d, b], ignore_index=True), cand)
        gmap = {m: 100 * g.gap.mean() for m, g in gpi.groupby("method")}
        lines.append(f"\\midrule\\multicolumn{{5}}{{l}}{{\\textit{{{TOPO[topo]}}}, {len(keys.seed.unique())} "
                     f"seeds}}\\\\")
        mac(f"resIlp{tk}Seeds", str(len(keys.seed.unique())))
        for m in ["ILP-PerRequest", "GNN-MAPPO-E", "GNN-MAPPO", "Greedy-Latency", "Greedy-Energy", "Greedy-Mix:1"]:
            g = d if m == "ILP-PerRequest" else b[b.method == m]
            if g.empty:
                continue
            lines.append(f"{LABEL.get(m, m)} & {100 * g.acceptance.mean():.1f} & {g.avg_power_w.mean():,.0f} & "
                         f"{fmt(gmap.get(m, np.nan), 1)} & {g.decision_ms_mean.mean():.1f}\\\\")
            k = m.replace("-", "").replace(":", "")
            mac(f"resIlp{tk}{k}Acc", f"{100 * g.acceptance.mean():.1f}")
            mac(f"resIlp{tk}{k}Pow", f"{g.avg_power_w.mean():,.0f}")
            mac(f"resIlp{tk}{k}Dec", f"{g.decision_ms_mean.mean():.0f}")
            for ld, gl in g.groupby("load"):
                mac(f"resIlp{tk}{k}Acc{ld:g}", f"{100 * gl.acceptance.mean():.1f}")
                mac(f"resIlp{tk}{k}Dec{ld:g}", f"{gl.decision_ms_mean.mean():.0f}")
        for m, v in gmap.items():
            mac(f"resIlp{tk}{m.replace('-', '')}Gap", fmt(v, 1))
            mac(f"resIlp{tk}{m.replace('-', '')}GapAbs", fmt(abs(v), 1))
        ref = "GNN-MAPPO-E" if "GNN-MAPPO-E" in set(b.method) else "GNN-MAPPO"
        gi, gg = d.avg_power_w.mean(), b[b.method == ref].avg_power_w.mean()
        mac(f"resIlp{tk}GNNvsIlpPow", fmt(100 * (gg - gi) / gi, 1))
        mac(f"resIlp{tk}GNNvsIlpAcc", fmt(100 * (b[b.method == ref].acceptance.mean() - d.acceptance.mean()), 1))
    (out / "tab_ilp.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Per-request optimum (ILP: minimum incremental power for each "
        "request given the current state, with the protection constraint) against the online methods on the same "
        "traces ($\\rho\\in\\{0.75,1,1.25\\}$). Gap: power relative to the Greedy-Mix front at the same acceptance "
        "(\\%); decision time per request in ms.}\n\\label{tab:ilp}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Acc. (\\%) & Power (W) & Gap & Dec.\\\\\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------------------- reservation on GÉANT
def tab_reserve(sel: pd.DataFrame, sel_ens: pd.DataFrame, test: pd.DataFrame, e1: pd.DataFrame, out: Path):
    """Selection sweep (selection stream) and test comparison with the chosen reservation share.

    The share is the smallest R at which the deployed policy (the ten-seed ensemble when available) accepts
    at most one percentage point less than Greedy-Latency under the same R, on the selection traces.
    """
    sel = pd.concat([sel, sel_ens], ignore_index=True) if not sel_ens.empty else sel.copy()
    sel["R"] = sel["reserve"]
    pol = "GNN-MAPPO-E" if "GNN-MAPPO-E" in set(sel.method) else "GNN-MAPPO"
    seeds = sorted(sel[sel.method == pol].seed.unique())
    sel = sel[sel.seed.isin(seeds)]
    ms = [m for m in [pol, "GNN-MAPPO", "Greedy-Latency", "Greedy-Mix:3", "Greedy-Mix:1", "Greedy-Mix:0.3",
                      "Greedy-Energy"] if m in set(sel.method)]
    ms = list(dict.fromkeys(ms))
    Rs = sorted(sel.R.unique())
    lines = []
    for m in ms:
        g = sel[sel.method == m].groupby("R")
        a, p = g.acceptance.mean() * 100, g.avg_power_w.mean()
        lines.append(f"{LABEL[m]} & " + " & ".join(f"{a.get(r, np.nan):.1f} / {p.get(r, np.nan):,.0f}" for r in Rs)
                     + "\\\\")
    a = sel.groupby(["R", "method"]).acceptance.mean().unstack()
    rstar = float(test.reserve.iloc[0]) if not test.empty else 0.3
    mac("resResTest", f"{rstar:g}")
    mac("resResSelSeeds", str(len(seeds)))
    ic = sel.groupby(["R", "method"]).acc_industrial_control.mean().unstack()
    pw = sel.groupby(["R", "method"]).avg_power_w.mean().unstack()
    for r in Rs:
        k = f"{int(round(100 * r))}"
        for m, mk in ((pol, "GNN"), ("Greedy-Latency", "GL"), ("Greedy-Energy", "GE")):
            if m in a.columns:
                mac(f"resResSel{mk}Acc{k}", f"{100 * a.loc[r, m]:.1f}")
                mac(f"resResSel{mk}IC{k}", f"{100 * ic.loc[r, m]:.1f}")
                mac(f"resResSel{mk}Pow{k}", f"{pw.loc[r, m]:,.0f}")
        mac(f"resResSelGap{k}", fmt(100 * (a.loc[r, pol] - a.loc[r, "Greedy-Latency"]), 2))
    (out / "tab_reserve_sel.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Choice of the reservation share $R$ on GÉANT, on the selection "
        f"traces (never on the test traces; $\\rho\\in\\{{0.75,1,1.25\\}}$, {len(seeds)} traces). Acceptance (\\%) / "
        "average power (W). GNN-MAPPO without fine-tuning for the reservation.}\n\\label{tab:reserve_sel}\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l" + "r" * len(Rs) + "}\n\\toprule\nMethod & "
        + " & ".join(f"$R={r:g}$" for r in Rs) + "\\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    if test.empty:
        return rstar
    learned = [m for m in ["GNN-MAPPO-R-E", "GNN-MAPPO-E"] if m in set(test.method)]
    done = set(test.seed)
    for m in learned:
        done &= set(test[test.method == m].seed)
    test = test[test.seed.isin(done)]
    base = e1[(e1.topology == "real-geant2012") & e1.seed.isin(done)]
    mac("resResTestSeeds", str(len(done)))
    gp0 = gaps(base, [m for m in learned if m in set(base.method)])
    gpr = gaps(test, learned)
    LABEL.setdefault("GNN-MAPPO-R-E", "GNN-MAPPO, fine-tuned with $R$, ensemble")
    cols = ["acceptance", "avg_power_w", "acc_industrial_control"]

    def blk(df, m):
        return df[df.method == m].groupby(["load", "seed"])[cols].mean()

    lines = []
    for m in learned + HEUR + MIX:
        a0, a1 = base[base.method == m], test[test.method == m]
        if a1.empty and a0.empty:
            continue
        g0 = 100 * gp0[gp0.method == m].gap.mean() if m in set(gp0.get("method", [])) else np.nan
        g1 = 100 * gpr[gpr.method == m].gap.mean() if m in learned else np.nan
        c0 = f"{100 * a0.acceptance.mean():.2f} & {a0.avg_power_w.mean():,.0f}" if not a0.empty else "-- & --"
        c1 = f"{100 * a1.acceptance.mean():.2f} & {a1.avg_power_w.mean():,.0f}" if not a1.empty else "-- & --"
        i0 = f"{100 * a0.acc_industrial_control.mean():.1f}" if not a0.empty else "--"
        i1 = f"{100 * a1.acc_industrial_control.mean():.1f}" if not a1.empty else "--"
        lines.append(f"{LABEL.get(m, m)} & {c0} & {i0} & {fmt(g0, 1)} & {c1} & {i1} & {fmt(g1, 1)}\\\\")
        k = m.replace("-", "").replace(":", "")
        if not a0.empty:
            mac(f"resResZero{k}IC", f"{100 * a0.acc_industrial_control.mean():.1f}")
            mac(f"resResZero{k}Acc", f"{100 * a0.acceptance.mean():.2f}")
            mac(f"resResZero{k}Pow", f"{a0.avg_power_w.mean():,.0f}")
            if not np.isnan(g0):
                mac(f"resResZero{k}Gap", fmt(g0, 1))
        if not a1.empty:
            mac(f"resRes{k}IC", f"{100 * a1.acc_industrial_control.mean():.1f}")
            mac(f"resRes{k}Acc", f"{100 * a1.acceptance.mean():.2f}")
            mac(f"resRes{k}Pow", f"{a1.avg_power_w.mean():,.0f}")
            if m in learned:
                gs = gpr[gpr.method == m].groupby(["load", "seed"]).gap.mean() * 100
                mu, lo, hi = boot_ci(gs)
                mac(f"resRes{k}Gap", fmt(mu, 1))
                mac(f"resRes{k}GapLo", fmt(lo, 1))
                mac(f"resRes{k}GapHi", fmt(hi, 1))
                mac(f"resRes{k}GapP", fmtp(wil(gs.values)))
                mac(f"resRes{k}GapBetter", f"{100 * (gs < 0).mean():.0f}")
    # paired block-level comparisons (Holm within this family)
    top = learned[0]
    pairs = {"TopGL": (blk(test, top), blk(test, "Greedy-Latency")),
             "TopGE": (blk(test, top), blk(test, "Greedy-Energy")),
             "Tune": (blk(test, top), blk(test, "GNN-MAPPO-E")) if len(learned) > 1 else None,
             "EGL": (blk(test, "GNN-MAPPO-E"), blk(test, "Greedy-Latency")),
             "ResE": (blk(test, "GNN-MAPPO-E"), blk(base, "GNN-MAPPO-E")),
             "ZeroGL": (blk(base, "GNN-MAPPO-E"), blk(base, "Greedy-Latency"))}
    res, ps, names = {}, [], []
    for k, v in pairs.items():
        if v is None:
            continue
        x, y = v
        x, y = x.align(y, join="inner", axis=0)
        da = 100 * (x.acceptance - y.acceptance)
        dp = 100 * (x.avg_power_w - y.avg_power_w) / y.avg_power_w
        di = 100 * (x.acc_industrial_control - y.acc_industrial_control)
        res[k] = (da, dp, di)
        ps += [wil(da.values), wil(dp.values)]
        names += [(k, "Acc"), (k, "Pow")]
    adj = holm(np.nan_to_num(np.array(ps), nan=1.0)) if ps else []
    for (k, w), p in zip(names, adj):
        mac(f"resRes{k}{w}P", fmtp(p))
    for k, (da, dp, di) in res.items():
        mac(f"resRes{k}Acc", fmt(da.mean(), 2))
        mac(f"resRes{k}AccAbs", fmt(abs(da.mean()), 2))
        mac(f"resRes{k}Pow", fmt(dp.mean(), 1))
        mac(f"resRes{k}PowAbs", fmt(abs(dp.mean()), 1))
        mac(f"resRes{k}IC", fmt(di.mean(), 1))
        mac(f"resRes{k}PowLower", f"{100 * (dp < 0).mean():.0f}")
        mac(f"resRes{k}Dom", f"{100 * ((da >= 0) & (dp < 0)).mean():.0f}")
        mac(f"resRes{k}N", str(len(da)))
    # kept for backwards compatibility with earlier text
    if "TopGL" in res:
        da, dp, _ = res["TopGL"]
        mac("resResTopAccGL", fmt(da.mean(), 2))
        mac("resResTopAccAbsGL", fmt(abs(da.mean()), 2))
        mac("resResTopPowGL", fmt(dp.mean(), 1))
        mac("resResTopPowAbsGL", fmt(abs(dp.mean()), 1))
    (out / "tab_reserve.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{GÉANT without and with capacity reservation "
        f"($R={rstar:g}$) on the test traces (five loads, {len(done)} seeds, protection and shield on). "
        "Acceptance (\\%), average power (W) and frontier gap (\\%); the frontier is recomputed under the same "
        "$R$; IC: acceptance of the industrial-control class (\\%).}\n\\label{tab:reserve}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n & \\multicolumn{4}{c}{$R=0$} "
        f"& \\multicolumn{{4}}{{c}}{{$R={rstar:g}$}}\\\\\n\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}\n"
        "Method & Acc. & Power & IC & Gap & Acc. & Power & IC & Gap\\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return rstar


# ----------------------------------------------------------------------------- decision time (idle machine)
def decision_macros(tm: pd.DataFrame):
    for topo in TOPO:
        d = tm[tm.topology == topo]
        for m, g in d.groupby("method"):
            k = m.replace("-", "").replace(":", "")
            mac(f"resDec{TOPO_KEY[topo]}{k}", f"{g.decision_ms_mean.mean():.1f}")
            mac(f"resDec{TOPO_KEY[topo]}{k}Pn", f"{g.decision_ms_p95.mean():.1f}")
        e, s1 = d[d.method == "GNN-MAPPO-E"], d[d.method == "GNN-MAPPO"]
        if not e.empty and not s1.empty:
            mac(f"resDec{TOPO_KEY[topo]}EnsRatio", f"{e.decision_ms_mean.mean() / s1.decision_ms_mean.mean():.1f}")


# ----------------------------------------------------------------------------- figures on the new data
FIG_METHODS = ["GNN-MAPPO-E", "DQN-Lagrangian", "Greedy-Latency", "Greedy-Energy", "Best-Fit", "Greedy-Mix:0.3",
               "Greedy-Mix:1"]
FIG_STYLE = {"GNN-MAPPO-E": ("o", "C3", "-"), "GNN-MAPPO": ("o", "C5", "-"), "MLP-MAPPO": ("s", "C1", "-"), "DQN-Lagrangian": ("D", "C4", "-"),
             "Greedy-Latency": ("^", "C0", "--"), "Greedy-Energy": ("v", "C2", "--"), "Least-Loaded": ("<", "C7", "--"),
             "Best-Fit": (">", "C8", "--"), "Greedy-Mix:0.3": ("x", "0.45", ":"), "Greedy-Mix:1": ("+", "0.2", ":")}


def figures_r8(e1: pd.DataFrame, req: pd.DataFrame, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    topos = [t for t in TOPO if t in set(e1.topology)]
    metrics = [("acceptance", 100, "Acceptance (%)"), ("avg_power_w", 1, "Average power (W)"),
               ("energy_per_accepted_j", 1e-3, "Energy / accepted req. (kJ)")]
    fig, axes = plt.subplots(len(metrics), len(topos), figsize=(3.3 * len(topos), 2.4 * len(metrics)), squeeze=False)
    for j, t in enumerate(topos):
        dt = e1[e1.topology == t]
        for i, (mt, sc, lab) in enumerate(metrics):
            ax = axes[i][j]
            for m in [x for x in FIG_METHODS if x in set(dt.method)]:
                sm = dt[dt.method == m].groupby(["load", "seed"])[mt].mean().reset_index()
                g = sm.groupby("load")[mt]
                mu, se = g.mean() * sc, 1.96 * g.std() / np.sqrt(g.count()) * sc
                mk, c, ls = FIG_STYLE[m]
                ax.errorbar(mu.index, mu.values, yerr=se.values, color=c, marker=mk, ms=4, lw=1.2, ls=ls,
                            capsize=2, label=LABEL[m].replace("$", ""))
            if i == 0:
                ax.set_title(TOPO[t])
            if i == len(metrics) - 1:
                ax.set_xlabel(r"Offered load $\rho$")
            if j == 0:
                ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig_vs_load.pdf", bbox_inches="tight")
    plt.close(fig)
    if req.empty:
        return
    d = req[(req.topology == "fog-12e6f-s1") & np.isclose(req.load, 1.0) & req.accepted.astype(bool)]
    tight = {"industrial_control", "ar_offload"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (title, cls) in zip(axes, [("Tight budgets (industrial control, AR)", True),
                                         ("Loose budgets (e-health, video, metering)", False)]):
        dd = d[d.cls.isin(tight) == cls]
        for m in [x for x in FIG_METHODS if x in set(dd.method)]:
            v = np.sort((dd[dd.method == m].latency_ms / dd[dd.method == m].max_latency_ms).values)
            if len(v) == 0:
                continue
            mk, c, ls = FIG_STYLE[m]
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, marker=mk, markevery=max(1, len(v) // 6),
                    ms=4, lw=1.2, label=LABEL[m].replace("$", ""))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"Latency / budget $D(\mathbf{x}_r)/D_r$")
        ax.set_xlim(0, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CDF")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "fig_latency_cdf.pdf", bbox_inches="tight")
    plt.close(fig)
    for t in tight | {"ehealth"}:
        for m in ["GNN-MAPPO-E", "GNN-MAPPO", "Greedy-Latency", "Greedy-Mix:1", "Greedy-Energy"]:
            x = d[(d.cls == t) & (d.method == m)]
            if len(x):
                mac(f"resSlack{t.replace('_', '')}{m.replace('-', '').replace(':', '')}",
                    f"{100 * (1 - (x.latency_ms / x.max_latency_ms).median()):.0f}")
    d = req[(req.topology == "fog-12e6f-s1") & np.isclose(req.load, 1.25)]
    tab = d.groupby(["method", "cls"]).accepted.mean().unstack()
    tab = tab.loc[[m for m in FIG_METHODS if m in tab.index]]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.8 / len(tab)
    for i, m in enumerate(tab.index):
        mk, c, ls = FIG_STYLE[m]
        ax.bar(np.arange(tab.shape[1]) + i * w, 100 * tab.loc[m].values, width=w, color=c,
               label=LABEL[m].replace("$", ""), hatch="//" if ls == ":" else None, edgecolor="white", lw=0.3)
    ax.set_xticks(np.arange(tab.shape[1]) + 0.4 - w / 2)
    ax.set_xticklabels([c.replace("_", " ") for c in tab.columns])
    ax.set_ylim(50, 101)
    ax.set_ylabel("Acceptance (%)")
    ax.legend(frameon=False, ncol=4, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(figdir / "fig_class_acceptance.pdf", bbox_inches="tight")
    plt.close(fig)
    for m in tab.index:
        for c in tab.columns:
            mac(f"resCls{m.replace('-', '').replace(':', '')}{c.replace('_', '')}", f"{100 * tab.loc[m, c]:.1f}")


def load_requests(pattern: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(R / pattern))):
        args = json.loads((Path(f).parent / "infrastructure.json").read_text())
        d = pd.read_csv(f)
        d["topology"] = args.get("name")
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["method"] = df["method"].replace(RENAME)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "paper_v3" / "gen_r8"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e1 = load("E1/*/*_s*/summary.csv")
    e1.to_csv(R / "E1_merged_summary.csv", index=False)
    mac("resNumSeeds", str(e1.seed.nunique()))
    mac("resNumEvalRequests", f"{int(e1.requests.sum()):,}")
    tab_main(e1, out)
    gp_test = tab_frontier(e1, out)
    # single-seed DQN vs single-seed PPO on the IoT fog network, paired over the common training seeds
    gf = gp_test[gp_test.topology == "fog-12e6f-s1"].groupby(["method", "seed"]).gap.mean().unstack(0)
    if {"DQN-Lagrangian", "GNN-MAPPO"} <= set(gf.columns):
        dd = (100 * (gf["DQN-Lagrangian"] - gf["GNN-MAPPO"])).dropna()
        mac("resLrnDQNvsPPO", fmt(dd.mean(), 1))
        mac("resLrnDQNvsPPON", str(len(dd)))
        mac("resLrnDQNvsPPOP", fmtp(wil(dd.values)))
    sels = load("SELSEED/*/j*_*/summary.csv")
    if not sels.empty:
        tab_seeds(sels, gp_test, out)
    tab_loads(e1, out)
    tab_tier(e1, out)
    fig_frontier(e1, out / "figs")
    e0 = load("E0/*/s*/summary.csv")
    if not e0.empty:
        tab_sla(e0, e1, out)
    abl = load("E5_ablation/s*/summary.csv")
    if not abl.empty:
        tab_ablation(abl, load("E5_ablation_ens/*/summary.csv"), e1, out,
                     {"GNN-MAPPO": "GNN-MAPPO (GINE, path features, shared actor)", "GCN-PPO": "GCN encoder",
                      "GATv2-PPO": "GATv2 encoder", "MLP-MAPPO": "Node-wise MLP (no message passing)",
                      "GNN-basicfeat": "Without path features",
                      "GCN-basicfeat-sep": "GCN, basic features, one head per position",
                      "GNN-resmask": "Trained with CPU/RAM-only mask"})
    off = load("E5_noshield/s*/summary.csv")
    if not off.empty:
        tab_shield(off, e1, out)
    wtd = load("E5_weighted/*/summary.csv")
    tab_learner(e1, wtd, out)
    sel = load("SEL_fog_learners/summary.csv")
    if not sel.empty:
        sel_learners(sel)
    e4 = load("E4/*/s*/summary.csv")
    if not e4.empty:
        tab_zeroshot(e4, e1, out)
    ilp = load("E1_ilp/*/s*/summary.csv")
    if not ilp.empty:
        tab_ilp(ilp, e1, out)
    selr = load("SEL_reserve/*/summary.csv")
    if not selr.empty:
        tab_reserve(selr, load("SEL_reserve_ens/*/summary.csv"), load("E1R/*_s*/summary.csv"), e1, out)
    tm = load("TIMING/*/summary.csv")
    decision_macros(tm if not tm.empty else e1)
    req = load_requests("E1/*/*_s*/requests.csv.gz")
    figures_r8(e1, req, out / "figs")
    # Shrink any table that is wider than the column (adjustbox only scales down).
    for tf in out.glob("tab_*.tex"):
        t = tf.read_text()
        if "adjustbox" not in t:
            t = t.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}")
            t = t.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
            tf.write_text(t)
    # Text forms of p-values: \\fooPt expands to "<0.001" or "=0.023", for use as $p\\fooPt$ in running text.
    import re as _re
    for k, v in list(MACROS.items()):
        if k.endswith("P") and (v == "$<$0.001" or _re.fullmatch(r"[01]\.\d{3}", v) or v == "--"):
            MACROS[k + "t"] = "<0.001" if v == "$<$0.001" else ("=" + v if v != "--" else "=\\text{--}")
    (out / "macros_r8.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))
    (out / "macros_r8.json").write_text(json.dumps(MACROS, indent=1, ensure_ascii=False))
    print(f"{len(MACROS)} macros, tables in {out}")


if __name__ == "__main__":
    main()
