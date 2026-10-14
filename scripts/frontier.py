#!/usr/bin/env python3
"""Power at equal acceptance against the frontier of tuned heuristics.

For every block (topology, load, seed) the heuristic points (acceptance, power)
of the reference family (Greedy-Mix:w for all w, plus Greedy-Energy and
Greedy-Latency as its end points) are reduced to their Pareto front, and the
front is interpolated linearly in acceptance (a time-sharing mixture of two
neighbouring rules). A method's gap is

    gap = (P_method - P_front(acc_method)) / P_front(acc_method),

negative when the method needs less power than the best tuned heuristic that
reaches the same acceptance. A method with a higher acceptance than every
heuristic point is compared with the highest-acceptance point (conservative).

    python scripts/frontier.py --results results/E1_fog ... --methods GNN-MAPPO MLP-MAPPO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FAMILY_DEFAULT = ("Greedy-Energy", "Greedy-Latency")


def pareto(points: np.ndarray) -> np.ndarray:
    """Rows (acc, power) not dominated (higher acc and lower power are better), sorted by acc."""
    pts = points[np.lexsort((points[:, 1], -points[:, 0]))]      # acc desc, power asc
    keep, best_p = [], np.inf
    for a, p in pts:
        if p < best_p - 1e-9:
            keep.append((a, p))
            best_p = p
    return np.array(sorted(keep))


def front_power(front: np.ndarray, acc: float) -> float:
    if acc >= front[-1, 0]:
        return float(front[-1, 1])
    if acc <= front[0, 0]:
        return float(front[0, 1])
    return float(np.interp(acc, front[:, 0], front[:, 1]))


def gaps(df: pd.DataFrame, methods, family_prefix="Greedy-Mix:", family_extra=FAMILY_DEFAULT) -> pd.DataFrame:
    rows = []
    for (topo, load, seed), g in df.groupby(["topology", "load", "seed"]):
        fam = g[g.method.str.startswith(family_prefix) | g.method.isin(family_extra)]
        if len(fam) < 2:
            continue
        front = pareto(fam[["acceptance", "avg_power_w"]].to_numpy(float))
        for m in methods:
            r = g[g.method == m]
            if r.empty:
                continue
            a, p = float(r.acceptance.iloc[0]), float(r.avg_power_w.iloc[0])
            pf = front_power(front, a)
            rows.append(dict(topology=topo, load=load, seed=seed, method=m, acceptance=a, power=p,
                             front_power=pf, gap=(p - pf) / pf, beyond=a > front[-1, 0]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(Path(r) / "summary.csv") for r in a.results], ignore_index=True)
    gp = gaps(df, a.methods)
    if a.out:
        gp.to_csv(a.out, index=False)
    for (topo, m), g in gp.groupby(["topology", "method"]):
        per_seed = g.groupby("seed").gap.mean()
        p = wilcoxon(per_seed).pvalue if len(per_seed) >= 5 and per_seed.abs().sum() > 0 else float("nan")
        print(f"{topo:20s} {m:14s} blocks={len(g):3d} gap={100 * g.gap.mean():+6.2f}% "
              f"(seed-level median {100 * per_seed.median():+6.2f}%, p={p:.3g}) "
              f"better_in={100 * (g.gap < 0).mean():.0f}% beyond_front={100 * g.beyond.mean():.0f}%")


if __name__ == "__main__":
    main()
