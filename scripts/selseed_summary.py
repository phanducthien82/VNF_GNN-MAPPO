#!/usr/bin/env python3
"""Frontier gaps of single seeds and seed ensembles on the selection stream (results_v3/SELSEED)."""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier import gaps  # noqa: E402

rows = []
for t in ["fog", "real_nobel_germany", "real_geant2012"]:
    for j in (0, 1):
        fs = glob.glob(f"results_v3/SELSEED/{t}/j{j}_*/summary.csv")
        if not any("heur" in f for f in fs) or len(fs) < 2:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs])
        rl = [m for m in d.method.unique() if m.startswith("GNN")]
        g = gaps(d, rl)
        g["gap"] *= 100
        rows.append(g)
g = pd.concat(rows)
for t, gt in g.groupby("topology"):
    s = gt.groupby("method").agg(gap=("gap", "mean"), acc=("acceptance", "mean"), n=("gap", "size"))
    s["gap_rho_ge1"] = gt[gt.load >= 1.0].groupby("method").gap.mean()
    print(t)
    print(s.sort_values("gap").round(2).to_string())
