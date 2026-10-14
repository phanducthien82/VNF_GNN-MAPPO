"""Statistical analysis for multi-seed, multi-scenario RL benchmarks.

* Bootstrap confidence intervals and the interquartile mean (IQM) over runs
  (Agarwal et al., NeurIPS 2021).
* Paired comparisons of a reference method against every other method on
  matched blocks (topology, load, seed): Wilcoxon signed-rank and paired t,
  Holm-Bonferroni correction, Hedges' g_z and Cliff's delta effect sizes.
* Friedman test + Nemenyi critical difference across methods (Demsar, JMLR 2006).
* Minimum attainable two-sided Wilcoxon p-value for n pairs (2 / 2^n), to
  document why n = 5 seeds can never reach p < 0.05.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.multitest import multipletests

BLOCK = ["topology", "load", "seed"]

# Nemenyi critical values q_alpha (alpha = 0.05), Demsar (2006), Table 5(a).
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def min_wilcoxon_p(n: int) -> float:
    return 2.0 / (2 ** n)


def iqm(x: Sequence[float]) -> float:
    return float(st.trim_mean(np.asarray(x, dtype=float), 0.25))


def bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def hedges_gz(d: Sequence[float]) -> float:
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan
    j = 1 - 3 / (4 * (n - 1) - 1)
    return float(j * d.mean() / d.std(ddof=1))


def summary_table(df: pd.DataFrame, metrics: List[str], by: List[str] = ("method",)) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_runs"] = len(g)
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].values)
            r[f"{m}_mean"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = mean, lo, hi
            r[f"{m}_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
            r[f"{m}_iqm"] = iqm(g[m].values)
        rows.append(r)
    return pd.DataFrame(rows)


def paired_vs_reference(df: pd.DataFrame, reference: str, metrics: Iterable[str],
                        block: List[str] = BLOCK, higher_is_better: dict | None = None) -> pd.DataFrame:
    """Paired tests of ``reference`` against every other method; Holm over all tests."""
    higher_is_better = higher_is_better or {}
    wide = {m: df.pivot_table(index=block, columns="method", values=m) for m in metrics}
    rows = []
    for m, w in wide.items():
        if reference not in w:
            continue
        for other in w.columns:
            if other == reference:
                continue
            pair = w[[reference, other]].dropna()
            a, b = pair[reference].values, pair[other].values
            d = a - b
            n = len(d)
            if n < 2:
                continue
            try:
                wp = float(st.wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                wp = np.nan
            tp = float(st.ttest_rel(a, b).pvalue) if np.std(d) > 0 else 1.0
            better = (d.mean() > 0) == higher_is_better.get(m, True)
            rows.append({"metric": m, "reference": reference, "other": other, "n_pairs": n,
                         "ref_mean": a.mean(), "other_mean": b.mean(), "mean_diff": d.mean(),
                         "ref_better": bool(better), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                         "wilcoxon_p": wp, "ttest_p": tp, "hedges_gz": hedges_gz(d),
                         "cliffs_delta": cliffs_delta(a, b), "min_attainable_wilcoxon_p": min_wilcoxon_p(n)})
    res = pd.DataFrame(rows)
    if len(res):
        res["wilcoxon_p_holm"] = multipletests(res["wilcoxon_p"].fillna(1.0), method="holm")[1]
        res["ttest_p_holm"] = multipletests(res["ttest_p"].fillna(1.0), method="holm")[1]
        res["significant_0.05"] = res["wilcoxon_p_holm"] < 0.05
    return res


def friedman_nemenyi(df: pd.DataFrame, metric: str, higher_is_better: bool = True,
                     block: List[str] = BLOCK) -> dict:
    w = df.pivot_table(index=block, columns="method", values=metric).dropna()
    k, n = w.shape[1], w.shape[0]
    ranks = (-w if higher_is_better else w).rank(axis=1, method="average")
    avg = ranks.mean().sort_values()
    stat, p = st.friedmanchisquare(*[w[c].values for c in w.columns]) if k >= 3 else (np.nan, np.nan)
    q = NEMENYI_Q05.get(k, np.nan)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n)) if n else np.nan
    return {"metric": metric, "k": k, "n_blocks": n, "friedman_chi2": float(stat), "friedman_p": float(p),
            "nemenyi_cd": float(cd), "avg_ranks": avg.to_dict()}


def pareto_front(points: pd.DataFrame, x: str, y: str, minimise_x=True, maximise_y=True) -> pd.Series:
    """Boolean mask of non-dominated rows (e.g. x = power, y = acceptance)."""
    X = points[x].values * (1 if minimise_x else -1)
    Y = points[y].values * (-1 if maximise_y else 1)
    nd = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        dom = (X <= X[i]) & (Y <= Y[i]) & ((X < X[i]) | (Y < Y[i]))
        nd[i] = not dom.any()
    return pd.Series(nd, index=points.index)


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """2-D hypervolume for minimisation of both columns w.r.t. reference point."""
    p = points[(points[:, 0] <= ref[0]) & (points[:, 1] <= ref[1])]
    if len(p) == 0:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    # Sweep in increasing x: each new non-dominated point adds a rectangle.
    hv, prev_y = 0.0, ref[1]
    for px, py in p:
        if py < prev_y:
            hv += (ref[0] - px) * (prev_y - py)
            prev_y = py
    return float(hv)
