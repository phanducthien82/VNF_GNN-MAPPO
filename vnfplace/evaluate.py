"""Benchmark harness: every method on identical traces, per-request records."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from .baselines import HEURISTICS, make_heuristic
from .env import EnvConfig, FogEnv, run_policy
from .topology import Infrastructure
from .traffic import arrival_rate, generate_trace


TEST_TRACE_BASE = 9_000_000
SELECTION_TRACE_BASE = 7_500_000   # traces for selecting deployment parameters (e.g. the reservation share)


def test_trace_seed(seed: int, load: float, base: int = TEST_TRACE_BASE) -> int:
    """Trace seed for evaluation seed ``seed``.

    The test traces never coincide with any training (2e7+1e5*s+episode) or validation
    (5e6+1000*s+i) traces of the policy trained with seed s, which is the policy evaluated on them.
    ``SELECTION_TRACE_BASE`` gives a further stream, disjoint from all training, validation and test
    traces, for tuning deployment parameters without touching the test traces.
    """
    return base + 1000 * int(seed) + int(round(load * 100))


def summarise(rows: list, summ: dict) -> dict:
    df = pd.DataFrame(rows)
    acc = df[df.accepted]
    out = {k: v for k, v in summ.items() if k != "reject_reasons"}
    out.update({
        "latency_mean_ms": float(acc.latency_ms.mean()) if len(acc) else np.nan,
        "latency_p95_ms": float(acc.latency_ms.quantile(0.95)) if len(acc) else np.nan,
        "latency_p99_ms": float(acc.latency_ms.quantile(0.99)) if len(acc) else np.nan,
        "sla_slack_mean": float((1 - acc.latency_ms / acc.max_latency_ms).mean()) if len(acc) else np.nan,
        "inc_power_mean_w": float(acc.inc_power_w.mean()) if len(acc) else np.nan,
        "decision_ms_mean": float(df.decision_ms.mean()),
        "decision_ms_p95": float(df.decision_ms.quantile(0.95)),
    })
    for k, v in summ["reject_reasons"].items():
        out[f"rej_{k}"] = v / max(1, summ["requests"])
    for c, g in df.groupby("cls"):
        out[f"acc_{c}"] = float(g.accepted.mean())
    return out


def run_benchmark(infra: Infrastructure, env_cfg: EnvConfig, methods: Dict[str, Callable[[int], Callable]],
                  loads: Iterable[float], seeds: Iterable[int], n_requests: int, warmup: int,
                  mean_holding_s: float = 300.0, out_dir: Path | None = None,
                  keep_requests: bool = True, verbose: bool = True,
                  trace_base: int = TEST_TRACE_BASE, warmup_holdings: float = 0.0) -> pd.DataFrame:
    """``methods``: name -> factory(seed) returning a ``choose(env, cand)`` callable."""
    summaries, req_frames = [], []
    for load in loads:
        for seed in seeds:
            if warmup_holdings > 0:
                # Time-based warm-up: every request arriving before warmup_holdings mean holding times
                # is warm-up; the next (n_requests - warmup) requests are measured.
                t_w = warmup_holdings * mean_holding_s
                n_w = int(np.ceil(arrival_rate(infra, load, mean_holding_s) * t_w * 1.3)) + 50
                full = generate_trace(infra, n_w + n_requests - warmup, load,
                                      test_trace_seed(seed, load, trace_base), mean_holding_s)
                w_idx = next(i for i, r in enumerate(full) if r.t_arrival >= t_w)
                trace, w_eff = full[:w_idx + n_requests - warmup], w_idx
            else:
                trace = generate_trace(infra, n_requests, load, test_trace_seed(seed, load, trace_base), mean_holding_s)
                w_eff = warmup
            for name, factory in methods.items():
                env = FogEnv(infra, env_cfg)
                t0 = time.time()
                rows, summ = run_policy(env, trace, factory(seed), warmup=w_eff)
                s = summarise(rows, summ)
                s.update(method=name, load=load, seed=seed, topology=infra.name, wall_s=time.time() - t0)
                summaries.append(s)
                if keep_requests:
                    df = pd.DataFrame(rows)
                    df["method"], df["load"], df["seed"] = name, load, seed
                    req_frames.append(df)
                if verbose:
                    print(f"[{infra.name} load={load} seed={seed}] {name:18s} acc={s['acceptance']:.3f} "
                          f"P={s['avg_power_w']:.0f}W E/acc={s['energy_per_accepted_j']:.0f}J "
                          f"lat={s['latency_mean_ms']:.1f}ms dec={s['decision_ms_mean']:.2f}ms", flush=True)
    res = pd.DataFrame(summaries)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_dir / "summary.csv", index=False)
        if keep_requests and req_frames:
            pd.concat(req_frames).to_csv(out_dir / "requests.csv.gz", index=False)
    return res


def heuristic_factories(names: Iterable[str] | None = None) -> Dict[str, Callable]:
    names = list(names) if names else list(HEURISTICS)
    return {n: (lambda s, n=n: make_heuristic(n, seed=s)) for n in names}
