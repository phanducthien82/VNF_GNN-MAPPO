#!/usr/bin/env python3
"""Command-line entry point.

    # train one policy (one seed)
    python scripts/run.py train --topo fog --topo-seed 1 --encoder gine --seed 0 --out runs/fog/gine_s0

    # benchmark heuristics, ILP and trained policies on identical traces
    python scripts/run.py bench --topo fog --topo-seed 1 --loads 0.5 0.75 1.0 1.25 1.5 \
        --seeds 0 1 2 3 4 5 6 7 8 9 --rl "GNN-PPO=runs/fog/gine_s{seed}" --ilp --out results/fog

    # statistics and figures
    python scripts/run.py stats --results results/fog --reference GNN-PPO
    python scripts/run.py plots --results results/fog

``{seed}`` in an --rl path is replaced by the evaluation seed, so seed k of the
benchmark uses the policy trained with seed k (matched seeds).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vnfplace.env import EnvConfig  # noqa: E402
from vnfplace.evaluate import heuristic_factories, run_benchmark  # noqa: E402
from vnfplace.ppo import EnsembleChooser, RLChooser, RewardConfig, TrainConfig, load_policy, train  # noqa: E402
from vnfplace.topology import build_infrastructure  # noqa: E402


def env_cfg_from(a) -> EnvConfig:
    return EnvConfig(u_max=a.u_max, sleep_frac=a.sleep_frac, link_nj_per_bit=a.link_nj,
                     mask_mode=a.mask, track_sla_impact=getattr(a, "sla_impact", False),
                     shield=getattr(a, "shield", False), reserve=getattr(a, "reserve", 0.0),
                     reserve_budget_ms=getattr(a, "reserve_budget", 30.0),
                     protect_sla=getattr(a, "protect_sla", False))


def add_common(p):
    p.add_argument("--topo", default="fog", help="fog | fog:<edge>x<fog> | real:<name>")
    p.add_argument("--topo-seed", type=int, default=1)
    p.add_argument("--u-max", type=float, default=0.9)
    p.add_argument("--sleep-frac", type=float, default=0.0)
    p.add_argument("--link-nj", type=float, default=10.0)
    p.add_argument("--mask", default="full", choices=["lookahead", "full", "resource", "none"])
    p.add_argument("--holding", type=float, default=300.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--shield", action="store_true", help="lazy completion shield for every method")
    p.add_argument("--reserve", type=float, default=0.0,
                   help="CPU share of edge/fog nodes reserved for latency-critical requests (trunk reservation)")
    p.add_argument("--protect-sla", action="store_true",
                   help="admit a request only if no running service exceeds its latency budget afterwards")
    p.add_argument("--reserve-budget", type=float, default=30.0,
                   help="latency budget (ms) at or below which a request counts as latency-critical")


def cmd_train(a):
    infra = build_infrastructure(a.topo, seed=a.topo_seed)
    rc = RewardConfig(acc=a.r_acc, rej=a.r_rej, beta_energy=a.beta, eta_latency=a.eta,
                      lagrangian_target=a.lagrangian, lagrangian_init=a.lambda_init)
    cfg = TrainConfig(encoder=a.encoder, actor=a.actor, hidden=a.hidden, layers=a.layers,
                      feature_set=a.features, total_steps=a.steps, rollout_steps=a.rollout,
                      lr=a.lr, gamma=a.gamma, seed=a.seed, init_from=a.init_from,
                      continuing=a.continuing, val_warmup_holdings=a.val_warmup_holdings, train_loads=tuple(a.train_loads), mean_holding_s=a.holding,
                      reward=rc)
    if a.learner == "dqn":
        from vnfplace.dqn import DQNConfig, train_dqn
        dq = DQNConfig(buffer=a.buffer, batch=a.batch, train_every=a.train_every, learn_start=a.learn_start,
                       eps_frac=a.eps_frac, n_step=a.n_step)
        train_dqn(infra, env_cfg_from(a), cfg, dq, Path(a.out))
        return
    train(infra, env_cfg_from(a), cfg, Path(a.out))


def cmd_bench(a):
    infra = build_infrastructure(a.topo, seed=a.topo_seed)
    env_cfg = env_cfg_from(a)
    methods = {} if a.heuristics == ["none"] else heuristic_factories(a.heuristics or None)
    if a.ilp:
        from vnfplace.ilp import PerRequestILP
        methods["ILP-PerRequest"] = lambda s: PerRequestILP(time_limit_s=a.ilp_time)
    for spec in a.rl or []:
        name, path = spec.split("=", 1)

        def factory(s, path=path):
            # "ens:<dir>,<dir>,..." averages the policies of several runs (glob patterns allowed);
            # "<dir>@last.pt" loads the final instead of the selected checkpoint.
            if path.startswith("ens:"):
                dirs = [d for pat in path[4:].split(",") for d in sorted(glob.glob(pat.format(seed=s)))]
                loaded = [load_policy(infra, env_cfg, Path(d)) for d in dirs]
                return EnsembleChooser([m for m, _ in loaded], loaded[0][1].feature_set)
            d, _, which = path.format(seed=s).partition("@")
            model, cfg = load_policy(infra, env_cfg, Path(d), which or "best.pt")
            return RLChooser(model, cfg.feature_set)
        methods[name] = factory
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=2))
    (out / "infrastructure.json").write_text(json.dumps(infra.summary(), indent=2))
    run_benchmark(infra, env_cfg, methods, a.loads, a.seeds, a.requests, a.warmup,
                  mean_holding_s=a.holding, out_dir=out, trace_base=a.trace_base,
                  warmup_holdings=a.warmup_holdings)


def cmd_stats(a):
    import pandas as pd
    from vnfplace import stats as S
    frames = [pd.read_csv(Path(r) / "summary.csv") for r in a.results]
    df = pd.concat(frames, ignore_index=True)
    out = Path(a.out or a.results[0])
    metrics = ["acceptance", "avg_power_w", "energy_per_accepted_j", "avg_active_nodes",
               "latency_mean_ms", "latency_p95_ms", "decision_ms_mean"]
    hib = {"acceptance": True, "avg_power_w": False, "energy_per_accepted_j": False,
           "avg_active_nodes": False, "latency_mean_ms": False, "latency_p95_ms": False,
           "decision_ms_mean": False}
    S.summary_table(df, metrics, by=["topology", "load", "method"]).to_csv(out / "table_by_load.csv", index=False)
    S.summary_table(df, metrics, by=["topology", "method"]).to_csv(out / "table_overall.csv", index=False)
    if a.reference:
        pt = S.paired_vs_reference(df, a.reference, metrics, higher_is_better=hib)
        pt.to_csv(out / "paired_tests.csv", index=False)
        print(pt[["metric", "other", "n_pairs", "mean_diff", "ref_better", "wilcoxon_p_holm", "cliffs_delta"]]
              .to_string(index=False))
    fr = [S.friedman_nemenyi(df, m, hib[m]) for m in ["acceptance", "avg_power_w", "energy_per_accepted_j"]]
    (out / "friedman_nemenyi.json").write_text(json.dumps(fr, indent=2))
    print(json.dumps(fr, indent=2))


def cmd_plots(a):
    from vnfplace.plots import make_all
    make_all(Path(a.results), Path(a.out or Path(a.results) / "figures"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    add_common(t)
    t.add_argument("--encoder", default="gine", choices=["mlp", "gcn", "gine", "gatv2"])
    t.add_argument("--actor", default="shared", choices=["shared", "separate"])
    t.add_argument("--features", default="path", choices=["path", "basic"])
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--steps", type=int, default=300_000)
    t.add_argument("--rollout", type=int, default=4096)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--init-from", default="", help="state_dict to warm-start from (continued training)")
    t.add_argument("--lambda-init", type=float, default=3.0)
    t.add_argument("--continuing", action="store_true",
                   help="train on one continuing task (no substrate reset between episodes, no terminal at episode ends)")
    t.add_argument("--val-warmup-holdings", type=float, default=0.0,
                   help="time-based validation warm-up in mean holding times (0: 100 requests)")
    t.add_argument("--train-loads", type=float, nargs=2, default=[0.5, 1.5])
    t.add_argument("--r-acc", type=float, default=1.0)
    t.add_argument("--r-rej", type=float, default=1.0)
    t.add_argument("--beta", type=float, default=1.0, help="energy weight")
    t.add_argument("--eta", type=float, default=0.1, help="latency weight")
    t.add_argument("--lagrangian", type=float, default=None, help="target acceptance for PPO-Lagrangian")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--learner", default="ppo", choices=["ppo", "dqn"])
    t.add_argument("--buffer", type=int, default=50_000, help="DQN replay size")
    t.add_argument("--batch", type=int, default=64, help="DQN batch size")
    t.add_argument("--train-every", type=int, default=4, help="DQN environment steps per gradient step")
    t.add_argument("--learn-start", type=int, default=5_000, help="DQN steps before learning")
    t.add_argument("--eps-frac", type=float, default=0.3, help="DQN share of training with decaying epsilon")
    t.add_argument("--n-step", type=int, default=3, help="DQN n-step return")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    b = sub.add_parser("bench")
    add_common(b)
    b.add_argument("--loads", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    b.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    b.add_argument("--requests", type=int, default=2000)
    b.add_argument("--warmup", type=int, default=300)
    b.add_argument("--heuristics", nargs="*", default=None)
    b.add_argument("--ilp", action="store_true")
    b.add_argument("--ilp-time", type=float, default=2.0)
    b.add_argument("--rl", nargs="*", help="NAME=run_dir (use {seed} for matched seeds)")
    b.add_argument("--sla-impact", action="store_true")
    b.add_argument("--warmup-holdings", type=float, default=0.0,
                   help="time-based warm-up in mean holding times (0: use --warmup requests)")
    b.add_argument("--trace-base", type=int, default=9_000_000,
                   help="trace stream: 9000000 = test traces (default), 7500000 = selection traces for tuning")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("stats")
    s.add_argument("--results", nargs="+", required=True)
    s.add_argument("--reference", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_stats)

    pl = sub.add_parser("plots")
    pl.add_argument("--results", required=True)
    pl.add_argument("--out", default=None)
    pl.set_defaults(fn=cmd_plots)

    a = p.parse_args()
    if hasattr(a, "threads"):
        torch.set_num_threads(a.threads)
    a.fn(a)


if __name__ == "__main__":
    main()
