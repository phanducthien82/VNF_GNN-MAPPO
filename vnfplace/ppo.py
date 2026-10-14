"""PPO training of the autoregressive placement policy.

MDP: one decision step per VNF. The reward is zero inside a chain and is paid
on the step that completes (accept) or fails (reject) the request:

    accept:  r = lam_acc - beta * sum(marginal power)/P_ref - eta * latency/D_max
    reject:  r = -lam_rej

The objective therefore trades acceptance against energy. With
``lagrangian_target`` set, lam_acc = lam_rej = lambda is a Lagrange
multiplier updated by dual ascent so that the acceptance ratio meets the
target while energy is minimised (PPO-Lagrangian).

Training uses a fixed substrate, fresh request traces per episode with the
offered load sampled uniformly from ``train_loads`` (one model for all loads),
GAE, value normalisation, minibatch PPO with clipping, linear LR / entropy
annealing, and checkpoint selection on a held-out validation trace.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from .env import EnvConfig, FogEnv, run_policy
from .policy import PlacementPolicy
from .topology import Infrastructure
from .traffic import generate_trace, trace_with_warmup


@dataclass
class RewardConfig:
    acc: float = 1.0
    rej: float = 1.0
    beta_energy: float = 1.0
    eta_latency: float = 0.1
    lagrangian_target: Optional[float] = None   # e.g. 0.97 -> PPO-Lagrangian on acceptance
    lagrangian_lr: float = 2.0
    lagrangian_init: float = 3.0
    lagrangian_max: float = 20.0


@dataclass
class TrainConfig:
    encoder: str = "gine"
    actor: str = "shared"
    hidden: int = 64
    layers: int = 3
    feature_set: str = "path"
    total_steps: int = 300_000
    rollout_steps: int = 4096
    minibatch: int = 512
    epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    grad_clip: float = 0.5
    episode_requests: int = 400
    train_loads: Tuple[float, float] = (0.5, 1.5)
    mean_holding_s: float = 300.0
    val_requests: int = 600
    val_warmup: int = 100
    val_loads: Tuple[float, ...] = (0.75, 1.25)
    val_every: int = 5               # updates
    seed: int = 0
    init_from: str = ""              # warm start: path of a state_dict (continued training)
    continuing: bool = False         # one continuing task: no reset between episodes, bootstrap across them
    val_warmup_holdings: float = 0.0  # >0: time-based validation warm-up (mean holding times)
    reward: RewardConfig = field(default_factory=RewardConfig)


class ValueNorm:
    def __init__(self, beta=0.995):
        self.m, self.v, self.beta, self.w = 0.0, 1.0, beta, 0.0

    def update(self, x: torch.Tensor):
        self.w = self.beta * self.w + (1 - self.beta)
        bm = float(x.mean())
        bv = float(x.var(unbiased=False))
        self.m = self.beta * self.m + (1 - self.beta) * bm
        self.v = self.beta * self.v + (1 - self.beta) * (bv + bm * bm)

    @property
    def mean(self):
        return self.m / max(self.w, 1e-8)

    @property
    def std(self):
        return float(np.sqrt(max(self.v / max(self.w, 1e-8) - self.mean ** 2, 1e-6)))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


def make_policy(env: FogEnv, cfg: TrainConfig) -> PlacementPolicy:
    return PlacementPolicy(FogEnv.NODE_FEATURES, FogEnv.EDGE_FEATURES, FogEnv.CTX_FEATURES,
                           encoder=cfg.encoder, hidden=cfg.hidden, layers=cfg.layers, actor=cfg.actor)


def obs_tensors(env: FogEnv, cand, feature_set: str):
    x, ef, ctx = env.observe(cand, feature_set)
    return (torch.from_numpy(x).unsqueeze(0), torch.from_numpy(ef).unsqueeze(0),
            torch.from_numpy(ctx).unsqueeze(0), torch.from_numpy(cand.mask.copy()).unsqueeze(0),
            torch.tensor([env.j]))


class RLChooser:
    """Wraps a trained policy as a ``choose(env, cand)`` callable for run_policy."""

    def __init__(self, model: PlacementPolicy, feature_set: str, deterministic: bool = True):
        self.model, self.fs, self.det = model.eval(), feature_set, deterministic
        self.ei = None

    @torch.no_grad()
    def __call__(self, env: FogEnv, cand):
        if self.ei is None:
            self.ei = torch.from_numpy(env.edge_index.copy())
        x, ef, ctx, m, pos = obs_tensors(env, cand, self.fs)
        logits, _ = self.model(x, ef, ctx, m, pos, self.ei)
        if self.det:
            return int(logits.argmax(-1))
        return int(Categorical(logits=logits).sample())


class EnsembleChooser:
    """Averages the action distributions of several trained policies and takes the most likely node."""

    def __init__(self, models, feature_set: str):
        self.models, self.fs = [m.eval() for m in models], feature_set
        self.ei = None
        self._vm = None
        try:  # evaluate all members in one vectorised call when torch.func supports the encoder
            import copy
            from torch.func import functional_call, stack_module_state
            params, buffers = stack_module_state(self.models)
            base = copy.deepcopy(self.models[0]).to("meta")

            def one(p, b, *args):
                return torch.softmax(functional_call(base, (p, b), args)[0], -1)
            self._vm = (torch.vmap(one, in_dims=(0, 0) + (None,) * 6), params, buffers)
        except Exception:
            self._vm = None

    @torch.no_grad()
    def __call__(self, env: FogEnv, cand):
        if self.ei is None:
            self.ei = torch.from_numpy(env.edge_index.copy())
        x, ef, ctx, m, pos = obs_tensors(env, cand, self.fs)
        if self._vm is not None:
            try:
                f, params, buffers = self._vm
                return int(f(params, buffers, x, ef, ctx, m, pos, self.ei).sum(0).argmax(-1))
            except Exception:
                self._vm = None
        prob = sum(torch.softmax(mod(x, ef, ctx, m, pos, self.ei)[0], -1) for mod in self.models)
        return int(prob.argmax(-1))


def request_reward(rc: RewardConfig, lam: float, accepted: bool, inc_power: float,
                   latency: float, max_latency: float, p_ref: float) -> float:
    if not accepted:
        return -(lam if rc.lagrangian_target is not None else rc.rej)
    acc = lam if rc.lagrangian_target is not None else rc.acc
    return acc - rc.beta_energy * inc_power / p_ref - rc.eta_latency * latency / max_latency


def score_rows(rows, rc: RewardConfig, p_ref: float) -> float:
    """Mean per-request objective (same as the training reward with lambda=1)."""
    s = 0.0
    for r in rows:
        if r["accepted"]:
            s += rc.acc - rc.beta_energy * r["inc_power_w"] / p_ref - rc.eta_latency * r["latency_ms"] / r["max_latency_ms"]
        else:
            s -= rc.rej
    return s / max(1, len(rows))


class Collector:
    def __init__(self, infra: Infrastructure, env_cfg: EnvConfig, cfg: TrainConfig):
        self.infra, self.cfg = infra, cfg
        self.env = FogEnv(infra, env_cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.episode = 0
        self.ei = torch.from_numpy(self.env.edge_index.copy())
        rc = cfg.reward
        self.lam = rc.lagrangian_init if rc.lagrangian_target is not None else 1.0
        self.ep_acc: List[float] = []
        self._new_episode()

    def _new_episode(self):
        cfg = self.cfg
        load = float(self.rng.uniform(*cfg.train_loads))
        # Training stream 2e7 + 1e5*seed + episode: disjoint from the validation (5e6 + 1000*seed + i),
        # selection (7.5e6 + ...) and test (9e6 + ...) streams for every seed.
        trace_seed = 20_000_000 + 100_000 * cfg.seed + self.episode
        self.trace = generate_trace(self.infra, cfg.episode_requests, load, trace_seed, cfg.mean_holding_s)
        self.ti = 0
        if cfg.continuing and self.episode > 0:
            # Continue from the current occupancy: shift arrival times and request ids of the new segment.
            t0, base = self.env.t, self.rid_base
            for r in self.trace:
                r.t_arrival += t0
                r.rid += base
        else:
            self.env.reset()
            self.rid_base = 0
        self.rid_base += len(self.trace)
        self.episode += 1
        self.n_acc = self.n_req = 0

    def _next_decision(self):
        """Advance to the next decision with a non-empty mask. Returns (cand, episode_ended)."""
        env, ended = self.env, False
        while True:
            if env.req is None:
                if self.ti >= len(self.trace):
                    self.ep_acc.append(self.n_acc / max(1, self.n_req))
                    self._new_episode()
                    ended = True
                env.begin(self.trace[self.ti])
                self.ti += 1
                self.n_req += 1
                self.inc_p = 0.0
            cand = env.candidates()
            if cand.mask.any():
                return cand, ended
            env.abort("no_candidate" if env.j == 0 else "dead_end")
            if self.open_idx is not None:
                self._close(False)

    def _shield(self, cand, m, logits, dist, a, max_tries: int = 8):
        """Completion shield inside the rollout (same rule as run_policy).

        Refused nodes are removed from the stored mask, so that PPO later
        evaluates the executed action under exactly the distribution it was
        sampled from. A repair that the policy did not propose is stored with
        a one-hot mask (log-probability 0, no gradient)."""
        env = self.env
        if env.j == env.req.length - 1:
            return m, dist, a
        mask = m.clone()
        for _ in range(max_tries):
            n = int(a)
            if env._completes(n):
                return mask, dist, a
            mask[0, n] = False
            if not mask.any():
                return m, Categorical(logits=logits), torch.tensor([int(a)])
            dist = Categorical(logits=logits.masked_fill(~mask, -1e9))
            a = dist.sample()
        order = np.argsort(np.where(mask[0].numpy(), cand.latency_lb, np.inf), kind="stable")
        for k in order:
            if not mask[0, k]:
                break
            if env._completes(int(k)):
                one = torch.zeros_like(mask)
                one[0, int(k)] = True
                return one, Categorical(logits=logits.masked_fill(~one, -1e9)), torch.tensor([int(k)])
        return mask, dist, a

    def _close(self, accepted: bool, latency: float = 0.0, max_lat: float = 1.0):
        rc = self.cfg.reward
        self.rew[self.open_idx] = request_reward(rc, self.lam, accepted, self.inc_p, latency, max_lat, self.env.p_ref)
        self.n_acc += int(accepted)
        self.open_idx = None

    @torch.no_grad()
    def collect(self, model: PlacementPolicy, n_steps: int):
        fs = self.cfg.feature_set
        X, EF, C, M, P, A, LP, V = [], [], [], [], [], [], [], []
        self.rew = np.zeros(n_steps, dtype=np.float32)
        done = np.zeros(n_steps, dtype=np.float32)
        self.open_idx = None
        cand, _ = self._next_decision()
        for t in range(n_steps):
            x, ef, ctx, m, pos = obs_tensors(self.env, cand, fs)
            logits, v = model(x, ef, ctx, m, pos, self.ei)
            dist = Categorical(logits=logits)
            a = dist.sample()
            if self.env.cfg.shield:
                m, dist, a = self._shield(cand, m, logits, dist, a)
            X.append(x); EF.append(ef); C.append(ctx); M.append(m); P.append(pos)
            A.append(a); LP.append(dist.log_prob(a)); V.append(v)
            n = int(a)
            self.inc_p += float(cand.inc_power[n])
            self.open_idx = t
            req = self.env.req
            status = self.env.place(n)
            if status == "accept":
                self._close(True, self.env.last_latency, req.max_latency_ms)
            elif status == "reject":
                self._close(False)
            cand, ended = self._next_decision()
            if ended and not self.cfg.continuing:
                done[t] = 1.0
        # Bootstrap value for the state after the rollout.
        x, ef, ctx, m, pos = obs_tensors(self.env, cand, fs)
        _, v_next = model(x, ef, ctx, m, pos, self.ei)
        batch = dict(x=torch.cat(X), ef=torch.cat(EF), ctx=torch.cat(C), mask=torch.cat(M),
                     pos=torch.cat(P), act=torch.cat(A), logp=torch.cat(LP), val=torch.cat(V),
                     rew=torch.from_numpy(self.rew.copy()), done=torch.from_numpy(done))
        return batch, float(v_next)


def gae(rew, val, done, v_next, gamma, lam):
    T = len(rew)
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nv = v_next if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def evaluate_policy(infra, env_cfg, choose, trace, warmup):
    env = FogEnv(infra, env_cfg)
    return run_policy(env, trace, choose, warmup=warmup)


def train(infra: Infrastructure, env_cfg: EnvConfig, cfg: TrainConfig, out_dir: Path,
          log_every: int = 1, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    col = Collector(infra, env_cfg, cfg)
    model = make_policy(col.env, cfg)
    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location="cpu", weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    vnorm = ValueNorm()
    n_updates = max(1, cfg.total_steps // cfg.rollout_steps)
    if cfg.val_warmup_holdings > 0:
        val_sets = [trace_with_warmup(infra, cfg.val_requests - cfg.val_warmup, ld, 5_000_000 + 1000 * cfg.seed + i,
                                      cfg.mean_holding_s, cfg.val_warmup_holdings)
                    for i, ld in enumerate(cfg.val_loads)]
    else:
        val_sets = [(generate_trace(infra, cfg.val_requests, ld, 5_000_000 + 1000 * cfg.seed + i, cfg.mean_holding_s),
                     cfg.val_warmup) for i, ld in enumerate(cfg.val_loads)]
    best, log = -np.inf, []
    (out_dir / "config.json").write_text(json.dumps({"train": asdict(cfg), "env": asdict(env_cfg),
                                                     "infra": infra.summary()}, indent=2))
    t_start = time.time()
    for upd in range(1, n_updates + 1):
        frac = (upd - 1) / n_updates
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1.0 - frac)
        ent_coef = cfg.entropy_start + (cfg.entropy_end - cfg.entropy_start) * frac

        model.eval()
        batch, v_next = col.collect(model, cfg.rollout_steps)
        model.train()
        val_dn = vnorm.denorm(batch["val"])
        adv, ret = gae(batch["rew"], val_dn, batch["done"], vnorm.denorm(torch.tensor(v_next)),
                       cfg.gamma, cfg.gae_lambda)
        vnorm.update(ret)
        ret_n = vnorm.norm(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T = len(adv)
        stats = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(T)
            for s in range(0, T, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                logits, v = model(batch["x"][idx], batch["ef"][idx], batch["ctx"][idx],
                                  batch["mask"][idx], batch["pos"][idx], col.ei)
                dist = Categorical(logits=logits)
                lp = dist.log_prob(batch["act"][idx])
                ratio = torch.exp(lp - batch["logp"][idx])
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
                v_old = batch["val"][idx]
                v_clip = v_old + (v - v_old).clamp(-cfg.clip, cfg.clip)
                vl = torch.max((v - ret_n[idx]) ** 2, (v_clip - ret_n[idx]) ** 2).mean()
                # Entropy only over valid actions (masked logits are -1e9 -> p = 0).
                ent = dist.entropy().mean()
                loss = pg + cfg.value_coef * vl - ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                with torch.no_grad():
                    kl = float((batch["logp"][idx] - lp).mean())
                    cf = float(((ratio - 1).abs() > cfg.clip).float().mean())
                stats.append((pg.item(), vl.item(), ent.item(), kl, cf))

        acc_roll = float(np.mean(col.ep_acc[-5:])) if col.ep_acc else float("nan")
        rc = cfg.reward
        if rc.lagrangian_target is not None and col.ep_acc:
            col.lam = float(np.clip(col.lam + rc.lagrangian_lr * (rc.lagrangian_target - acc_roll),
                                    0.05, rc.lagrangian_max))
        s = np.mean(stats, axis=0)
        row = {"update": upd, "steps": upd * cfg.rollout_steps, "time_s": time.time() - t_start,
               "reward_mean": float(batch["rew"].sum() / max(1, (batch["rew"] != 0).sum())),
               "train_acceptance": acc_roll, "lambda": col.lam,
               "pg_loss": s[0], "v_loss": s[1], "entropy": s[2], "approx_kl": s[3], "clip_frac": s[4]}
        if upd % cfg.val_every == 0 or upd == n_updates:
            chooser = RLChooser(model, cfg.feature_set)
            scores, accs, pw = [], [], []
            for tr, w in val_sets:
                rows, summ = evaluate_policy(infra, env_cfg, chooser, tr, w)
                scores.append(score_rows(rows, rc, col.env.p_ref))
                accs.append(summ["acceptance"])
                pw.append(summ["avg_power_w"])
            model.train()
            if rc.lagrangian_target is not None:
                # Constrained selection: lowest power subject to the acceptance target.
                scores = [-float(np.mean(pw)) / 1000.0 - 100.0 * max(0.0, rc.lagrangian_target - float(np.mean(accs)))]
            row.update(val_score=float(np.mean(scores)), val_acceptance=float(np.mean(accs)),
                       val_power_w=float(np.mean(pw)))
            if row["val_score"] > best:
                best = row["val_score"]
                torch.save(model.state_dict(), out_dir / "best.pt")
                row["best"] = True
        log.append(row)
        if verbose and (upd % log_every == 0):
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(model.state_dict(), out_dir / "last.pt")
    return model, log


def load_policy(infra: Infrastructure, env_cfg: EnvConfig, run_dir: Path, which: str = "best.pt"):
    meta = json.loads((run_dir / "config.json").read_text())["train"]
    meta["reward"] = RewardConfig(**meta["reward"])
    meta["train_loads"] = tuple(meta["train_loads"])
    meta["val_loads"] = tuple(meta["val_loads"])
    cfg = TrainConfig(**meta)
    model = make_policy(FogEnv(infra, env_cfg), cfg)
    model.load_state_dict(torch.load(run_dir / which, map_location="cpu", weights_only=True))
    return model.eval(), cfg
