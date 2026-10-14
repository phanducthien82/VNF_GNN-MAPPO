"""Deep Q-network baseline for the same placement MDP (one decision per VNF).

This is the value-based counterpart of the PPO learner, in the spirit of the
DQN-based SFC placement methods in the literature (e.g. Li et al., Computer
Communications 2020). Everything except the learning rule is shared with
PPO, so that a comparison isolates the learner:

* the same state features, sound mask and (optionally) completion shield;
* the same network: the per-node scores of ``PlacementPolicy`` are read as
  Q(s, a) for placing the current VNF on node a (masked nodes get -1e9 and are
  never selected nor bootstrapped from);
* the same request-level reward, paid on the step that accepts or rejects the
  request, and the same acceptance constraint handled by the same dual update
  of lambda when ``lagrangian_target`` is set. The replay buffer stores the
  outcome of each request (accepted / rejected and its energy term) and
  rewards are relabelled with the current lambda when a batch is sampled;
* the same training stream (``Collector``), validation traces and constrained
  checkpoint selection, and the same checkpoint format, so that a trained DQN
  is benchmarked with ``RLChooser`` (greedy action = argmax Q) like PPO.

The learner is double DQN with n-step returns, a Huber loss, a Polyak-averaged
target network and linearly decaying epsilon-greedy exploration over the
valid nodes.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .env import EnvConfig, shield_choice
from .ppo import (Collector, RLChooser, TrainConfig, evaluate_policy, make_policy, obs_tensors,
                  score_rows)
from .topology import Infrastructure
from .traffic import generate_trace, trace_with_warmup


@dataclass
class DQNConfig:
    buffer: int = 50_000
    batch: int = 64
    train_every: int = 4          # environment steps per gradient step
    learn_start: int = 5_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.3         # share of training over which epsilon decays
    tau: float = 0.005            # Polyak factor of the target network
    n_step: int = 3
    double: bool = True
    huber: float = 1.0


class Replay:
    """Ring buffer of consecutive decision steps of one stream.

    Step g stores the observation, the action, and the request outcome
    attached to it (flag +1 accepted, -1 rejected, 0 still open) with the
    energy/latency penalty of an accepted request. Transitions are formed at
    sampling time from steps g .. g + n, so every stored observation is also
    the next state of the previous step."""

    def __init__(self, cap: int, x, ef, ctx, mask):
        self.cap, self.g = cap, 0
        self.x = torch.zeros((cap,) + tuple(x.shape[1:]), dtype=x.dtype)
        self.ef = torch.zeros((cap,) + tuple(ef.shape[1:]), dtype=ef.dtype)
        self.ctx = torch.zeros((cap,) + tuple(ctx.shape[1:]), dtype=ctx.dtype)
        self.mask = torch.zeros((cap,) + tuple(mask.shape[1:]), dtype=torch.bool)
        self.pos = torch.zeros(cap, dtype=torch.long)
        self.act = torch.zeros(cap, dtype=torch.long)
        self.flag = torch.zeros(cap)
        self.pen = torch.zeros(cap)
        self.done = torch.zeros(cap)

    def add(self, x, ef, ctx, mask, pos, act) -> int:
        i = self.g % self.cap
        self.x[i], self.ef[i], self.ctx[i], self.mask[i] = x[0], ef[0], ctx[0], mask[0]
        self.pos[i], self.act[i] = int(pos[0]), int(act)
        self.flag[i] = self.pen[i] = self.done[i] = 0.0
        self.g += 1
        return i

    def sample(self, b: int, n: int, rng: np.random.Generator):
        lo = max(0, self.g - self.cap)
        g = torch.from_numpy(rng.integers(lo, self.g - n, size=b))
        steps = (g.unsqueeze(1) + torch.arange(n + 1).unsqueeze(0)) % self.cap   # [b, n+1]
        return g % self.cap, steps


def _rewards(flag, pen, rc, lam):
    """Request reward (same as ppo.request_reward), divided by the acceptance weight so that
    the Q scale does not follow lambda. Dividing by a positive constant keeps the optimum."""
    if rc.lagrangian_target is not None:
        acc, rej = lam, lam
    else:
        acc, rej = rc.acc, rc.rej
    scale = max(acc, rej)
    r = (flag > 0).float() * (acc - pen) - (flag < 0).float() * rej
    return r / scale


class DQNCollector(Collector):
    """The PPO training stream with request outcomes written into the replay buffer."""

    def _close(self, accepted: bool, latency: float = 0.0, max_lat: float = 1.0):
        rc, i = self.cfg.reward, self.open_idx
        self.buf.flag[i] = 1.0 if accepted else -1.0
        if accepted:
            self.buf.pen[i] = rc.beta_energy * self.inc_p / self.env.p_ref + rc.eta_latency * latency / max_lat
        self.n_acc += int(accepted)
        self.open_idx = None


def train_dqn(infra: Infrastructure, env_cfg: EnvConfig, cfg: TrainConfig, dq: DQNConfig, out_dir: Path,
              verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 777)
    col = DQNCollector(infra, env_cfg, cfg)
    q = make_policy(col.env, cfg)
    if cfg.init_from:
        q.load_state_dict(torch.load(cfg.init_from, map_location="cpu", weights_only=True))
    tgt = copy.deepcopy(q).eval()
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr, eps=1e-5)
    rc, ei, fs, n = cfg.reward, col.ei, cfg.feature_set, dq.n_step
    if cfg.val_warmup_holdings > 0:
        val_sets = [trace_with_warmup(infra, cfg.val_requests - cfg.val_warmup, ld, 5_000_000 + 1000 * cfg.seed + i,
                                      cfg.mean_holding_s, cfg.val_warmup_holdings)
                    for i, ld in enumerate(cfg.val_loads)]
    else:
        val_sets = [(generate_trace(infra, cfg.val_requests, ld, 5_000_000 + 1000 * cfg.seed + i, cfg.mean_holding_s),
                     cfg.val_warmup) for i, ld in enumerate(cfg.val_loads)]
    (out_dir / "config.json").write_text(json.dumps({"train": asdict(cfg), "dqn": asdict(dq), "env": asdict(env_cfg),
                                                     "infra": infra.summary(), "learner": "dqn"}, indent=2))
    gam = torch.tensor([cfg.gamma ** k for k in range(n)])

    col.open_idx = None
    cand, _ = col._next_decision()
    x, ef, ctx, m, pos = obs_tensors(col.env, cand, fs)
    col.buf = Replay(dq.buffer, x, ef, ctx, m)
    buf = col.buf

    def choose_eps(eps, cur, obs):
        def choose(env, c):
            if rng.random() < eps:
                return int(rng.choice(np.flatnonzero(c.mask)))
            # The shield asks again with a reduced mask: only the mask changes.
            xx, ee, cc, mm, pp = obs if c is cur else obs_tensors(env, c, fs)
            with torch.no_grad():
                return int(q(xx, ee, cc, mm, pp, ei)[0].argmax(-1))
        return choose

    best, log, losses, t_start = -np.inf, [], [], time.time()
    total, period = cfg.total_steps, cfg.rollout_steps
    for step in range(1, total + 1):
        frac = (step - 1) / total
        eps = dq.eps_end + (dq.eps_start - dq.eps_end) * max(0.0, 1.0 - frac / dq.eps_frac)
        for gr in opt.param_groups:
            gr["lr"] = cfg.lr * (1.0 - frac)
        choose = choose_eps(eps, cand, (x, ef, ctx, m, pos))
        a = choose(col.env, cand)
        if col.env.cfg.shield:
            a = shield_choice(col.env, cand, choose, a)
        i = buf.add(x, ef, ctx, m, pos, a)
        col.inc_p += float(cand.inc_power[a])
        col.open_idx = i
        req = col.env.req
        status = col.env.place(a)
        if status == "accept":
            col._close(True, col.env.last_latency, req.max_latency_ms)
        elif status == "reject":
            col._close(False)
        cand, ended = col._next_decision()
        if ended and not cfg.continuing:
            buf.done[i] = 1.0
        x, ef, ctx, m, pos = obs_tensors(col.env, cand, fs)

        if step >= dq.learn_start and step % dq.train_every == 0 and buf.g > n + dq.batch:
            idx, steps = buf.sample(dq.batch, n, rng)
            # Steps whose request is still open (last few) carry flag 0, i.e. reward 0 so far:
            # only transitions whose n steps are complete are sampled, and the newest step is excluded.
            r = _rewards(buf.flag[steps[:, :n]], buf.pen[steps[:, :n]], rc, col.lam)      # [b, n]
            d = buf.done[steps[:, :n]]
            alive = torch.cumprod(torch.cat([torch.ones(len(idx), 1), 1.0 - d[:, :-1]], 1), 1)
            ret = (r * gam * alive).sum(1)
            boot = alive[:, -1] * (1.0 - d[:, -1]) * cfg.gamma ** n
            nx = steps[:, n]
            with torch.no_grad():
                qt, _ = tgt(buf.x[nx], buf.ef[nx], buf.ctx[nx], buf.mask[nx], buf.pos[nx], ei)
                if dq.double:
                    qo, _ = q(buf.x[nx], buf.ef[nx], buf.ctx[nx], buf.mask[nx], buf.pos[nx], ei)
                    v = qt.gather(1, qo.argmax(1, keepdim=True)).squeeze(1)
                else:
                    v = qt.max(1).values
                y = ret + boot * v
            qs, _ = q(buf.x[idx], buf.ef[idx], buf.ctx[idx], buf.mask[idx], buf.pos[idx], ei)
            qa = qs.gather(1, buf.act[idx].unsqueeze(1)).squeeze(1)
            loss = F.huber_loss(qa, y, delta=dq.huber)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip * 20)
            opt.step()
            with torch.no_grad():
                for pt, po in zip(tgt.parameters(), q.parameters()):
                    pt.mul_(1.0 - dq.tau).add_(po, alpha=dq.tau)
            losses.append(loss.item())

        if step % period == 0:
            upd = step // period
            acc_roll = float(np.mean(col.ep_acc[-5:])) if col.ep_acc else float("nan")
            if rc.lagrangian_target is not None and col.ep_acc:
                col.lam = float(np.clip(col.lam + rc.lagrangian_lr * (rc.lagrangian_target - acc_roll),
                                        0.05, rc.lagrangian_max))
            row = {"update": upd, "steps": step, "time_s": time.time() - t_start, "epsilon": eps,
                   "train_acceptance": acc_roll, "lambda": col.lam,
                   "td_loss": float(np.mean(losses)) if losses else float("nan")}
            losses = []
            if upd % cfg.val_every == 0 or step == total:
                q.eval()
                chooser = RLChooser(q, fs)
                scores, accs, pw = [], [], []
                for tr, w in val_sets:
                    rows, summ = evaluate_policy(infra, env_cfg, chooser, tr, w)
                    scores.append(score_rows(rows, rc, col.env.p_ref))
                    accs.append(summ["acceptance"])
                    pw.append(summ["avg_power_w"])
                if rc.lagrangian_target is not None:
                    scores = [-float(np.mean(pw)) / 1000.0
                              - 100.0 * max(0.0, rc.lagrangian_target - float(np.mean(accs)))]
                row.update(val_score=float(np.mean(scores)), val_acceptance=float(np.mean(accs)),
                           val_power_w=float(np.mean(pw)))
                if row["val_score"] > best:
                    best = row["val_score"]
                    torch.save(q.state_dict(), out_dir / "best.pt")
                    row["best"] = True
            log.append(row)
            if verbose:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    torch.save(q.state_dict(), out_dir / "last.pt")
    return q, log
