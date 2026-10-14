"""Heuristic placement baselines.

All heuristics see exactly the same feasibility mask as the learned policies
(``Candidates.mask``) and place VNFs in chain order, so they differ only in
the node-selection rule.
"""
from __future__ import annotations

import numpy as np

from .env import Candidates, FogEnv


def _pick(mask: np.ndarray, score: np.ndarray) -> int:
    """argmin of score over masked nodes (ties broken by node id)."""
    s = np.where(mask, score, np.inf)
    return int(np.argmin(s))


class Heuristic:
    name = "base"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: FogEnv, cand: Candidates) -> int:
        raise NotImplementedError


class RandomFit(Heuristic):
    name = "Random"

    def __call__(self, env, cand):
        return int(self.rng.choice(np.flatnonzero(cand.mask)))


class FirstFit(Heuristic):
    """Lowest node id (edge -> fog -> cloud in the synthetic topology)."""
    name = "First-Fit"

    def __call__(self, env, cand):
        return int(np.flatnonzero(cand.mask)[0])


class BestFit(Heuristic):
    """Tightest residual CPU after placement (consolidation)."""
    name = "Best-Fit"

    def __call__(self, env, cand):
        c = env.req.cpu[env.j]
        resid = (env.res_cap_cpu - env.used_cpu - c) / env.I.cap_cpu
        return _pick(cand.mask, resid + 1e-6 * cand.latency_lb)


class LeastLoaded(Heuristic):
    """Largest residual CPU after placement (load balancing / worst fit)."""
    name = "Least-Loaded"

    def __call__(self, env, cand):
        return _pick(cand.mask, cand.util_after + 1e-6 * cand.latency_lb)


class GreedyLatency(Heuristic):
    """Smallest latency lower bound (latency-first)."""
    name = "Greedy-Latency"

    def __call__(self, env, cand):
        return _pick(cand.mask, cand.latency_lb + 1e-6 * cand.util_after)


class GreedyEnergy(Heuristic):
    """Smallest marginal power (energy-first), latency as tie-break."""
    name = "Greedy-Energy"

    def __call__(self, env, cand):
        return _pick(cand.mask, cand.inc_power + 1e-3 * cand.latency_lb)


class GreedyMix(Heuristic):
    """Parametric energy-latency rule: argmin  dP/P_ref + w * D_lb / D_r.

    w = 0 recovers Greedy-Energy and w -> inf recovers Greedy-Latency; sweeping w traces
    the operating points that a tunable heuristic can reach (the heuristic frontier).
    """
    name = "Greedy-Mix"

    def __init__(self, seed: int = 0, w: float = 1.0):
        super().__init__(seed)
        self.w = float(w)

    def __call__(self, env, cand):
        score = cand.inc_power / env.p_ref + self.w * cand.latency_lb / env.req.max_latency_ms
        return _pick(cand.mask, score + 1e-9 * cand.latency_lb)


HEURISTICS = {h.name: h for h in [RandomFit, FirstFit, BestFit, LeastLoaded, GreedyLatency, GreedyEnergy]}


def make_heuristic(name: str, seed: int = 0) -> Heuristic:
    """``name`` is a key of HEURISTICS or ``Greedy-Mix:<w>``."""
    if name.startswith("Greedy-Mix:"):
        return GreedyMix(seed=seed, w=float(name.split(":", 1)[1]))
    return HEURISTICS[name](seed=seed)
