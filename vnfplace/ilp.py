"""Per-request optimal placement with OR-Tools CP-SAT.

For each arriving request the solver returns the placement with minimum
marginal power (node wake-up + dynamic CPU power + link transport power),
subject to CPU (u_max), RAM, per-link bandwidth (with multiplicity) and the
end-to-end latency SLA on the precomputed shortest-delay paths.

Processing delay depends on the final utilisation of each host, which is
non-linear in the decision variables. The model therefore uses an optimistic
(relaxed) delay for every (VNF, node) pair - utilisation after adding that VNF
alone - and every returned solution is verified with the exact simulator
model. If verification fails the assignment is excluded with a no-good cut and
the model is re-solved. Because the relaxed feasible set contains the true one
and the objective is exact, the first verified solution is optimal for the
true per-request problem (up to the solver time limit).

This is the *myopic* per-request optimum; it is not the offline optimum of
the whole online problem, which is NP-hard and depends on future arrivals.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .env import Candidates, FogEnv

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover
    cp_model = None

SCALE = 1000  # ms -> us, W -> mW for the integer model


class PerRequestILP:
    name = "ILP-PerRequest"

    def __init__(self, time_limit_s: float = 2.0, latency_weight: float = 1e-3,
                 max_cuts: int = 20, workers: int = 1):
        if cp_model is None:
            raise ImportError("ortools is required for the ILP baseline (pip install ortools)")
        self.time_limit_s = time_limit_s
        self.latency_weight = latency_weight
        self.max_cuts = max_cuts
        self.workers = workers
        self.plan: Optional[List[int]] = None
        self.plan_rid = -1
        self.solve_ms: List[float] = []
        self.status_counts: Dict[str, int] = {}

    def __call__(self, env: FogEnv, cand: Candidates):
        req = env.req
        if env.j == 0 or self.plan_rid != req.rid:
            t0 = time.perf_counter()
            self.plan = self._solve(env)
            self.solve_ms.append((time.perf_counter() - t0) * 1e3)
            self.plan_rid = req.rid
        if self.plan is None:
            return None
        n = self.plan[env.j]
        return n if cand.mask[n] else None

    # ------------------------------------------------------------------
    def _verify(self, env: FogEnv, plan: List[int]) -> bool:
        I, req = env.I, env.req
        cpu = env.used_cpu.copy()
        for k, n in enumerate(plan):
            cpu[n] += req.cpu[k]
        u = np.minimum(cpu / I.cap_cpu, 0.999)
        lat, prev = 0.0, req.src
        for k, n in enumerate(plan):
            lat += I.sp_delay[prev, n] + req.proc_ms[k] / I.speed[n] / max(1.0 - u[n], 1e-3)
            prev = n
        lat += I.sp_delay[prev, req.dst]
        if lat > req.max_latency_ms + 1e-9:
            return False
        if env.cfg.protect_sla:
            # Exact check that no running service exceeds its budget after the admission.
            g = 1.0 / np.maximum(1.0 - u, 1e-3)
            if np.any(env._sl_hop + env._sl_coef @ g > env._sl_D + 1e-9):
                return False
        return True

    @staticmethod
    def _protect_cap(env: FogEnv) -> np.ndarray:
        """Largest CPU that can be added on each node without pushing a running service hosted
        there over its budget when only that node changes (a necessary condition; the exact
        joint check is done in ``_verify``)."""
        g, slack, _ = env._sla_slack()
        coef = env._sl_coef
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(coef > 0, np.maximum(slack, 0.0)[:, None] / coef, np.inf).min(axis=0)
        u_max = np.where(np.isfinite(h), 1.0 - 1.0 / (g + h), np.inf)
        return np.minimum(u_max, 0.999) * env.I.cap_cpu - env.used_cpu

    def _solve(self, env: FogEnv) -> Optional[List[int]]:
        I, req = env.I, env.req
        N, L, b = env.N, req.length, req.bw
        res_bw = I.cap_bw - env.used_bw
        cap_mult = np.floor((res_bw + 1e-9) / b).astype(int)
        active = env.active()
        nj = env.cfg.link_nj_per_bit * 1e-3 * b           # W per traversed link
        wake = np.where(active, 0.0, I.p_idle * (1.0 - env.cfg.sleep_frac))
        D = req.max_latency_ms

        # Residual CPU per node (reservation share and, if enabled, protection of running services).
        cpu_room = env.res_cap_cpu - env.used_cpu
        if env.cfg.protect_sla:
            cpu_room = np.minimum(cpu_room, self._protect_cap(env))

        # Per-(k, n) admissibility and optimistic processing delay.
        allowed = []
        proc = np.zeros((L, N))
        for k in range(L):
            ok = (req.cpu[k] <= cpu_room + 1e-9) & \
                 (env.used_ram + req.ram[k] <= I.cap_ram + 1e-9)
            u = np.minimum((env.used_cpu + req.cpu[k]) / I.cap_cpu, 0.999)
            proc[k] = req.proc_ms[k] / I.speed / np.maximum(1.0 - u, 1e-3)
            ok &= I.sp_delay[req.src] + proc[k] + I.sp_delay[:, req.dst] <= D + 1e-9
            allowed.append(np.flatnonzero(ok))
            if allowed[-1].size == 0:
                return None

        m = cp_model.CpModel()
        x = {(k, n): m.NewBoolVar(f"x{k}_{n}") for k in range(L) for n in allowed[k]}
        for k in range(L):
            m.AddExactlyOne(x[k, n] for n in allowed[k])
        y = {}
        for k in range(L - 1):
            for a in allowed[k]:
                for c in allowed[k + 1]:
                    y[k, a, c] = m.NewBoolVar(f"y{k}_{a}_{c}")
            for a in allowed[k]:
                m.Add(sum(y[k, a, c] for c in allowed[k + 1]) == x[k, a])
            for c in allowed[k + 1]:
                m.Add(sum(y[k, a, c] for a in allowed[k]) == x[k + 1, c])

        # CPU / RAM per node.
        for n in range(N):
            ks = [k for k in range(L) if (k, n) in x]
            if len(ks) > 1:
                m.Add(sum(int(np.ceil(req.cpu[k] * SCALE)) * x[k, n] for k in ks)
                      <= int(np.floor(cpu_room[n] * SCALE + 1e-6)))
                m.Add(sum(int(np.ceil(req.ram[k] * SCALE)) * x[k, n] for k in ks)
                      <= int(np.floor((I.cap_ram[n] - env.used_ram[n]) * SCALE + 1e-6)))

        # Bandwidth per link: count traversals of each link.
        uses: Dict[int, list] = {}

        def leg(lit, a, c):
            if a == c:
                return
            pe = I.path_edges[a, c]
            for e in pe[pe >= 0]:
                uses.setdefault(int(e), []).append(lit)

        for n in allowed[0]:
            leg(x[0, n], req.src, n)
        for n in allowed[L - 1]:
            leg(x[L - 1, n], n, req.dst)
        for (k, a, c), lit in y.items():
            leg(lit, a, c)
        for e, lits in uses.items():
            if len(lits) > cap_mult[e]:
                m.Add(sum(lits) <= int(cap_mult[e]))

        # Latency (optimistic processing delays).
        lat_terms = []
        for n in allowed[0]:
            lat_terms.append(int(round((I.sp_delay[req.src, n] + proc[0, n]) * SCALE)) * x[0, n])
        for k in range(1, L):
            for n in allowed[k]:
                lat_terms.append(int(round(proc[k, n] * SCALE)) * x[k, n])
        for n in allowed[L - 1]:
            lat_terms.append(int(round(I.sp_delay[n, req.dst] * SCALE)) * x[L - 1, n])
        for (k, a, c), lit in y.items():
            lat_terms.append(int(round(I.sp_delay[a, c] * SCALE)) * lit)
        lat_expr = sum(lat_terms)
        m.Add(lat_expr <= int(np.floor(D * SCALE)))

        # Objective: marginal power (+ tiny latency tie-break).
        z = {}
        for n in range(N):
            ks = [k for k in range(L) if (k, n) in x]
            if ks and wake[n] > 0:
                z[n] = m.NewBoolVar(f"z{n}")
                for k in ks:
                    m.AddImplication(x[k, n], z[n])
        obj = []
        for (k, n), lit in x.items():
            obj.append(int(round(env.p_dyn_per_vcpu[n] * req.cpu[k] * SCALE)) * lit)
        for n, lit in z.items():
            obj.append(int(round(wake[n] * SCALE)) * lit)
        for n in allowed[0]:
            obj.append(int(round(nj * I.sp_hops[req.src, n] * SCALE)) * x[0, n])
        for n in allowed[L - 1]:
            obj.append(int(round(nj * I.sp_hops[n, req.dst] * SCALE)) * x[L - 1, n])
        for (k, a, c), lit in y.items():
            obj.append(int(round(nj * I.sp_hops[a, c] * SCALE)) * lit)
        lw = max(1, int(round(self.latency_weight * SCALE)))
        m.Minimize(SCALE * sum(obj) + lw * lat_expr)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_workers = self.workers
        for _ in range(self.max_cuts + 1):
            st = solver.Solve(m)
            name = solver.StatusName(st)
            self.status_counts[name] = self.status_counts.get(name, 0) + 1
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None
            plan = [int(next(n for n in allowed[k] if solver.Value(x[k, n]))) for k in range(L)]
            if self._verify(env, plan):
                return plan
            m.AddBoolOr([x[k, plan[k]].Not() for k in range(L)])   # no-good cut
        return None
