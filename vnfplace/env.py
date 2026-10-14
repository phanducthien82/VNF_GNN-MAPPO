"""Event-driven online SFC placement simulator for IoT fog networks.

Decision process
----------------
Requests are served one at a time in arrival order. For a request with chain
v_1 -> ... -> v_L the controller places the VNFs *in chain order* (so the
predecessor's location, and hence the path, is known when placing v_j).
After each placement the CPU/RAM of the host and the bandwidth on the path
from the predecessor are tentatively reserved; after v_L the last leg to the
destination is reserved and the end-to-end latency is checked. A request is
accepted only if every constraint holds, otherwise all tentative reservations
are rolled back.

Physical model
--------------
* Processing delay of VNF k on node n: proc_k / speed_n / (1 - u_n)   (M/M/1-PS
  style load dependence, u_n = CPU utilisation of n after admission).
* Network delay: shortest-delay path (propagation + switching + serialisation).
* End-to-end latency: src -> v_1 -> ... -> v_L -> dst.
* Node power: P_idle + (P_max - P_idle) u when it hosts >= 1 VNF (or is an
  always-on cloud), sleep_frac * P_idle otherwise.
* Network power: link_nj_per_bit * carried traffic on every traversed link.
* Energy = exact time integral of total power between events.

Feasibility masks (``mask_mode``)
---------------------------------
* ``full``     - CPU (<= u_max), RAM, bandwidth on the path from the
                 predecessor, and a latency *lower bound* for the remaining
                 VNFs. At the last VNF the check is exact (incl. last-leg
                 bandwidth), so an accepted request never violates its SLA at
                 admission time (soundness). Completeness is not guaranteed:
                 a partial placement can still dead-end and is then rejected.
* ``lookahead``- ``full`` plus a completion oracle: a candidate is kept only if
                 a latency-greedy completion of the remaining chain exists from
                 it. Once a request has started it can then never dead-end
                 (the oracle's own next node is always verified at the next
                 step), unless the oracle finds no completion at all, in which
                 case the sound ``full`` mask is used.
* ``resource`` - CPU/RAM only (the original R6 design; bandwidth and latency
                 failures are discovered only after the full placement).
* ``none``     - no masking (infeasible actions cause rejection).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .catalog import MAX_CHAIN_LEN, SERVICE_CLASS_NAMES, VNF_TYPE_NAMES
from .topology import Infrastructure
from .traffic import Request

EPS = 1e-9


@dataclass
class EnvConfig:
    u_max: float = 0.90              # admission cap on CPU utilisation per node
    sleep_frac: float = 0.0          # sleep power as a fraction of P_idle
    link_nj_per_bit: float = 10.0    # network energy per bit per traversed link
    mask_mode: str = "full"          # lookahead | full | resource | none
    track_sla_impact: bool = False   # re-check co-located services after each admission
    shield: bool = False             # lazy completion shield applied to every method (see run_policy)
    reserve: float = 0.0             # CPU share of edge/fog nodes kept free for latency-critical requests
    reserve_budget_ms: float = 30.0  # a request is latency-critical if its budget is <= this value
    protect_sla: bool = False        # at the last VNF, also require that no running service exceeds its budget


@dataclass
class Service:
    req: Request
    nodes: List[int]
    node_cpu: Dict[int, float]
    node_ram: Dict[int, float]
    node_cnt: Dict[int, int]
    edge_bw: Dict[int, float]
    latency_ms: float
    t_end: float
    hop_ms: float = 0.0              # network part of the latency (fixed after admission)
    proc_coef: Dict[int, float] = field(default_factory=dict)   # node -> sum proc/speed of its VNFs there
    violated: bool = False           # SLA exceeded at some time after admission


@dataclass
class Candidates:
    mask: np.ndarray        # [N] bool, according to mask_mode
    full_mask: np.ndarray   # [N] bool, physically feasible w.r.t. the full check
    util_after: np.ndarray  # [N]
    inc_power: np.ndarray   # [N] W, marginal power if VNF placed there
    hop_delay: np.ndarray   # [N] ms, predecessor -> candidate
    latency_lb: np.ndarray  # [N] ms, lower bound of the final latency (exact at last VNF)


@dataclass
class Stats:
    arrived: int = 0
    accepted: int = 0
    rejected: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    sla_impacted: int = 0
    sla_violated: int = 0            # accepted services whose SLA was exceeded at least once after admission
    shield_repairs: int = 0
    vnf_tier: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1


class FogEnv:
    def __init__(self, infra: Infrastructure, cfg: EnvConfig | None = None):
        self.I = infra
        self.cfg = cfg or EnvConfig()
        I = infra
        self.N, self.E = I.n, I.n_edges
        self.cap_cpu_all = I.cap_cpu * self.cfg.u_max
        # Trunk reservation: non-critical requests may fill edge/fog nodes only up to u_max - reserve.
        self.cap_cpu_loose = np.where(I.tier < 2, I.cap_cpu * max(self.cfg.u_max - self.cfg.reserve, 0.0),
                                      self.cap_cpu_all)
        self.res_cap_cpu = self.cap_cpu_all
        self.max_speed = float(I.speed.max())
        self.p_dyn_per_vcpu = (I.p_max - I.p_idle) / I.cap_cpu
        self.p_ref = float(np.max(I.p_max[I.tier < 2])) if (I.tier < 2).any() else float(I.p_max.max())
        # Directed edge list for message passing (both directions).
        e = I.edges
        self.edge_index = np.concatenate([e.T, e[:, ::-1].T], axis=1)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        N, E = self.N, self.E
        self.used_cpu = np.zeros(N)
        self.used_ram = np.zeros(N)
        self.n_hosted = np.zeros(N, dtype=np.int64)
        self.used_bw = np.zeros(E)
        self.t = 0.0
        self.energy_j = 0.0
        self.node_energy_j = 0.0
        self.tier_energy_j = np.zeros(3)
        self.link_energy_j = 0.0
        self.active_node_seconds = 0.0
        self.services: Dict[int, Service] = {}
        self.node_services: List[set] = [set() for _ in range(N)]
        # Dense per-service latency model for the SLA check: lat_s = hop_s + coef_s . g(u).
        self._sl_coef = np.zeros((256, N))
        self._sl_hop = np.zeros(256)
        self._sl_D = np.full(256, np.inf)
        self._sl_free: List[int] = list(range(255, -1, -1))
        self._sl_of: Dict[int, int] = {}
        self.departures: list = []
        self.stats = Stats()
        self.req: Optional[Request] = None

    def utilisation(self) -> np.ndarray:
        return self.used_cpu / self.I.cap_cpu

    def active(self) -> np.ndarray:
        return (self.n_hosted > 0) | self.I.always_on

    def node_power(self) -> np.ndarray:
        I = self.I
        u = self.utilisation()
        return np.where(self.active(), I.p_idle + (I.p_max - I.p_idle) * u, self.cfg.sleep_frac * I.p_idle)

    def power(self) -> tuple[float, float]:
        node = self.node_power().sum()
        link = float(self.used_bw.sum()) * self.cfg.link_nj_per_bit * 1e-3   # Mbps * nJ/bit -> W
        return float(node), link

    def _integrate(self, t_new: float) -> None:
        dt = t_new - self.t
        if dt > 0:
            pv = self.node_power()
            pn = float(pv.sum())
            pl = float(self.used_bw.sum()) * self.cfg.link_nj_per_bit * 1e-3
            self.tier_energy_j += np.bincount(self.I.tier, weights=pv, minlength=3)[:3] * dt
            self.node_energy_j += pn * dt
            self.link_energy_j += pl * dt
            self.energy_j += (pn + pl) * dt
            self.active_node_seconds += float((self.active() & ~self.I.always_on).sum()) * dt
            self.t = t_new

    def advance(self, t_new: float) -> None:
        while self.departures and self.departures[0][0] <= t_new:
            t_dep, sid = heapq.heappop(self.departures)
            self._integrate(t_dep)
            self._release(self.services.pop(sid))
        self._integrate(t_new)

    def _release(self, s: Service) -> None:
        for n, c in s.node_cpu.items():
            self.used_cpu[n] -= c
            self.used_ram[n] -= s.node_ram[n]
            self.n_hosted[n] -= s.node_cnt[n]
            self.node_services[n].discard(s.req.rid)
        for e, b in s.edge_bw.items():
            self.used_bw[e] -= b
        slot = self._sl_of.pop(s.req.rid, None)
        if slot is not None:
            self._sl_coef[slot] = 0.0
            self._sl_hop[slot] = 0.0
            self._sl_D[slot] = np.inf
            self._sl_free.append(slot)
        # Clean round-off.
        np.maximum(self.used_cpu, 0.0, out=self.used_cpu)
        np.maximum(self.used_ram, 0.0, out=self.used_ram)
        np.maximum(self.used_bw, 0.0, out=self.used_bw)

    # --------------------------------------------------------------- requests
    def begin(self, req: Request) -> None:
        self.advance(req.t_arrival)
        self.req = req
        self.stats.arrived += 1
        critical = req.max_latency_ms <= self.cfg.reserve_budget_ms
        self.res_cap_cpu = self.cap_cpu_all if (critical or self.cfg.reserve <= 0) else self.cap_cpu_loose
        self.j = 0
        self.prev = req.src
        self.nodes: List[int] = []
        self.tent_cpu: Dict[int, float] = {}
        self.tent_ram: Dict[int, float] = {}
        self.tent_cnt: Dict[int, int] = {}
        self.tent_bw: Dict[int, float] = {}
        self.proc_coef = np.zeros(self.N)   # sum proc_k / speed_n of chain VNFs on n
        self.trans_acc = 0.0
        # Lower bound on processing of VNFs j..L-1 (unloaded, fastest node).
        p = np.asarray(req.proc_ms, dtype=np.float64) / self.max_speed
        self.proc_lb_suffix = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])

    def _path(self, a: int, b: int) -> np.ndarray:
        pe = self.I.path_edges[a, b]
        return pe[pe >= 0]

    def chain_proc_ms(self) -> float:
        u = self.utilisation()
        return float((self.proc_coef / np.maximum(1.0 - u, 1e-3)).sum())

    def candidates(self) -> Candidates:
        full, cpu_ok_phys, ram_ok, u_after, inc_power, lat_lb, hop = self._core(with_power=True)
        mode = self.cfg.mask_mode
        if mode == "lookahead":
            mask = full.copy()
            if self.j < self.req.length - 1:
                for n in np.flatnonzero(full):
                    mask[n] = self._completes(int(n))
                if not mask.any():          # oracle found nothing: fall back to the sound mask
                    mask = full
        elif mode == "full":
            mask = full
        elif mode == "resource":
            mask = cpu_ok_phys & ram_ok
        elif mode == "none":
            mask = np.ones(self.N, dtype=bool)
        else:
            raise ValueError(mode)
        return Candidates(mask=mask, full_mask=full, util_after=u_after, inc_power=inc_power,
                          hop_delay=hop.copy(), latency_lb=lat_lb)

    def _core(self, with_power: bool = False):
        I, req, j = self.I, self.req, self.j
        L = req.length
        c, r, b, p = req.cpu[j], req.ram[j], req.bw, req.proc_ms[j]
        prev, dst = self.prev, req.dst
        last = j == L - 1

        cpu_ok_phys = self.used_cpu + c <= I.cap_cpu * (1.0 - 1e-6)
        cpu_ok = self.used_cpu + c <= self.res_cap_cpu + EPS
        ram_ok = self.used_ram + r <= I.cap_ram + EPS
        u_cur = self.utilisation()
        u_after = np.minimum((self.used_cpu + c) / I.cap_cpu, 0.999)

        res = I.cap_bw - self.used_bw
        pe = I.path_edges[prev]                          # [N, L]
        bott = np.where(pe >= 0, res[np.maximum(pe, 0)], np.inf).min(axis=1)
        bott[prev] = np.inf
        bw_ok = bott + EPS >= b

        # Exact processing delay of the partial chain if the candidate is used.
        slow_cur = 1.0 / np.maximum(1.0 - u_cur, 1e-3)
        slow_new = 1.0 / np.maximum(1.0 - u_after, 1e-3)
        proc_partial = float((self.proc_coef * slow_cur).sum())
        proc_new = (proc_partial - self.proc_coef * slow_cur + self.proc_coef * slow_new
                    + p / I.speed * slow_new)
        hop = I.sp_delay[prev]
        lat_lb = self.trans_acc + hop + proc_new + self.proc_lb_suffix[j + 1] + I.sp_delay[:, dst]
        lat_ok = lat_lb <= req.max_latency_ms + EPS

        full = cpu_ok & ram_ok & bw_ok & lat_ok
        if last and full.any():
            # Exact last-leg bandwidth, accounting for edges shared by both legs.
            idx = np.flatnonzero(full)
            la = I.path_edges[prev, idx]                 # [K, L] leg prev -> n
            lb = I.path_edges[idx, dst]                  # [K, L] leg n -> dst
            both = ((la[:, :, None] == lb[:, None, :]) & (lb[:, None, :] >= 0)).any(-1)
            ok_a = (la < 0) | (res[np.maximum(la, 0)] + EPS >= b * (1.0 + both))
            ok_b = (lb < 0) | (res[np.maximum(lb, 0)] + EPS >= b)
            full[idx[~(ok_a.all(1) & ok_b.all(1))]] = False
            if self.cfg.protect_sla:
                idx = np.flatnonzero(full)
                full[idx[~self._running_ok_vec(idx, c)]] = False

        if not with_power:
            return full, cpu_ok_phys, ram_ok, u_after, None, lat_lb, hop
        hops = I.sp_hops[prev] + (I.sp_hops[:, dst] if last else 0)
        wake = np.where(self.active(), 0.0, I.p_idle * (1.0 - self.cfg.sleep_frac))
        inc_power = wake + self.p_dyn_per_vcpu * c + b * self.cfg.link_nj_per_bit * 1e-3 * hops
        return full, cpu_ok_phys, ram_ok, u_after, inc_power, lat_lb, hop

    def _reserve_path(self, a: int, b_node: int) -> bool:
        if a == b_node:
            return True
        es = self._path(a, b_node)
        b = self.req.bw
        if np.any(self.I.cap_bw[es] - self.used_bw[es] + EPS < b):
            return False
        for e in es:
            e = int(e)
            self.used_bw[e] += b
            self.tent_bw[e] = self.tent_bw.get(e, 0.0) + b
        return True

    def _reserve_step(self, n: int) -> str:
        """Tentatively reserve VNF j on n (no commit). Returns '' or a failure reason."""
        I, req, j = self.I, self.req, self.j
        c, r = req.cpu[j], req.ram[j]
        if self.used_cpu[n] + c > I.cap_cpu[n] * (1.0 - 1e-6) or self.used_ram[n] + r > I.cap_ram[n] + EPS:
            return "resource"
        if not self._reserve_path(self.prev, n):
            return "bandwidth"
        self.used_cpu[n] += c
        self.used_ram[n] += r
        self.n_hosted[n] += 1
        self.tent_cpu[n] = self.tent_cpu.get(n, 0.0) + c
        self.tent_ram[n] = self.tent_ram.get(n, 0.0) + r
        self.tent_cnt[n] = self.tent_cnt.get(n, 0) + 1
        self.proc_coef[n] += req.proc_ms[j] / I.speed[n]
        self.trans_acc += I.sp_delay[self.prev, n]
        self.prev = n
        self.nodes.append(n)
        self.j += 1
        return ""

    def place(self, n: int) -> str:
        """Place VNF j on node n. Returns 'continue', 'accept' or 'reject'."""
        I, req = self.I, self.req
        n = int(n)
        why = self._reserve_step(n)
        if why:
            return self.abort(why)
        if self.j < req.length:
            return "continue"
        if not self._reserve_path(n, req.dst):
            return self.abort("bandwidth")
        latency = self.trans_acc + I.sp_delay[n, req.dst] + self.chain_proc_ms()
        if latency > req.max_latency_ms + EPS:
            return self.abort("latency")
        if self.cfg.protect_sla and not self._sla_slack()[2]:
            return self.abort("sla_protect")
        self._commit(latency)
        return "accept"

    # ---------------------------------------------------- completion oracle
    def _snapshot(self):
        return (self.used_cpu.copy(), self.used_ram.copy(), self.n_hosted.copy(), self.used_bw.copy(),
                self.proc_coef.copy(), self.trans_acc, self.prev, self.j, dict(self.tent_cpu),
                dict(self.tent_ram), dict(self.tent_cnt), dict(self.tent_bw), list(self.nodes))

    def _restore(self, s) -> None:
        (self.used_cpu, self.used_ram, self.n_hosted, self.used_bw, self.proc_coef, self.trans_acc,
         self.prev, self.j, self.tent_cpu, self.tent_ram, self.tent_cnt, self.tent_bw, self.nodes) = s

    def _completes(self, n: int) -> bool:
        """True if placing VNF j on n admits a feasible completion found by latency-greedy search.

        The last step of ``_core`` is exact, so a completion found here is feasible.
        """
        snap = self._snapshot()
        try:
            if self._reserve_step(int(n)):
                return False
            while self.j < self.req.length:
                full, *_rest, lat_lb, _hop = self._core()
                if not full.any():
                    return False
                self._reserve_step(int(np.argmin(np.where(full, lat_lb, np.inf))))
            return True
        finally:
            self._restore(snap)

    def abort(self, reason: str = "dead_end") -> str:
        for n, c in self.tent_cpu.items():
            self.used_cpu[n] -= c
            self.used_ram[n] -= self.tent_ram[n]
            self.n_hosted[n] -= self.tent_cnt[n]
        for e, b in self.tent_bw.items():
            self.used_bw[e] -= b
        np.maximum(self.used_cpu, 0.0, out=self.used_cpu)
        np.maximum(self.used_ram, 0.0, out=self.used_ram)
        np.maximum(self.used_bw, 0.0, out=self.used_bw)
        self.stats.reject(reason)
        self.last_reason = reason
        self.last_latency = float("nan")
        self.req = None
        return "reject"

    def _commit(self, latency: float) -> None:
        req = self.req
        last = self.nodes[-1] if self.nodes else req.src
        s = Service(req=req, nodes=list(self.nodes), node_cpu=dict(self.tent_cpu),
                    node_ram=dict(self.tent_ram), node_cnt=dict(self.tent_cnt),
                    edge_bw=dict(self.tent_bw), latency_ms=float(latency),
                    t_end=req.t_arrival + req.holding_s,
                    hop_ms=float(self.trans_acc + self.I.sp_delay[last, req.dst]),
                    proc_coef={int(n): float(self.proc_coef[n]) for n in np.flatnonzero(self.proc_coef > 0)})
        self.stats.vnf_tier += np.bincount(self.I.tier[np.asarray(self.nodes, dtype=np.int64)], minlength=3)[:3]
        self.services[req.rid] = s
        for n in s.node_cpu:
            self.node_services[n].add(req.rid)
        if not self._sl_free:
            k = len(self._sl_hop)
            self._sl_coef = np.vstack([self._sl_coef, np.zeros_like(self._sl_coef)])
            self._sl_hop = np.concatenate([self._sl_hop, np.zeros(k)])
            self._sl_D = np.concatenate([self._sl_D, np.full(k, np.inf)])
            self._sl_free = list(range(2 * k - 1, k - 1, -1))
        slot = self._sl_free.pop()
        self._sl_of[req.rid] = slot
        for n, cf in s.proc_coef.items():
            self._sl_coef[slot, n] = cf
        self._sl_hop[slot] = s.hop_ms
        self._sl_D[slot] = req.max_latency_ms
        heapq.heappush(self.departures, (s.t_end, req.rid))
        self.stats.accepted += 1
        self.last_latency = float(latency)
        if self.cfg.track_sla_impact:
            self._check_sla_impact(s)
        self.req = None

    def service_latency(self, s: Service) -> float:
        I, req = self.I, s.req
        u = self.utilisation()
        lat = 0.0
        prev = req.src
        for k, n in enumerate(s.nodes):
            lat += I.sp_delay[prev, n] + req.proc_ms[k] / I.speed[n] / max(1.0 - u[n], 1e-3)
            prev = n
        return float(lat + I.sp_delay[prev, req.dst])

    @staticmethod
    def _latency_at(s: Service, u: np.ndarray) -> float:
        return s.hop_ms + sum(c / max(1.0 - u[n], 1e-3) for n, c in s.proc_coef.items())

    def _running_ok(self, n: Optional[int], c: float) -> bool:
        """True if no running service exceeds its budget with the tentative reservations of the
        current request (plus ``c`` vCPU on node ``n`` when given)."""
        I = self.I
        u = self.used_cpu / I.cap_cpu
        nodes = set(self.tent_cpu)
        if n is not None:
            u = u.copy()
            u[n] = min((self.used_cpu[n] + c) / I.cap_cpu[n], 0.999)
            nodes.add(n)
        for m in nodes:
            for sid in self.node_services[m]:
                s = self.services[sid]
                if self._latency_at(s, u) > s.req.max_latency_ms + EPS:
                    return False
        return True

    def _sla_slack(self):
        """Slowdowns g = 1/(1-u) with the tentative reservations, the latency slack of every
        running service, and whether the services on the reserved nodes are all within budget."""
        g = 1.0 / np.maximum(1.0 - self.used_cpu / self.I.cap_cpu, 1e-3)
        slack = self._sl_D + EPS - (self._sl_hop + self._sl_coef @ g)
        tn = list(self.tent_cpu)
        ok = True
        if tn:
            on_t = (self._sl_coef[:, tn] > 0).any(axis=1)
            ok = not bool((slack[on_t] < 0).any())
        return g, slack, ok

    def _running_ok_vec(self, cand: np.ndarray, c: float) -> np.ndarray:
        """Vectorised ``_running_ok(n, c)`` for every node in ``cand``.

        Adding c vCPU on n changes only the slowdown g_n = 1/(1-u_n), so a running
        service s on n stays within budget iff lat_s + coef_{s,n} * (g_n' - g_n) <= D_s."""
        g, slack, ok = self._sla_slack()
        if not ok:
            return np.zeros(len(cand), dtype=bool)
        I = self.I
        u_new = np.minimum((self.used_cpu[cand] + c) / I.cap_cpu[cand], 0.999)
        dg = 1.0 / np.maximum(1.0 - u_new, 1e-3) - g[cand]
        cc = self._sl_coef[:, cand]
        return ~((cc > 0) & (cc * dg[None, :] > slack[:, None])).any(axis=0)

    def _check_sla_impact(self, new: Service) -> None:
        affected = set()
        for n in new.node_cpu:
            affected |= self.node_services[n]
        affected.discard(new.req.rid)
        u = self.utilisation()
        for sid in affected:
            s = self.services[sid]
            if self._latency_at(s, u) > s.req.max_latency_ms + EPS:
                self.stats.sla_impacted += 1
                if not s.violated:
                    s.violated = True
                    self.stats.sla_violated += 1

    # ---------------------------------------------------------- observations
    NODE_FEATURES = 21
    PATH_FEATURES = slice(14, 21)
    EDGE_FEATURES = 4
    CTX_FEATURES = len(SERVICE_CLASS_NAMES) + len(VNF_TYPE_NAMES) + 12

    def observe(self, cand: Candidates, feature_set: str = "path"):
        """Node features [N, F], edge features [2E, Fe], context [Fc]."""
        I, req, j = self.I, self.req, self.j
        D = req.max_latency_ms
        u = self.utilisation()
        x = np.zeros((self.N, self.NODE_FEATURES), dtype=np.float32)
        x[:, 0] = 1.0 - u
        x[:, 1] = 1.0 - self.used_ram / I.cap_ram
        x[:, 2] = u
        x[:, 3] = self.active()
        x[:, 4] = I.speed / self.max_speed
        x[:, 5] = np.log1p(I.cap_cpu) / np.log1p(512.0)
        x[:, 6] = I.p_idle / self.p_ref
        x[:, 7] = np.minimum(self.p_dyn_per_vcpu / 10.0, 2.0)
        x[np.arange(self.N), 8 + I.tier] = 1.0
        x[self.prev, 11] = 1.0
        x[req.src, 12] = 1.0
        x[req.dst, 13] = 1.0
        x[:, 14] = np.minimum(cand.hop_delay / D, 2.0)
        x[:, 15] = np.minimum(I.sp_delay[:, req.dst] / D, 2.0)
        x[:, 16] = self.proc_coef > 0
        x[:, 17] = np.minimum(cand.inc_power / self.p_ref, 2.0)
        x[:, 18] = np.minimum(cand.latency_lb / D, 2.0)
        x[:, 19] = cand.util_after
        x[:, 20] = cand.mask
        if feature_set == "basic":
            x[:, 14:20] = 0.0          # keep only the mask bit of the path block
        elif feature_set != "path":
            raise ValueError(feature_set)

        res = (I.cap_bw - self.used_bw) / I.cap_bw
        on_chain = np.zeros(self.E, dtype=np.float32)
        for e in self.tent_bw:
            on_chain[e] = 1.0
        ef = np.stack([res, np.log10(I.cap_bw) / 4.0, np.minimum(I.link_delay / D, 2.0), on_chain], axis=1)
        ef = np.concatenate([ef, ef], axis=0).astype(np.float32)

        L = req.length
        ctx = np.zeros(self.CTX_FEATURES, dtype=np.float32)
        ctx[req.cls] = 1.0
        o = len(SERVICE_CLASS_NAMES)
        ctx[o + req.vnf_types[j]] = 1.0
        o += len(VNF_TYPE_NAMES)
        tier_mask = I.tier < 2
        budget = D - self.trans_acc - self.chain_proc_ms()
        ctx[o:o + 12] = [
            j / MAX_CHAIN_LEN, (L - j) / MAX_CHAIN_LEN, L / MAX_CHAIN_LEN,
            req.cpu[j] / 4.0, req.ram[j] / 2.0, req.bw / 60.0, D / 150.0,
            budget / D, sum(req.cpu[j:]) / 10.0,
            float(u[tier_mask].mean()), float(self.active()[tier_mask].mean()),
            np.log1p(len(self.services)) / 6.0,
        ]
        return x, ef, ctx


def shield_choice(env: FogEnv, cand: Candidates, choose, n, max_tries: int = 8):
    """Lazy completion shield (identical for every method).

    The method's proposal is accepted only if the completion oracle certifies
    that the rest of the chain can still be placed; otherwise that node is
    masked out and the method is asked again. After ``max_tries`` refusals, or
    if the method returns None (e.g. the ILP found no verified plan), the
    shield repairs the decision with the latency-greedy certified node. If no
    candidate is certified, the original proposal is kept (no worse than the
    sound ``full`` mask). Proposals at the last VNF are already exact.
    """
    if env.j == env.req.length - 1:
        if n is None:
            return int(np.argmin(np.where(cand.mask, cand.latency_lb, np.inf)))
        return n
    first = n
    mask = cand.mask.copy()
    for _ in range(max_tries):
        if n is None or not mask[int(n)]:
            break
        if env._completes(int(n)):
            return int(n)
        mask[int(n)] = False
        if not mask.any():
            return first
        cand = Candidates(mask=mask.copy(), full_mask=cand.full_mask, util_after=cand.util_after,
                          inc_power=cand.inc_power, hop_delay=cand.hop_delay, latency_lb=cand.latency_lb)
        n = choose(env, cand)
    for m in np.argsort(np.where(mask, cand.latency_lb, np.inf), kind="stable"):
        if not mask[m]:
            break
        if env._completes(int(m)):
            env.stats.shield_repairs += 1
            return int(m)
    return first


def run_policy(env: FogEnv, trace: List[Request], choose, warmup: int = 0, record: bool = True):
    """Run a placement policy ``choose(env, cand) -> node`` over a trace.

    Returns per-request records (after warm-up) and a summary dict. Energy and
    activity are integrated only over the measurement window [t_warm, t_end].
    """
    import time
    env.reset()
    rows = []
    e0 = n0 = l0 = a0 = t0 = None
    for i, req in enumerate(trace):
        if i == warmup:
            env.advance(req.t_arrival)
            e0, n0, l0, a0, t0 = env.energy_j, env.node_energy_j, env.link_energy_j, env.active_node_seconds, env.t
            acc0, arr0, imp0 = env.stats.accepted, env.stats.arrived, env.stats.sla_impacted
            vio0, tier0, te0 = env.stats.sla_violated, env.stats.vnf_tier.copy(), env.tier_energy_j.copy()
            rep0 = env.stats.shield_repairs
            rr0 = dict(env.stats.reject_reasons)
        env.begin(req)
        tic = time.perf_counter()
        status, reason, inc_p = "continue", "", 0.0
        while status == "continue":
            cand = env.candidates()
            if not cand.mask.any():
                reason = "no_candidate" if env.j == 0 else "dead_end"
                status = env.abort(reason)
                break
            n = choose(env, cand)
            if env.cfg.shield:
                n = shield_choice(env, cand, choose, n)
            if n is None:
                status = env.abort("no_solution")
                break
            n = int(n)
            inc_p += float(cand.inc_power[n])
            status = env.place(n)
        if status == "reject":
            reason = env.last_reason
        dt_ms = (time.perf_counter() - tic) * 1e3
        if record and i >= warmup:
            rows.append({
                "rid": req.rid, "cls": SERVICE_CLASS_NAMES[req.cls], "len": req.length,
                "accepted": status == "accept", "reject_reason": reason,
                "latency_ms": env.last_latency if status == "accept" else np.nan,
                "max_latency_ms": req.max_latency_ms,
                "inc_power_w": inc_p if status == "accept" else np.nan,
                "decision_ms": dt_ms,
            })
    env.advance(trace[-1].t_arrival)
    T = env.t - t0
    acc = env.stats.accepted - acc0
    arr = env.stats.arrived - arr0
    summary = {
        "requests": arr,
        "acceptance": acc / max(1, arr),
        "energy_j": env.energy_j - e0,
        "node_energy_j": env.node_energy_j - n0,
        "link_energy_j": env.link_energy_j - l0,
        "avg_power_w": (env.energy_j - e0) / max(T, 1e-9),
        "energy_per_accepted_j": (env.energy_j - e0) / max(1, acc),
        "avg_active_nodes": (env.active_node_seconds - a0) / max(T, 1e-9),
        "sla_impacted": env.stats.sla_impacted - imp0,
        "sla_violated": env.stats.sla_violated - vio0,
        "energy_edge_j": float(env.tier_energy_j[0] - te0[0]),
        "energy_fog_j": float(env.tier_energy_j[1] - te0[1]),
        "energy_cloud_j": float(env.tier_energy_j[2] - te0[2]),
        "vnf_edge": int(env.stats.vnf_tier[0] - tier0[0]),
        "vnf_fog": int(env.stats.vnf_tier[1] - tier0[1]),
        "vnf_cloud": int(env.stats.vnf_tier[2] - tier0[2]),
        "shield_repairs": env.stats.shield_repairs - rep0,
        "window_s": T,
        "reject_reasons": {k: v - rr0.get(k, 0) for k, v in env.stats.reject_reasons.items()},
    }
    return rows, summary
