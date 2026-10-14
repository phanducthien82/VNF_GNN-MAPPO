"""IoT service-request traces.

Requests arrive as a Poisson process and hold their resources for an
exponentially distributed time (classic Erlang-style loss-system workload
used in VNF/SFC placement studies). The arrival rate is derived from a target
*offered load* defined on the edge+fog CPU capacity:

    load = lambda * E[CPU per request] * E[holding time] / sum(edge+fog vCPU)

so that load = 1.0 means the offered CPU demand equals the whole edge/fog
capacity (the cloud absorbs overflow at a latency cost).

A trace depends only on (topology, seed, load), so every method is evaluated
on exactly the same arrival sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List

import numpy as np

from .catalog import SERVICE_CLASSES, SERVICE_CLASS_NAMES, VNF_TYPES, VNF_TYPE_NAMES, vnf_cpu
from .topology import Infrastructure


@dataclass
class Request:
    rid: int
    t_arrival: float
    holding_s: float
    cls: int
    src: int
    dst: int
    vnf_types: List[int]
    bw: float
    cpu: List[float]
    ram: List[float]
    proc_ms: List[float]
    max_latency_ms: float

    @property
    def length(self) -> int:
        return len(self.vnf_types)

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_request(rng: np.random.Generator, infra: Infrastructure, rid: int, t: float,
                    mean_holding_s: float, weights: np.ndarray) -> Request:
    ci = int(rng.choice(len(SERVICE_CLASS_NAMES), p=weights))
    sc = SERVICE_CLASSES[SERVICE_CLASS_NAMES[ci]]
    bw = float(rng.uniform(*sc.bw_mbps))
    src = int(rng.choice(infra.source_nodes))
    if sc.destination == "cloud" and infra.cloud_nodes.size:
        dst = int(rng.choice(infra.cloud_nodes))
    else:
        dst = src
    types = [VNF_TYPES[v] for v in sc.chain]
    return Request(
        rid=rid, t_arrival=t, holding_s=float(rng.exponential(mean_holding_s)), cls=ci,
        src=src, dst=dst, vnf_types=[VNF_TYPE_NAMES.index(v) for v in sc.chain], bw=bw,
        cpu=[float(vnf_cpu(v, bw)) for v in types], ram=[float(v.ram_gb) for v in types],
        proc_ms=[float(v.proc_ms) for v in types],
        max_latency_ms=float(rng.uniform(*sc.max_latency_ms)),
    )


def class_weights() -> np.ndarray:
    w = np.array([SERVICE_CLASSES[c].weight for c in SERVICE_CLASS_NAMES], dtype=np.float64)
    return w / w.sum()


@lru_cache(maxsize=None)
def mean_cpu_per_request(n_samples: int = 20000, seed: int = 12345) -> float:
    rng = np.random.default_rng(seed)
    w = class_weights()
    tot = 0.0
    for _ in range(n_samples):
        sc = SERVICE_CLASSES[SERVICE_CLASS_NAMES[int(rng.choice(len(w), p=w))]]
        bw = rng.uniform(*sc.bw_mbps)
        tot += sum(vnf_cpu(VNF_TYPES[v], bw) for v in sc.chain)
    return tot / n_samples


def arrival_rate(infra: Infrastructure, load: float, mean_holding_s: float) -> float:
    cap = float(infra.cap_cpu[infra.tier < 2].sum())
    return load * cap / (mean_cpu_per_request() * mean_holding_s)


def generate_trace(infra: Infrastructure, n_requests: int, load: float, seed: int,
                   mean_holding_s: float = 300.0) -> List[Request]:
    rng = np.random.default_rng(seed)
    lam = arrival_rate(infra, load, mean_holding_s)
    w = class_weights()
    t = 0.0
    out = []
    for rid in range(n_requests):
        t += float(rng.exponential(1.0 / lam))
        out.append(_sample_request(rng, infra, rid, t, mean_holding_s, w))
    return out


def trace_with_warmup(infra: Infrastructure, n_measure: int, load: float, seed: int,
                      mean_holding_s: float = 300.0, warmup_holdings: float = 2.0):
    """Trace whose first part (arrivals before ``warmup_holdings`` mean holding times) is warm-up.

    Returns (trace, warm_idx); the ``n_measure`` requests after ``warm_idx`` are measured.
    """
    t_w = warmup_holdings * mean_holding_s
    n_w = int(np.ceil(arrival_rate(infra, load, mean_holding_s) * t_w * 1.3)) + 50
    full = generate_trace(infra, n_w + n_measure, load, seed, mean_holding_s)
    w_idx = next(i for i, r in enumerate(full) if r.t_arrival >= t_w)
    return full[:w_idx + n_measure], w_idx
