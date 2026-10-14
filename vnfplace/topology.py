"""Substrate infrastructures: synthetic IoT-fog continuum and real backbones.

Two families are provided:

* ``fog``      - synthetic hierarchical IoT edge / fog / cloud network laid out
                 in a metropolitan area (Waxman-style fog mesh, edge access
                 links, WAN uplink to a cloud data centre).
* ``real:<n>`` - a real backbone (Internet Topology Zoo / SNDlib, bundled as JSON
                 under ``data/topologies``) where every PoP hosts a fog site and a
                 cloud data centre is attached to the most central PoP.

Link delay = propagation (distance / 200 km per ms, i.e. ~5 us/km in fibre)
           + per-hop switching delay
           + serialisation of one packet (packet_bits / capacity).
Routing uses the shortest-delay path (precomputed, single path).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from .catalog import NODE_CLASSES

DATA_DIR = Path(__file__).resolve().parent / "data" / "topologies"
FIBRE_KM_PER_MS = 200.0


@dataclass
class Infrastructure:
    name: str
    graph: nx.Graph
    node_class: List[str]
    tier: np.ndarray            # 0 edge, 1 fog, 2 cloud
    cap_cpu: np.ndarray
    cap_ram: np.ndarray
    speed: np.ndarray
    p_idle: np.ndarray
    p_max: np.ndarray
    always_on: np.ndarray       # bool
    edges: np.ndarray           # [E, 2] undirected edge list (u < v)
    cap_bw: np.ndarray          # [E] Mbps
    link_delay: np.ndarray      # [E] ms (prop + switching + serialisation)
    sp_delay: np.ndarray        # [N, N] ms shortest-delay path delay
    sp_hops: np.ndarray         # [N, N]
    path_edges: np.ndarray      # [N, N, L] edge ids on the path, -1 padded
    source_nodes: np.ndarray    # nodes where IoT requests may originate / terminate locally
    cloud_nodes: np.ndarray

    @property
    def n(self) -> int:
        return len(self.node_class)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def summary(self) -> dict:
        non_cloud = self.tier < 2
        return {
            "name": self.name,
            "nodes": int(self.n),
            "links": int(self.n_edges),
            "edge_nodes": int((self.tier == 0).sum()),
            "fog_nodes": int((self.tier == 1).sum()),
            "cloud_nodes": int((self.tier == 2).sum()),
            "edge_fog_vcpu": float(self.cap_cpu[non_cloud].sum()),
            "mean_link_delay_ms": float(self.link_delay.mean()),
            "diameter_delay_ms": float(self.sp_delay.max()),
        }


def _finalise(name: str, G: nx.Graph, classes: List[str], link_cap: Dict, link_km: Dict,
              switching_ms: float, packet_bytes: int) -> Infrastructure:
    n = G.number_of_nodes()
    assert sorted(G.nodes) == list(range(n))
    nc = [NODE_CLASSES[c] for c in classes]
    tier_map = {"edge": 0, "fog": 1, "cloud": 2}
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in G.edges), dtype=np.int64)
    eid = {(int(u), int(v)): i for i, (u, v) in enumerate(edges)}
    cap_bw = np.array([link_cap[(u, v)] for u, v in edges], dtype=np.float64)
    pkt_bits = packet_bytes * 8.0
    link_delay = np.array([
        link_km[(u, v)] / FIBRE_KM_PER_MS + switching_ms + pkt_bits / (link_cap[(u, v)] * 1e3)
        for u, v in edges
    ], dtype=np.float64)
    for (u, v), d in zip(edges, link_delay):
        G.edges[int(u), int(v)]["delay"] = float(d)

    sp_delay = np.zeros((n, n))
    sp_hops = np.zeros((n, n), dtype=np.int64)
    paths = {}
    for s in range(n):
        dist, path = nx.single_source_dijkstra(G, s, weight="delay")
        for t in range(n):
            sp_delay[s, t] = dist[t]
            p = path[t]
            sp_hops[s, t] = len(p) - 1
            paths[(s, t)] = [eid[(min(a, b), max(a, b))] for a, b in zip(p[:-1], p[1:])]
    L = max(1, int(sp_hops.max()))
    path_edges = -np.ones((n, n, L), dtype=np.int64)
    for (s, t), es in paths.items():
        path_edges[s, t, :len(es)] = es

    tier = np.array([tier_map[c.tier] for c in nc], dtype=np.int64)
    sources = np.flatnonzero(tier == 0)
    if sources.size == 0:                      # real backbones: every PoP is an access site
        sources = np.flatnonzero(tier == 1)
    return Infrastructure(
        name=name, graph=G, node_class=classes, tier=tier,
        cap_cpu=np.array([c.vcpu for c in nc], dtype=np.float64),
        cap_ram=np.array([c.ram_gb for c in nc], dtype=np.float64),
        speed=np.array([c.speed for c in nc], dtype=np.float64),
        p_idle=np.array([c.p_idle_w for c in nc], dtype=np.float64),
        p_max=np.array([c.p_max_w for c in nc], dtype=np.float64),
        always_on=np.array([c.always_on for c in nc], dtype=bool),
        edges=edges, cap_bw=cap_bw, link_delay=link_delay,
        sp_delay=sp_delay, sp_hops=sp_hops, path_edges=path_edges,
        source_nodes=sources, cloud_nodes=np.flatnonzero(tier == 2),
    )


def build_fog(n_edge: int = 12, n_fog: int = 6, area_km: float = 20.0, seed: int = 0,
              access_mbps=(300.0, 1000.0), fog_mbps: float = 10_000.0,
              wan_mbps: float = 10_000.0, wan_km: float = 1500.0,
              switching_ms: float = 0.05, packet_bytes: int = 1500,
              edge_mix: float = 0.5, fog_server_share: float = 0.34) -> Infrastructure:
    """Synthetic metropolitan IoT-fog network with one cloud data centre.

    Node ids: [0, n_edge) edge, [n_edge, n_edge + n_fog) fog, last = cloud.
    The default (12 edge + 6 fog + 1 cloud = 19 nodes) mirrors the scale of the
    original 20-node study but with explicit tiers.
    """
    rng = np.random.default_rng(seed)
    n = n_edge + n_fog + 1
    pos = rng.uniform(0.0, area_km, size=(n, 2))
    # Fog nodes are spread more evenly: jitter around a grid.
    g = math.ceil(math.sqrt(n_fog))
    for k in range(n_fog):
        cx, cy = (k % g + 0.5) / g, (k // g + 0.5) / g
        pos[n_edge + k] = np.clip(np.array([cx, cy]) * area_km + rng.normal(0, area_km * 0.05, 2), 0, area_km)
    classes = []
    for i in range(n_edge):
        classes.append("edge-rpi4" if rng.random() < edge_mix else "edge-jetson")
    n_srv = max(1, int(round(fog_server_share * n_fog)))
    fog_cls = ["fog-server"] * n_srv + ["fog-nuc"] * (n_fog - n_srv)
    rng.shuffle(fog_cls)
    classes += fog_cls + ["cloud-dc"]
    cloud = n - 1

    G = nx.Graph()
    G.add_nodes_from(range(n))
    cap, km = {}, {}

    def add(u, v, mbps, dist):
        a, b = min(u, v), max(u, v)
        if a == b or G.has_edge(a, b):
            return
        G.add_edge(a, b)
        cap[(a, b)] = float(mbps)
        km[(a, b)] = float(dist)

    fog_ids = list(range(n_edge, n_edge + n_fog))
    # Fog mesh: Waxman-like + spanning tree for connectivity.
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))
    T = nx.minimum_spanning_tree(nx.Graph([(a, b, {"weight": d(a, b)}) for a in fog_ids for b in fog_ids if a < b]))
    for a, b in T.edges:
        add(a, b, fog_mbps, d(a, b))
    for a in fog_ids:
        for b in fog_ids:
            if a < b and rng.random() < 0.6 * math.exp(-d(a, b) / (0.5 * area_km)):
                add(a, b, fog_mbps, d(a, b))
    # Edge access: nearest fog (+ second nearest with prob 0.5), occasional edge-edge mesh.
    for e in range(n_edge):
        order = sorted(fog_ids, key=lambda f: d(e, f))
        add(e, order[0], rng.uniform(*access_mbps), d(e, order[0]))
        if len(order) > 1 and rng.random() < 0.5:
            add(e, order[1], rng.uniform(*access_mbps), d(e, order[1]))
    for a in range(n_edge):
        for b in range(a + 1, n_edge):
            if d(a, b) < 0.2 * area_km and rng.random() < 0.3:
                add(a, b, rng.uniform(*access_mbps), d(a, b))
    # WAN uplink(s) from the two most central fog nodes to the cloud.
    bc = nx.betweenness_centrality(G.subgraph(fog_ids))
    for f in sorted(fog_ids, key=lambda f: -bc.get(f, 0))[:2]:
        add(f, cloud, wan_mbps, wan_km)
    assert nx.is_connected(G)
    return _finalise(f"fog-{n_edge}e{n_fog}f-s{seed}", G, classes, cap, km, switching_ms, packet_bytes)


def list_real_topologies() -> List[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.json"))


def build_real(name: str, seed: int = 0, link_mbps: float = 1_000.0,
               server_share: float = 0.4, cloud_km: float = 500.0,
               cloud_mbps: float = 10_000.0, switching_ms: float = 0.05,
               packet_bytes: int = 1500) -> Infrastructure:
    """Real backbone where each PoP is a fog site; one cloud DC attached to the most central PoP."""
    spec = json.loads((DATA_DIR / f"{name}.json").read_text())
    rng = np.random.default_rng(seed)
    n0 = len(spec["nodes"])
    G = nx.Graph()
    G.add_nodes_from(range(n0 + 1))
    cap, km = {}, {}
    for e in spec["edges"]:
        a, b = sorted((int(e["u"]), int(e["v"])))
        if a == b or G.has_edge(a, b):
            continue
        G.add_edge(a, b)
        cap[(a, b)] = float(link_mbps)
        km[(a, b)] = max(1.0, float(e["dist_km"]))
    bc = nx.betweenness_centrality(G.subgraph(range(n0)))
    hub = max(range(n0), key=lambda i: bc[i])
    cloud = n0
    G.add_edge(hub, cloud)
    cap[(hub, cloud)] = float(cloud_mbps)
    km[(hub, cloud)] = float(cloud_km)
    classes = ["fog-server" if rng.random() < server_share else "fog-nuc" for _ in range(n0)] + ["cloud-dc"]
    assert nx.is_connected(G), name
    return _finalise(f"real-{name}", G, classes, cap, km, switching_ms, packet_bytes)


def build_infrastructure(spec: str, seed: int = 0, **kw) -> Infrastructure:
    """``spec``: 'fog', 'fog:<n_edge>x<n_fog>', or 'real:<name>'."""
    if spec == "fog":
        return build_fog(seed=seed, **kw)
    if spec.startswith("fog:"):
        ne, nf = spec.split(":", 1)[1].lower().split("x")
        return build_fog(n_edge=int(ne), n_fog=int(nf), seed=seed, **kw)
    if spec.startswith("real:"):
        return build_real(spec.split(":", 1)[1], seed=seed, **kw)
    raise ValueError(f"Unknown topology spec: {spec}")
