"""Hardware, VNF and IoT service catalogues.

Units used throughout the package:
    CPU      vCPU (cores)
    RAM      GB
    traffic  Mbps
    delay    ms
    power    W
    time     s
    energy   J

All numeric values below are *representative assumptions* chosen to be in the
range reported by vendors and measurement studies. They are configurable and
MUST be listed in the paper's parameter table with their sources. Values that
the authors could not verify against a source should be presented as
assumptions and covered by the sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Compute node classes (IoT edge -> fog -> cloud continuum)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeClass:
    name: str
    tier: str            # "edge" | "fog" | "cloud"
    vcpu: float          # schedulable vCPU
    ram_gb: float
    speed: float         # relative per-core processing speed (fog NUC = 1.0)
    p_idle_w: float      # power when switched on and idle
    p_max_w: float       # power at 100 % CPU utilisation
    always_on: bool = False   # cloud: shared, never put to sleep


# Linear power model P(u) = P_idle + (P_max - P_idle) * u, widely used for servers
# (e.g. SPECpower-style measurements). Representative figures:
#   Raspberry Pi 4B   ~2.7 W idle / ~6.4 W full load
#   Jetson Xavier NX  10 W / 15 W power modes
#   Intel NUC-class   ~10 W idle / ~45 W full load
#   1U fog server     ~90 W idle / ~250 W full load
#   Cloud: marginal power only (shared, always on), ~9 W per vCPU incl. PUE 1.5
NODE_CLASSES: Dict[str, NodeClass] = {
    "edge-rpi4": NodeClass("edge-rpi4", "edge", 4, 4, 0.45, 2.7, 6.4),
    "edge-jetson": NodeClass("edge-jetson", "edge", 6, 8, 0.70, 5.0, 15.0),
    "fog-nuc": NodeClass("fog-nuc", "fog", 8, 16, 1.00, 10.0, 45.0),
    "fog-server": NodeClass("fog-server", "fog", 32, 64, 1.20, 90.0, 250.0),
    "cloud-dc": NodeClass("cloud-dc", "cloud", 512, 2048, 1.30, 0.0, 512 * 9.0, always_on=True),
}


# -----------------------------------------------------------------------------
# VNF types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class VNFType:
    name: str
    cpu_base: float       # vCPU independent of traffic
    cpu_per_mbps: float   # vCPU per Mbps of chain traffic
    ram_gb: float
    proc_ms: float        # per-packet processing time on an idle speed-1.0 core


VNF_TYPES: Dict[str, VNFType] = {
    "gateway": VNFType("gateway", 0.25, 0.005, 0.25, 0.30),      # MQTT/CoAP protocol translation
    "firewall": VNFType("firewall", 0.25, 0.004, 0.25, 0.10),
    "nat": VNFType("nat", 0.20, 0.003, 0.20, 0.08),
    "ids": VNFType("ids", 0.50, 0.020, 1.00, 0.50),
    "lb": VNFType("lb", 0.25, 0.004, 0.25, 0.10),
    "encrypt": VNFType("encrypt", 0.50, 0.010, 0.25, 0.30),      # TLS/IPsec termination
    "aggregator": VNFType("aggregator", 0.50, 0.005, 0.50, 0.40),
    "anomaly": VNFType("anomaly", 1.00, 0.020, 1.00, 1.50),      # lightweight ML inference
    "transcoder": VNFType("transcoder", 1.50, 0.030, 1.00, 2.00),
    "cache": VNFType("cache", 0.25, 0.002, 2.00, 0.20),
}
VNF_TYPE_NAMES: List[str] = list(VNF_TYPES)


# -----------------------------------------------------------------------------
# IoT service classes (SFC templates)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ServiceClass:
    name: str
    chain: Tuple[str, ...]
    bw_mbps: Tuple[float, float]       # uniform range
    max_latency_ms: Tuple[float, float]
    destination: str                   # "local" (back to an edge node) | "cloud"
    weight: float = 1.0                # share of the arrival mix


SERVICE_CLASSES: Dict[str, ServiceClass] = {
    "industrial_control": ServiceClass(
        "industrial_control", ("gateway", "firewall", "anomaly"), (2, 10), (8, 15), "local", 1.0),
    "video_analytics": ServiceClass(
        "video_analytics", ("firewall", "transcoder", "anomaly", "cache"), (10, 40), (40, 80), "cloud", 1.0),
    "ehealth": ServiceClass(
        "ehealth", ("gateway", "encrypt", "ids", "aggregator"), (1, 5), (20, 40), "cloud", 1.0),
    "smart_metering": ServiceClass(
        "smart_metering", ("gateway", "aggregator", "firewall", "nat"), (1, 8), (50, 150), "cloud", 1.0),
    "ar_offload": ServiceClass(
        "ar_offload", ("firewall", "lb", "transcoder", "cache", "nat"), (20, 60), (15, 30), "local", 1.0),
}
SERVICE_CLASS_NAMES: List[str] = list(SERVICE_CLASSES)
MAX_CHAIN_LEN: int = max(len(s.chain) for s in SERVICE_CLASSES.values())


def vnf_cpu(vnf: VNFType, bw_mbps: float) -> float:
    return vnf.cpu_base + vnf.cpu_per_mbps * bw_mbps
