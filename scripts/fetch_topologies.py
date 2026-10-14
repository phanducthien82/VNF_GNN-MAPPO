#!/usr/bin/env python3
"""Re-export the bundled real topologies (requires `pip install topohub`)."""
import json
from pathlib import Path

import topohub

NAMES = {"abilene": "topozoo/Abilene", "nobel_germany": "sndlib/nobel-germany", "janos_us": "sndlib/janos-us",
         "geant2012": "topozoo/Geant2012", "germany50": "sndlib/germany50", "uninett2011": "topozoo/Uninett2011"}
OUT = Path(__file__).resolve().parents[1] / "vnfplace" / "data" / "topologies"

for short, key in NAMES.items():
    g = topohub.get(key)
    ids = {n["id"]: i for i, n in enumerate(g["nodes"])}
    spec = {"name": short, "source": f"{key} (via topohub; original data: Internet Topology Zoo / SNDlib)",
            "nodes": [{"id": ids[n["id"]], "name": n.get("name", str(n["id"])), "lon": n["pos"][0], "lat": n["pos"][1]}
                      for n in g["nodes"]],
            "edges": [{"u": ids[e["source"]], "v": ids[e["target"]], "dist_km": round(float(e["dist"]), 2)}
                      for e in g["edges"]]}
    (OUT / f"{short}.json").write_text(json.dumps(spec, indent=1))
    print(short, len(spec["nodes"]), len(spec["edges"]))
