#!/usr/bin/env python3
"""Run a list of shell jobs with N parallel workers, skipping finished ones (resumable).

Each line of the job file is   name <TAB> done-marker <TAB> command
A job is skipped when its done-marker file exists. Output goes to logs/Q/<name>.log.
With --claimdir, every job is claimed through an exclusive file before it starts, so that
several queue instances can share one job file (add workers by starting another instance).
A job may wait for a file with a fourth column:  name <TAB> marker <TAB> command <TAB> needs

    python scripts/jobq.py jobs.txt --workers 4
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("jobfile")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--logdir", default="logs/Q")
ap.add_argument("--claimdir", default=None)
a = ap.parse_args()

jobs = []
for line in Path(a.jobfile).read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    name, marker, cmd, *needs = line.split("\t")
    if Path(marker).exists():
        continue
    jobs.append((name, cmd, needs[0] if needs else None))
Path(a.logdir).mkdir(parents=True, exist_ok=True)
print(f"{len(jobs)} jobs to run", flush=True)
running = []
t0 = time.time()
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(f"[{time.time() - t0:7.0f}s] done {n} rc={p.returncode}", flush=True)]
    while jobs and len(running) < a.workers:
        ready = [j for j in jobs if j[2] is None or Path(j[2]).exists()]
        if not ready:
            break
        name, cmd, _ = ready[0]
        jobs.remove(ready[0])
        if a.claimdir:
            Path(a.claimdir).mkdir(parents=True, exist_ok=True)
            try:
                open(Path(a.claimdir) / name, "x").close()
            except FileExistsError:
                continue
        log = open(Path(a.logdir) / f"{name}.log", "w")
        running.append((name, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)))
        print(f"[{time.time() - t0:7.0f}s] start {name}", flush=True)
    time.sleep(5)
print("all done", flush=True)
sys.exit(0)
