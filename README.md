# vnfplace: energy-aware online SFC placement in IoT fog networks

## What is modelled

| Aspect | Model |
|---|---|
| Substrate | Synthetic IoT edge/fog/cloud continuum (`fog`, `fog:<edge>x<fog>`) or real backbones from SNDlib / Internet Topology Zoo with geographic link lengths (`real:abilene`, `real:nobel_germany`, `real:janos_us`, `real:geant2012`, `real:germany50`, `real:uninett2011`) |
| Hardware | Heterogeneous node classes (Raspberry Pi 4, Jetson, NUC, 1U fog server, cloud DC) with vCPU, RAM, relative speed, idle/max power (`vnfplace/catalog.py`) |
| Services | Five IoT service classes (industrial control, video analytics, e-health, smart metering, AR offload) with SFC templates of length 3-5 over ten VNF types, per-class bandwidth and latency SLA |
| Traffic | Poisson arrivals, exponential holding times, offered load defined on edge+fog CPU |
| Latency | Shortest-delay routing (propagation 5 us/km + switching + serialisation) + load-dependent processing delay `proc / speed / (1 - u)` |
| Energy | Linear server power with sleep for idle nodes, always-on cloud, per-bit link energy; exact time integral between events |
| Feasibility | Chain-order placement; sound mask (CPU <= u_max, RAM, path bandwidth, latency lower bound, exact last step); optional completion oracle (`lookahead`) |

## Methods

* Heuristics: Random, First-Fit, Best-Fit, Least-Loaded, Greedy-Latency, Greedy-Energy and the
  weighted family Greedy-Mix:w (`baselines.py`)
* Per-request optimum: CP-SAT model with exact verification and no-good cuts (`ilp.py`)
* Learned: PPO with a graph encoder (`mlp`, `gcn`, `gine`, `gatv2`), shared or per-position actor heads,
  centralised critic, optional PPO-Lagrangian acceptance constraint (`policy.py`, `ppo.py`);
  DQN-Lagrangian with the same network, mask, shield and constraint (`--learner dqn`)
* Deployment rule: `ens:<glob>` averages the action distributions of several trained
  seeds over the feasible set (seed ensemble)
* Feasibility for every method: sound mask, lazy completion shield (`--shield`),
  protection of running services (`--protect-sla`) and optional capacity reservation
  for latency-critical classes (`--reserve R`)

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests

# train (one seed)
python scripts/run.py train --topo fog --topo-seed 1 --encoder gine --beta 10 --eta 0 \
    --lagrangian 0.98 --steps 300000 --seed 0 --out runs/fog/gnn_s0

# benchmark everything on identical traces (matched seeds via {seed})
python scripts/run.py bench --topo fog --topo-seed 1 --loads 0.5 0.75 1.0 1.25 1.5 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --ilp --rl "GNN-PPO=runs/fog/gnn_s{seed}" --out results/fog

python scripts/run.py stats --results results/fog --reference GNN-PPO
python scripts/run.py plots --results results/fog
```

`scripts/run_all.sh` contains the full experiment matrix (E1-E5) of the revision plan.

## Reproducing the manuscript (R8)

Every experiment of the paper is a line in one of the `jobs_*.txt` files
(`name <TAB> done-marker <TAB> command`). They run with a resumable queue:

```bash
python scripts/jobq.py jobs_r8_train.txt --workers 8    # extra training (DQN, fixed weights)
python scripts/jobq.py jobs_r8_eval.txt  --workers 8    # selection traces and test benchmarks
python scripts/jobq.py jobs_e1r.txt      --workers 8    # GEANT with capacity reservation R = 0.3
python scripts/jobq.py jobs_timing.txt   --workers 1    # decision time on an idle core
python scripts/make_tables_r8.py --out ../paper/gen_r8  # tables, figures and LaTeX macros
```

The trained policies used in the paper (`runs/`, `runs_v21/`, `runs_v3/`; `best.pt` is the
checkpoint selected on validation traces and the one every benchmark loads) are distributed as
`checkpoints_R8.zip`; unzip it in the repository root to rerun the benchmarks without training.

All evaluations use the protected protocol
`--shield --protect-sla --sla-impact --warmup-holdings 2`. Selection traces use
`--trace-base 7500000` and never overlap the test traces. The ensemble of ten seeds is
evaluated with `--rl 'GNN-MAPPO-E=ens:runs/<topo>/GNN-PPO_s*'`. The reservation-aware
policies on GEANT are fine-tuned from the base checkpoints:

```bash
python scripts/run.py train --topo real:geant2012 --reserve 0.3 --encoder gine --seed 0 \
    --steps 300000 --lr 1.5e-4 --lambda-init 12 --lagrangian 0.98 --beta 10 --eta 0 \
    --init-from runs/real_geant2012/GNN-PPO_s0/last.pt --out runs_v21/real_geant2012/GNN-C30_s0
```

Every number in Sections 6 and 7 of the manuscript is a macro written by
`make_tables_r8.py` from the `summary.csv` files, so text and tables cannot drift apart.

## Reproducibility

* Training traces use seeds `1e6*(seed+1)+episode`, validation `5e6+1000*seed+i`,
  test `9e6+1000*seed+100*load`; the three sets never overlap and every method
  sees the same test trace.
* All parameters are written to `config.json` / `bench_args.json` next to the results.
* Topology data: SNDlib (Orlowski et al., Networks 2010) and Internet Topology Zoo
  (Knight et al., IEEE JSAC 2011), exported with `topohub` (MIT licence).
