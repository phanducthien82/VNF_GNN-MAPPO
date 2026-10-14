#!/usr/bin/env python3
"""Job lists for the R8 experiment programme (consumed by scripts/jobq.py).

Every benchmark uses the R8 protocol: sound mask with protection of running services,
completion shield, SLA-impact tracking, time-based warm-up of two mean holding times,
2000 requests of which 1700 are measured, and the test trace stream.

    python scripts/make_jobs_r8.py bench > jobs_bench.txt
    python scripts/make_jobs_r8.py train > jobs_train.txt
"""
import sys

TOPOS = {"fog": "fog --topo-seed 1", "real_geant2012": "real:geant2012", "real_nobel_germany": "real:nobel_germany"}
LOADS = "0.5 0.75 1.0 1.25 1.5"
MID = "0.75 1.0 1.25"
PROT = "--shield --protect-sla --sla-impact --warmup-holdings 2"
MIX = "Greedy-Mix:0.1 Greedy-Mix:0.3 Greedy-Mix:1 Greedy-Mix:3"
RUN = "python3 scripts/run.py"


def job(name, out, cmd):
    print(f"{name}\t{out}/summary.csv\t{cmd} --out {out}")


def bench():
    for s in range(10):
        for t, T in TOPOS.items():
            o = f"results_v3/E1/{t}"
            job(f"E1_{t}_heur_s{s}", f"{o}/heur_s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} "
                f"--heuristics Greedy-Latency Greedy-Energy Least-Loaded Best-Fit")
            job(f"E1_{t}_mix_s{s}", f"{o}/mix_s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} --heuristics {MIX}")
            job(f"E1_{t}_rl_s{s}", f"{o}/rl_s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} --heuristics none "
                f"--rl 'GNN-MAPPO=runs/{t}/GNN-PPO_s{{seed}}' 'MLP-MAPPO=runs/{t}/MLP-PPO_s{{seed}}'")
    # SLA integrity without protection (same traces, first five seeds, middle loads).
    for s in range(5):
        for t, T in TOPOS.items():
            job(f"E0_{t}_s{s}", f"results_v3/E0/{t}/s{s}",
                f"{RUN} bench --topo {T} --shield --sla-impact --warmup-holdings 2 --loads {MID} --seeds {s} "
                f"--heuristics Greedy-Latency Greedy-Energy Least-Loaded Best-Fit Greedy-Mix:0.3 Greedy-Mix:1 "
                f"--rl 'GNN-MAPPO=runs/{t}/GNN-PPO_s{{seed}}' 'MLP-MAPPO=runs/{t}/MLP-PPO_s{{seed}}'")
    # Ablation variants (fog), shield off (fog).
    for s in range(10):
        job(f"E5_abl_s{s}", f"results_v3/E5_ablation/s{s}",
            f"{RUN} bench --topo fog --topo-seed 1 {PROT} --loads {MID} --seeds {s} --heuristics none --rl "
            + " ".join(f"'{v}=runs/fog/{v}_s{{seed}}'" for v in
                       ["GCN-PPO", "GATv2-PPO", "GNN-basicfeat", "GCN-basicfeat-sep", "GNN-resmask"]))
        job(f"E5_noshield_s{s}", f"results_v3/E5_noshield/s{s}",
            f"{RUN} bench --topo fog --topo-seed 1 --protect-sla --sla-impact --warmup-holdings 2 --loads {MID} "
            f"--seeds {s} --heuristics Greedy-Latency Greedy-Energy Least-Loaded Best-Fit Greedy-Mix:1 "
            f"--rl 'GNN-MAPPO=runs/fog/GNN-PPO_s{{seed}}' 'MLP-MAPPO=runs/fog/MLP-PPO_s{{seed}}'")


def train():
    common = "--lagrangian 0.98 --beta 10 --eta 0"
    for s in range(10):
        for t, T in TOPOS.items():
            out = f"runs_v3/{t}/DQN_s{s}"
            print(f"T_dqn_{t}_s{s}\t{out}/last.pt\t{RUN} train --topo {T} --learner dqn {common} --lr 1e-4 "
                  f"--seed {s} --out {out}")
        for w in (2, 7, 20):
            if s >= 5 and w != 7:
                continue
            out = f"runs_v3/fog/PPO-w{w}_s{s}"
            print(f"T_w{w}_fog_s{s}\t{out}/last.pt\t{RUN} train --topo fog --topo-seed 1 --r-acc {w} --r-rej {w} "
                  f"--beta 10 --eta 0 --seed {s} --out {out}")


def select_r():
    """Reservation share on GÉANT, chosen on the selection stream (never on test traces)."""
    for r in (0.0, 0.1, 0.2, 0.3, 0.4):
        for s in range(5):
            job(f"SEL_R{r}_s{s}", f"results_v3/SEL_reserve/r{r}_s{s}",
                f"{RUN} bench --topo real:geant2012 {PROT} --reserve {r} --trace-base 7500000 --loads {MID} "
                f"--seeds {s} --heuristics Greedy-Latency Greedy-Energy Greedy-Mix:0.3 Greedy-Mix:1 Greedy-Mix:3 "
                f"--rl 'GNN-MAPPO=runs/real_geant2012/GNN-PPO_s{{seed}}'")


def zeroshot(seeds=5):
    topos = {"fog_40x12": "fog:40x12 --topo-seed 1", "real_janos_us": "real:janos_us",
             "real_germany50": "real:germany50", "real_uninett2011": "real:uninett2011",
             "real_geant2012": "real:geant2012", "real_nobel_germany": "real:nobel_germany"}
    for s in range(seeds):
        for t, T in topos.items():
            job(f"E4_{t}_s{s}", f"results_v3/E4/{t}/s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {MID} --seeds {s} "
                f"--heuristics Greedy-Latency Greedy-Energy Greedy-Mix:0.3 Greedy-Mix:1 "
                f"--rl 'GNN-MAPPO-zs=runs/fog/GNN-PPO_s{{seed}}' 'MLP-MAPPO-zs=runs/fog/MLP-PPO_s{{seed}}'")


def ilp():
    for t, T in TOPOS.items():
        for s in range(3 if t == "real_geant2012" else 5):
            job(f"ILP_{t}_s{s}", f"results_v3/E1_ilp/{t}/s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {MID} --seeds {s} --heuristics none --ilp")


def learned_new():
    """Benchmarks of the newly trained learners on the E1 traces (merged with E1 by the table script)."""
    for s in range(10):
        for t, T in TOPOS.items():
            job(f"E1_{t}_dqn_s{s}", f"results_v3/E1/{t}/dqn_s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} --heuristics none "
                f"--rl 'DQN-Lagrangian=runs_v3/{t}/DQN_s{{seed}}'")
        rl = " ".join(f"'PPO-w{w}=runs_v3/fog/PPO-w{w}_s{{seed}}'" for w in ((2, 7, 20) if s < 5 else (7,)))
        job(f"E5_w_s{s}", f"results_v3/E5_weighted/s{s}",
            f"{RUN} bench --topo fog --topo-seed 1 {PROT} --loads {MID} --seeds {s} --heuristics none --rl {rl}")


def train2():
    """Training queue in priority order (several queue instances share it through --claimdir).

    DQN-Lagrangian: ten seeds on the IoT fog network, three on each backbone.
    Weighted-reward PPO: three seeds per weight on the IoT fog network.
    """
    common = "--lagrangian 0.98 --beta 10 --eta 0"

    def dqn(t, s):
        out = f"runs_v3/{t}/DQN_s{s}"
        print(f"T_dqn_{t}_s{s}\t{out}/last.pt\t{RUN} train --topo {TOPOS[t]} --learner dqn {common} --lr 1e-4 "
              f"--seed {s} --out {out}")

    def wppo(w, s):
        out = f"runs_v3/fog/PPO-w{w}_s{s}"
        print(f"T_w{w}_fog_s{s}\t{out}/last.pt\t{RUN} train --topo fog --topo-seed 1 --r-acc {w} --r-rej {w} "
              f"--beta 10 --eta 0 --seed {s} --out {out}")

    weighted = [(20, 0), (2, 1), (7, 1), (20, 1), (2, 2), (7, 2), (20, 2)]
    for s in range(1, 10):
        dqn("fog", s)
        if weighted:
            wppo(*weighted.pop(0))
    for s in range(3):
        dqn("real_geant2012", s)
        dqn("real_nobel_germany", s)


def eval2():
    """Evaluation queue that follows the main benchmark; learned models are waited for."""
    select_r()
    print("SEL_fog_learners\tresults_v3/SEL_fog_learners/summary.csv\t"
          f"{RUN} bench --topo fog --topo-seed 1 {PROT} --trace-base 7500000 --loads {MID} --seeds 0 "
        f"--heuristics Greedy-Latency Greedy-Energy Greedy-Mix:0.1 Greedy-Mix:0.3 Greedy-Mix:1 Greedy-Mix:3 "
        f"--rl 'GNNv2=runs/fog/GNN-PPO_s{{seed}}' 'CONT=runs_v3/fog/CONT_s{{seed}}' "
        f"'CONTprot=runs_v3/fog/GNN_s{{seed}}' 'EPprot=runs_v3/fog/EP_s{{seed}}' 'DQN=runs_v3/fog/DQN_s{{seed}}' "
          "--out results_v3/SEL_fog_learners\truns_v3/fog/DQN_s0/last.pt")
    for s in range(10):
        for t, T in TOPOS.items():
            if t != "fog" and s >= 3:
                continue
            out = f"results_v3/E1/{t}/dqn_s{s}"
            print(f"E1_{t}_dqn_s{s}\t{out}/summary.csv\t{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} "
                  f"--heuristics none --rl 'DQN-Lagrangian=runs_v3/{t}/DQN_s{{seed}}' --out {out}"
                  f"\truns_v3/{t}/DQN_s{s}/last.pt")
    for s in range(3):
        for w in (2, 7, 20):
            out = f"results_v3/E5_weighted/w{w}_s{s}"
            print(f"E5_w{w}_s{s}\t{out}/summary.csv\t{RUN} bench --topo fog --topo-seed 1 {PROT} --loads {LOADS} "
                  f"--seeds {s} --heuristics none --rl 'PPO-w{w}=runs_v3/fog/PPO-w{w}_s{{seed}}' --out {out}"
                  f"\truns_v3/fog/PPO-w{w}_s{s}/last.pt")
    ilp()
    zeroshot()


def select_r_ens():
    """Reservation share chosen for the deployed policy (ten-seed ensemble), selection stream."""
    for r in (0.0, 0.1, 0.2, 0.3, 0.4):
        for s in range(3):
            job(f"SEL_Rens{r}_s{s}", f"results_v3/SEL_reserve_ens/r{r}_s{s}",
                f"{RUN} bench --topo real:geant2012 {PROT} --reserve {r} --trace-base 7500000 --loads {MID} "
                f"--seeds {s} --heuristics none --rl 'GNN-MAPPO-E=ens:runs/real_geant2012/GNN-PPO_s*'")


def selseed():
    """Per-seed and ensemble policies on the selection stream (same traces for every policy).

    Used to choose the deployment protocol (one seed, or an ensemble of seeds) without test traces.
    """
    for t, T in TOPOS.items():
        for j in (0, 1):
            base = (f"{RUN} bench --topo {T} {PROT} --trace-base 7500000 --loads 0.5 1.0 1.5 --seeds {j}")
            o = f"results_v3/SELSEED/{t}/j{j}"
            job(f"SELSEED_{t}_j{j}_heur", f"{o}_heur", f"{base} --heuristics Greedy-Latency Greedy-Energy {MIX}")
            for a, b in ((0, 5), (5, 10)):
                rl = " ".join(f"'GNN-s{k}=runs/{t}/GNN-PPO_s{k}'" for k in range(a, b))
                job(f"SELSEED_{t}_j{j}_gnn{a}", f"{o}_gnn{a}", f"{base} --heuristics none --rl {rl}")
            job(f"SELSEED_{t}_j{j}_ens", f"{o}_ens",
                f"{base} --heuristics none --rl 'GNN-ens10=ens:runs/{t}/GNN-PPO_s*' "
                f"'GNN-ens5a=ens:runs/{t}/GNN-PPO_s[0-4]' 'GNN-ens5b=ens:runs/{t}/GNN-PPO_s[5-9]'")


def e1ens():
    """Ten-seed policy ensembles on the E1 test traces (merged with E1 by the table script)."""
    for enc, tag in (("GNN", "gnn"), ("MLP", "mlp")):
        for s in range(10):
            for t, T in TOPOS.items():
                job(f"E1_{t}_ens{tag}_s{s}", f"results_v3/E1/{t}/ens{tag}_s{s}",
                    f"{RUN} bench --topo {T} {PROT} --loads {LOADS} --seeds {s} --heuristics none "
                    f"--rl '{enc}-MAPPO-E=ens:runs/{t}/{enc}-PPO_s*'")


def phase3():
    """Ensemble-level ablations (IoT fog) and zero-shot transfer including the ensemble."""
    variants = ["GCN-PPO", "GATv2-PPO", "GNN-basicfeat", "GCN-basicfeat-sep", "GNN-resmask"]
    for s in range(10):
        for v in variants:
            job(f"E5ens_{v}_s{s}", f"results_v3/E5_ablation_ens/{v}_s{s}",
                f"{RUN} bench --topo fog --topo-seed 1 {PROT} --loads {MID} --seeds {s} --heuristics none "
                f"--rl '{v}-E=ens:runs/fog/{v}_s*'")
    topos = {"fog_40x12": "fog:40x12 --topo-seed 1", "real_janos_us": "real:janos_us",
             "real_germany50": "real:germany50", "real_uninett2011": "real:uninett2011",
             "real_geant2012": "real:geant2012", "real_nobel_germany": "real:nobel_germany"}
    for s in range(5):
        for t, T in topos.items():
            job(f"E4_{t}_s{s}", f"results_v3/E4/{t}/s{s}",
                f"{RUN} bench --topo {T} {PROT} --loads {MID} --seeds {s} "
                f"--heuristics Greedy-Latency Greedy-Energy Greedy-Mix:0.3 Greedy-Mix:1 "
                f"--rl 'GNN-MAPPO-zs=runs/fog/GNN-PPO_s{{seed}}' 'MLP-MAPPO-zs=runs/fog/MLP-PPO_s{{seed}}' "
                f"'GNN-MAPPO-E-zs=ens:runs/fog/GNN-PPO_s*'")


def timing():
    """Decision times on an otherwise idle machine (run alone, one job at a time)."""
    for t, T in TOPOS.items():
        job(f"TIMING_{t}", f"results_v3/TIMING/{t}",
            f"{RUN} bench --topo {T} {PROT} --loads 1.0 --seeds 0 --threads 1 "
            f"--heuristics Greedy-Latency Greedy-Energy Least-Loaded Best-Fit Greedy-Mix:1 "
            f"--rl 'GNN-MAPPO=runs/{t}/GNN-PPO_s{{seed}}' 'MLP-MAPPO=runs/{t}/MLP-PPO_s{{seed}}' "
            f"'DQN-Lagrangian=runs_v3/{t}/DQN_s{{seed}}'")


def reserve_test(r: float):
    """Test-trace benchmark on GÉANT with the reservation share chosen on the selection stream.

    GNN-MAPPO-R is fine-tuned from the GNN-MAPPO checkpoint with the reservation active
    (runs_v21/real_geant2012/GNN-C<100R>_s*; see reserve_finetune for other shares).
    """
    T, o, tag = TOPOS["real_geant2012"], "results_v3/E1R", f"C{int(round(100 * r))}"
    for s in range(10):
        job(f"E1R_heur_s{s}", f"{o}/heur_s{s}",
            f"{RUN} bench --topo {T} {PROT} --reserve {r} --loads {LOADS} --seeds {s} "
            f"--heuristics Greedy-Latency Greedy-Energy Least-Loaded Best-Fit")
        job(f"E1R_mix_s{s}", f"{o}/mix_s{s}",
            f"{RUN} bench --topo {T} {PROT} --reserve {r} --loads {LOADS} --seeds {s} --heuristics {MIX}")
        job(f"E1R_rl_s{s}", f"{o}/rl_s{s}",
            f"{RUN} bench --topo {T} {PROT} --reserve {r} --loads {LOADS} --seeds {s} --heuristics none "
            f"--rl 'GNN-MAPPO-E=ens:runs/real_geant2012/GNN-PPO_s*' "
            f"'GNN-MAPPO-R-E=ens:runs_v21/real_geant2012/GNN-{tag}_s*'")


def reserve_finetune(r: float):
    tag = f"C{int(round(100 * r))}"
    for s in range(10):
        out = f"runs_v21/real_geant2012/GNN-{tag}_s{s}"
        print(f"T_{tag}_s{s}\t{out}/last.pt\t{RUN} train --topo real:geant2012 --reserve {r} --lagrangian 0.98 "
              f"--beta 10 --eta 0 --lr 1.5e-4 --lambda-init 12 --init-from runs/real_geant2012/GNN-PPO_s{s}/last.pt "
              f"--seed {s} --out {out}")


if __name__ == "__main__":
    {"bench": bench, "train": train, "select_r": select_r, "zeroshot": zeroshot, "ilp": ilp,
     "learned_new": learned_new, "train2": train2, "eval2": eval2, "reserve_test": reserve_test, "timing": timing, "selseed": selseed, "select_r_ens": select_r_ens, "phase3": phase3, "e1ens": e1ens,
     "reserve_finetune": reserve_finetune}[sys.argv[1]](*map(float, sys.argv[2:]))
