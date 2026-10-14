#!/usr/bin/env bash
# Full experiment matrix of the revision plan. Adjust SEEDS/STEPS/JOBS to the available compute.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
STEPS=${STEPS:-300000}
JOBS=${JOBS:-4}
LOADS="0.5 0.75 1.0 1.25 1.5"
TOPOS=${TOPOS:-"fog real:nobel_germany real:geant2012"}
COMMON="--beta 10 --eta 0 --lagrangian 0.98 --steps $STEPS"

train() {  # name topo extra-args...
  local name=$1 topo=$2; shift 2
  for s in $SEEDS; do
    echo "python scripts/run.py train --topo $topo $COMMON --seed $s $* --out runs/${topo//:/_}/${name}_s$s"
  done
}

# E1 main benchmark + E5 ablations (encoder, features, actor, mask)
{
  for t in $TOPOS; do
    train GNN-PPO      "$t" --encoder gine
    train MLP-PPO      "$t" --encoder mlp
  done
  train GCN-PPO        fog --encoder gcn
  train GATv2-PPO      fog --encoder gatv2
  train GNN-basicfeat  fog --encoder gine --features basic
  train GCN-basicfeat-sep fog --encoder gcn --features basic --actor separate   # closest to original R6
  train GNN-resmask    fog --encoder gine --mask resource
} | xargs -P "$JOBS" -I{} bash -c "{}"

for t in $TOPOS; do
  d=${t//:/_}
  python scripts/run.py bench --topo "$t" --loads $LOADS --seeds $SEEDS --ilp \
    --rl "GNN-PPO=runs/$d/GNN-PPO_s{seed}" "MLP-PPO=runs/$d/MLP-PPO_s{seed}" --out "results/E1_$d"
  python scripts/run.py stats --results "results/E1_$d" --reference GNN-PPO
  python scripts/run.py plots --results "results/E1_$d"
done

# E5 ablation (fog only)
python scripts/run.py bench --topo fog --loads 0.75 1.25 --seeds $SEEDS --heuristics Greedy-Latency \
  --rl "GNN-PPO=runs/fog/GNN-PPO_s{seed}" "MLP-PPO=runs/fog/MLP-PPO_s{seed}" "GCN-PPO=runs/fog/GCN-PPO_s{seed}" \
       "GATv2-PPO=runs/fog/GATv2-PPO_s{seed}" "GNN-basicfeat=runs/fog/GNN-basicfeat_s{seed}" \
       "GCN-basicfeat-sep=runs/fog/GCN-basicfeat-sep_s{seed}" --out results/E5_ablation
python scripts/run.py bench --topo fog --mask resource --loads 0.75 1.25 --seeds $SEEDS --heuristics Greedy-Latency \
  --rl "GNN-resmask=runs/fog/GNN-resmask_s{seed}" --out results/E5_ablation_resmask
python scripts/run.py stats --results results/E5_ablation --reference GNN-PPO

# E3 scalability / E4 generalisation: policy trained on one topology, tested on others
for t in real:janos_us real:germany50 real:uninett2011 fog:40x12; do
  python scripts/run.py bench --topo "$t" --loads 1.0 --seeds $SEEDS \
    --heuristics Greedy-Latency Greedy-Energy Least-Loaded \
    --rl "GNN-PPO(zero-shot)=runs/fog/GNN-PPO_s{seed}" "MLP-PPO(zero-shot)=runs/fog/MLP-PPO_s{seed}" \
    --out "results/E4_${t//:/_}"
done
