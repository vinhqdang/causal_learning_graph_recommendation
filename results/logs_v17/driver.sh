#!/bin/bash
# v17 reruns: stage 1 LEAF-LGN+DAX (windowed) with DIN; stage 2 SIM with EASE; stage 3 QP family (alone); stage 4 LightGCN (alone)
cd /home/user/causal_learning_graph_recommendation
L=results/logs_v17
run() { name=$1; shift; s=$(date +%s); python3 "$@" > $L/$name.log 2>&1; rc=$?; echo "$name rc=$rc secs=$(( $(date +%s) - s ))" >> $L/status.txt; }
export OMP_NUM_THREADS=2
run lgn experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --seeds 5 --threads 2 --out results/v17/lgn.json &
run din experiments/run_din.py --arch din --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 2 --out results/v17/din.json &
wait
run sim experiments/run_din.py --arch sim --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 2 --out results/v17/sim.json &
run ease experiments/run_dax.py --backbone ease --seeds 5 --threads 2 --out results/v17/ease.json &
wait
export OMP_NUM_THREADS=4
run pw experiments/run_dax.py --backbone qp_posthoc --qp_source stored --select_from results/v11/dax_qp.json --epoch_window 3 --seeds 5 --threads 4 --out results/v17/pw.json
export OMP_NUM_THREADS=2
run lightgcn experiments/run_dax.py --backbone lightgcn --mode posthoc --seeds 5 --threads 2 --out results/v17/lightgcn.json
echo ALL_DONE >> $L/status.txt
