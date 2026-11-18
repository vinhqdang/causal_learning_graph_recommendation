#!/bin/bash
# v17 reruns, resumed after a restart: lane 1 DIN (resumes) then SIM; lane 2 EASE then LightGCN; then PW-LightGCN alone
cd /home/user/causal_learning_graph_recommendation
L=results/logs_v17
run() { name=$1; shift; [ -f "${@: -1}" ] && return 0; s=$(date +%s); python3 "$@" >> $L/$name.log 2>&1; rc=$?; echo "$name rc=$rc secs=$(( $(date +%s) - s ))" >> $L/status.txt; }
export OMP_NUM_THREADS=2
( run din experiments/run_din.py --arch din --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 2 --out results/v17/din.json
  run sim experiments/run_din.py --arch sim --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 2 --out results/v17/sim.json ) &
( run ease experiments/run_dax.py --backbone ease --seeds 5 --threads 2 --out results/v17/ease.json
  run lightgcn experiments/run_dax.py --backbone lightgcn --mode posthoc --seeds 5 --threads 2 --out results/v17/lightgcn.json ) &
wait
export OMP_NUM_THREADS=4
run pw experiments/run_dax.py --backbone qp_posthoc --qp_source stored --select_from results/v11/dax_qp.json --epoch_window 3 --seeds 5 --threads 4 --out results/v17/pw.json
echo ALL_DONE >> $L/status.txt
