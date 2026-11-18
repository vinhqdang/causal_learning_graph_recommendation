#!/bin/bash
# sequential, 4 threads each (restarts kill parallel lanes before checkpoints)
cd /home/user/causal_learning_graph_recommendation
L=results/logs_v17
run() { name=$1; shift; [ -f "${@: -1}" ] && return 0; s=$(date +%s); python3 "$@" >> $L/$name.log 2>&1; rc=$?; echo "$name rc=$rc secs=$(( $(date +%s) - s ))" >> $L/status.txt; }
export OMP_NUM_THREADS=4
run din experiments/run_din.py --arch din --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 4 --out results/v17/din.json
run lightgcn experiments/run_dax.py --backbone lightgcn --mode posthoc --seeds 5 --threads 4 --out results/v17/lightgcn.json
run pw experiments/run_dax.py --backbone qp_posthoc --qp_source stored --select_from results/v11/dax_qp.json --epoch_window 3 --seeds 5 --threads 4 --out results/v17/pw.json
run sim experiments/run_din.py --arch sim --dataset kuairand27k --seq data/raw/seq_27k.npz --threads 4 --out results/v17/sim.json
echo ALL_DONE >> $L/status.txt
