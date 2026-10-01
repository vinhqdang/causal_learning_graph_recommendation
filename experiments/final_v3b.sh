#!/usr/bin/env bash
# Local part of the KuaiRand run after the debiased group moved to a GPU runtime:
# the sample-split operators for the main draw and split seeds 1-4.
cd "$(dirname "$0")/.."
L=results/v3
R="--dataset kuairand --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.5 0.7 --floors 0.02 0.05 0.1 --lams 20 --imps add lr --cvs 0 0.5 1"
export OMP_NUM_THREADS=3
python3 experiments/run_filters.py $R --methods DR-split DRUP-split --out $L/filters_kuairand_pop_splitmain.json >> $L/log_filters_kuairand.txt 2>&1
for s in 1 2 3 4; do
  python3 experiments/run_filters.py $R --split_seed $s --methods DR-split DRUP-split --out $L/filters_kuairand_pop_split$s.json >> $L/log_filters_kuairand.txt 2>&1
done
