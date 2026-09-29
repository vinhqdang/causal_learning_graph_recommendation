#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
[ -f results/v3/semisynth_kuairec_dense.json ] || python3 experiments/semisynth_kuairec.py --reps 30 --density 0.13 \
  --pmin 0.02 --tau 0.02 --k5 --out results/v3/semisynth_kuairec_dense.json >> results/v3/log_semisynth_dense.txt 2>&1
[ -f results/v3/semisynth_kuairec_sparse.json ] || python3 experiments/semisynth_kuairec.py --reps 30 --density 0.03 \
  --pmin 0.005 --tau 0.01 --out results/v3/semisynth_kuairec_sparse.json >> results/v3/log_semisynth_sparse.txt 2>&1
