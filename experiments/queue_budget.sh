#!/usr/bin/env bash
# Tuning-budget curves: per-configuration validation and test scores.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 \
  --epochs 100 --batch 256 --seed_check 0 --dump_bank results/v2/bank_learned_coat.json \
  > results/v2/log_bank_learned_coat.txt 2>&1 &
python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr \
  --methods Obs EASE GF-CF IPS DR DRUP EASE-DR GF-CF-DR --dump_bank results/v2/bank_filters_coat.json \
  > results/v2/log_bank_filters_coat.txt 2>&1
python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 \
  --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1 \
  --methods Obs DR DRUP --dump_bank results/v2/bank_filters_yahoo.json > results/v2/log_bank_filters_yahoo.txt 2>&1
wait
export OMP_NUM_THREADS=2
python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
  --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr \
  --methods Obs DRUP --dump_bank results/v2/bank_filters_kuairec.json > results/v2/log_bank_filters_kuairec.txt 2>&1
