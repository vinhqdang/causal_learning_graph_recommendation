#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 \
  --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1 \
  --methods DR --redo --out results/v2/filters_yahoo_pop.json > results/v2/log_filters_yahoo_dr.txt 2>&1
rm -f results/v2/fat_yahoo_pop.json
python3 experiments/run_fat.py --dataset yahoo --prop pop --dtype float64 > results/v2/log_fat_yahoo.txt 2>&1
python3 experiments/tau_tradeoff.py --dataset yahoo --prop pop --frozen \
  --taus 0.0005 0.002 0.01 0.05 0.1 0.2 --tag _frozen > results/v2/log_tau_yahoo.txt 2>&1
python3 experiments/run_rerank.py --dataset yahoo --prop pop > results/v2/log_rerank_yahoo.txt 2>&1
python3 experiments/frontier.py --dataset yahoo --prop pop --alphas 0.5 --floors 0.1 0.2 \
  --lams 20 --cvs 0 1 > results/v2/log_frontier_yahoo.txt 2>&1
