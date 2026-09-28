#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_filters.py --dataset coat" > /dev/null; do sleep 30; done
export OMP_NUM_THREADS=1
python3 experiments/run_fat.py --dataset coat --prop given --dtype float64 > results/v2/log_fat_coat.txt 2>&1
python3 experiments/tau_tradeoff.py --dataset coat --prop given --frozen --taus 0.01 0.02 0.05 0.1 0.2 --tag _frozen > results/v2/log_tau_coat.txt 2>&1
python3 experiments/run_rerank.py --dataset coat --prop given > results/v2/log_rerank_coat.txt 2>&1
python3 experiments/frontier.py --dataset coat --prop given > results/v2/log_frontier_coat.txt 2>&1
python3 experiments/significance.py --dataset coat --prop given --ref DRUP DR > results/v2/log_sig_coat.txt 2>&1
