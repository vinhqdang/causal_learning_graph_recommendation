#!/usr/bin/env bash
# Runs the remaining v2 experiments once the filter runs have finished.
wait_for() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
cd "$(dirname "$0")/.."
wait_for "[r]un_filters.py --dataset coat"
OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset yahoo --prop pop --seeds 5 --threads 1 \
  --epochs 60 --batch 1024 --out results/v2/learned_yahoo_pop.json > results/v2/log_learned_yahoo.txt 2>&1 &
wait_for "[r]un_filters.py --dataset yahoo"
OMP_NUM_THREADS=1 python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 \
  --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1 \
  --methods IPS IPS+WC --redo --out results/v2/filters_yahoo_pop.json > results/v2/log_filters_yahoo_ips.txt 2>&1
OMP_NUM_THREADS=1 python3 experiments/run_fat.py --dataset yahoo --prop pop --dtype float64 \
  > results/v2/log_fat_yahoo.txt 2>&1
OMP_NUM_THREADS=1 python3 experiments/tau_tradeoff.py --dataset yahoo --prop pop --frozen \
  --taus 0.0005 0.002 0.01 0.05 0.1 0.2 --tag _frozen > results/v2/log_tau_yahoo.txt 2>&1
OMP_NUM_THREADS=1 python3 experiments/run_rerank.py --dataset yahoo --prop pop > results/v2/log_rerank_yahoo.txt 2>&1
wait_for "[r]un_filters.py --dataset kuairec"
OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 2 \
  --epochs 30 --patience 3 --batch 32768 --pairs_per_epoch 2000000 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 \
  --seed_check 2 --out results/v2/learned_kuairec_pop.json > results/v2/log_learned_kuairec.txt 2>&1 &
OMP_NUM_THREADS=2 python3 experiments/run_fat.py --dataset kuairec --prop pop --dtype float32 --reps 3 \
  > results/v2/log_fat_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/tau_tradeoff.py --dataset kuairec --prop pop --frozen --dtype float32 \
  --taus 0.01 0.05 0.1 0.2 --reps 2 --tag _frozen > results/v2/log_tau_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/run_rerank.py --dataset kuairec --prop pop --dtype float32 \
  > results/v2/log_rerank_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/frontier.py --dataset kuairec --prop pop --dtype float32 \
  --alphas 0.5 --floors 0.05 0.1 0.2 --lams 20 > results/v2/log_frontier_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/sparse_regime.py > results/v2/log_sparse_kuairec.txt 2>&1
