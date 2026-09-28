#!/usr/bin/env bash
cd "$(dirname "$0")/.."
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
( wait_pid 5210
  OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 2 \
    --epochs 30 --patience 3 --batch 32768 --pairs_per_epoch 2000000 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 \
    --seed_check 2 --out results/v2/learned_kuairec_pop.json > results/v2/log_learned_kuairec.txt 2>&1 ) &
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 experiments/run_v2.sh kuairec > results/v2/log_filters_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/run_fat.py --dataset kuairec --prop pop --dtype float32 --reps 3 \
  > results/v2/log_fat_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/tau_tradeoff.py --dataset kuairec --prop pop --frozen --dtype float32 \
  --taus 0.01 0.05 0.1 0.2 --reps 2 --tag _frozen > results/v2/log_tau_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/run_rerank.py --dataset kuairec --prop pop --dtype float32 \
  > results/v2/log_rerank_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/frontier.py --dataset kuairec --prop pop --dtype float32 \
  --alphas 0.5 --floors 0.05 0.1 0.2 --lams 20 > results/v2/log_frontier_kuairec.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/sparse_regime.py > results/v2/log_sparse_kuairec.txt 2>&1
wait
