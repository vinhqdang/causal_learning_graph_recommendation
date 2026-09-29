#!/usr/bin/env bash
cd "$(dirname "$0")/.."
OMP_NUM_THREADS=4 python3 experiments/run_fat.py --dataset kuairec --prop pop --dtype float32 \
  --sections explain privacy --methods DRUP > results/v2/log_fat_kuairec_b.txt 2>&1
OMP_NUM_THREADS=2 python3 experiments/sparse_regime.py > results/v2/log_sparse_kuairec.txt 2>&1 &
OMP_NUM_THREADS=2 python3 experiments/frontier.py --dataset kuairec --prop pop --dtype float32 \
  --alphas 0.5 --floors 0.05 0.1 0.2 --lams 20 > results/v2/log_frontier_kuairec.txt 2>&1
wait
