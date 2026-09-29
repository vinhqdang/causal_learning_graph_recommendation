#!/usr/bin/env bash
# One job at a time (running sparse regime and frontier together ran out of memory).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
python3 experiments/sparse_regime.py --qs 1.0 0.1 0.03 --floors 0.05 0.01 \
  > results/v2/log_sparse_kuairec.txt 2>&1
python3 experiments/frontier.py --dataset kuairec --prop pop --dtype float32 \
  --alphas 0.5 --floors 0.05 0.1 0.2 --lams 20 > results/v2/log_frontier_kuairec.txt 2>&1
