#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until [ -f results/v2/sparse_regime_kuairec.json ]; do sleep 60; done
pkill -f "[s]parse_regime.py" ; sleep 5
OMP_NUM_THREADS=2 python3 experiments/sparse_regime.py --qs 0.1 0.03 --floors 0.05 0.01 \
  --out results/v2/sparse_regime_kuairec_b.json > results/v2/log_sparse_kuairec_b.txt 2>&1
