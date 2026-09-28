#!/usr/bin/env bash
# Protocol v2: cross-fitted nuisances, imputation-based degree weights,
# global scale constants, per-user metrics. Usage: run_v2.sh <dataset>
set -e
mkdir -p results/v2
case "$1" in
coat)
  python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr \
    --methods Pop Impute Obs EASE GF-CF IPS IPS+WC DR DRUP EASE-DR GF-CF-DR DR-5hop DRUP-5hop \
    --out results/v2/filters_coat_given.json ;;
yahoo)
  python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 \
    --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1 \
    --methods Pop Impute Obs EASE GF-CF IPS IPS+WC DR DRUP EASE-DR GF-CF-DR \
    --out results/v2/filters_yahoo_pop.json ;;
kuairec)
  python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
    --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr \
    --methods Pop Impute Obs EASE GF-CF IPS IPS+WC DR DRUP EASE-DR GF-CF-DR \
    --out results/v2/filters_kuairec_pop.json
  # five hops: low-rank imputation and alpha = 0.5 only (cost)
  python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
    --alphas 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps lr \
    --methods DR-5hop DRUP-5hop --out results/v2/filters_kuairec_pop.json ;;
esac
