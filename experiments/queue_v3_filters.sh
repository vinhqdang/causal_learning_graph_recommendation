#!/usr/bin/env bash
# Round-3 training-free operators: every method re-run with AUC, BSPM added,
# the sample split over five draws of A (split seeds 0-4).
cd "$(dirname "$0")/.."
ALL="Pop Impute Obs EASE GF-CF BSPM IPS IPS+WC DR DRUP EASE-DR GF-CF-DR BSPM-DR DR-split DRUP-split"
case "$1" in
coat)
  python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr \
    --methods $ALL DR-5hop DRUP-5hop --out results/v3/filters_coat_given.json
  for s in 1 2 3 4; do
    python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr --split_seed $s \
      --methods DR-split DRUP-split --out results/v3/filters_coat_given_split$s.json; done ;;
yahoo)
  Y="--dataset yahoo --prop pop --xfit 10 --seeds 5 --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1"
  python3 experiments/run_filters.py $Y --methods $ALL --out results/v3/filters_yahoo_pop.json
  for s in 1 2 3 4; do
    python3 experiments/run_filters.py $Y --split_seed $s --methods DR-split DRUP-split \
      --out results/v3/filters_yahoo_pop_split$s.json; done ;;
kuairec)
  K="--dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr"
  python3 experiments/run_filters.py $K --methods $ALL --out results/v3/filters_kuairec_pop.json
  python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
    --alphas 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps lr --methods DR-5hop DRUP-5hop \
    --out results/v3/filters_kuairec_pop.json
  for s in 1 2 3 4; do
    python3 experiments/run_filters.py $K --split_seed $s --methods DR-split DRUP-split \
      --out results/v3/filters_kuairec_pop_split$s.json; done ;;
kuairand)
  R="--dataset kuairand --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.5 0.7 --floors 0.02 0.05 0.1 --lams 20 --imps add lr --cvs 0 0.5 1"
  python3 experiments/run_filters.py $R --methods $ALL --out results/v3/filters_kuairand_pop.json
  for s in 1 2 3 4; do
    python3 experiments/run_filters.py $R --split_seed $s --methods DR-split DRUP-split \
      --out results/v3/filters_kuairand_pop_split$s.json; done ;;
esac
