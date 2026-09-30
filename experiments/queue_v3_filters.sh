#!/usr/bin/env bash
# Round-3 training-free operators: every method re-run with AUC, BSPM added,
# the sample split over five draws of A (split seeds 0-4). Methods run in
# groups; a group whose methods are already in the result file is skipped, so
# the script can be restarted.
cd "$(dirname "$0")/.."
MGROUPS=("Pop Impute Obs EASE GF-CF BSPM" "IPS IPS+WC" "DR DRUP" "EASE-DR GF-CF-DR BSPM-DR" "DR-split DRUP-split")
BGROUPS=("Pop Impute Obs EASE GF-CF BSPM" "IPS IPS+WC DR DRUP EASE-DR GF-CF-DR BSPM-DR" "DR-split DRUP-split")
run_groups() {   # $1 = common args, $2 = out
  for g in "${MGROUPS[@]}"; do python3 experiments/run_filters.py $1 --methods $g --out $2; done
}
run_big() {      # large datasets: one run shares the cross-fitted imputations of all debiased operators
  for g in "${BGROUPS[@]}"; do python3 experiments/run_filters.py $1 --methods $g --out $2; done
}
run_seeds() {
  for s in 1 2 3 4; do python3 experiments/run_filters.py $1 --split_seed $s --methods DR-split DRUP-split \
      --out ${2%.json}_split$s.json; done
}
case "$1" in
coat)
  C="--dataset coat --prop given --xfit 10 --imps add lr"
  run_groups "$C" results/v3/filters_coat_given.json
  python3 experiments/run_filters.py $C --methods DR-5hop DRUP-5hop --out results/v3/filters_coat_given.json
  run_seeds "$C" results/v3/filters_coat_given.json ;;
yahoo)
  Y="--dataset yahoo --prop pop --xfit 10 --seeds 5 --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1"
  run_groups "$Y" results/v3/filters_yahoo_pop.json
  run_seeds "$Y" results/v3/filters_yahoo_pop.json ;;
kuairand)
  R="--dataset kuairand --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.5 0.7 --floors 0.02 0.05 0.1 --lams 20 --imps add lr --cvs 0 0.5 1"
  run_big "$R" results/v3/filters_kuairand_pop.json
  run_seeds "$R" results/v3/filters_kuairand_pop.json ;;
kuairec)
  K="--dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr"
  run_big "$K" results/v3/filters_kuairec_pop.json
  python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
    --alphas 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps lr --methods DR-5hop DRUP-5hop \
    --out results/v3/filters_kuairec_pop.json
  run_seeds "$K" results/v3/filters_kuairec_pop.json ;;
esac
