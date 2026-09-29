#!/usr/bin/env bash
# Sample-split variants (nuisances on 20% of the pairs, Assumption 2 on the rest).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr \
  --methods DR-split DRUP-split --out results/v2/filters_coat_given.json > results/v2/log_split_coat.txt 2>&1
python3 experiments/significance.py --dataset coat --prop given --ref DRUP-split --others DR-split DRUP DR \
  --out results/v2/significance_split_coat_given.json >> results/v2/log_split_coat.txt 2>&1
python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 \
  --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 0 0.25 0.5 1 \
  --methods DR-split DRUP-split --out results/v2/filters_yahoo_pop.json > results/v2/log_split_yahoo.txt 2>&1
python3 experiments/significance.py --dataset yahoo --prop pop --ref DRUP-split --others DR-split DRUP DR \
  --out results/v2/significance_split_yahoo_pop.json >> results/v2/log_split_yahoo.txt 2>&1
while pgrep -f "[d]ump_bank results/v2/bank_filters_kuairec" > /dev/null; do sleep 30; done
export OMP_NUM_THREADS=2
python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 \
  --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr \
  --methods DR-split DRUP-split --out results/v2/filters_kuairec_pop.json > results/v2/log_split_kuairec.txt 2>&1
python3 experiments/significance.py --dataset kuairec --prop pop --ref DRUP-split --others DR-split DRUP DR \
  --out results/v2/significance_split_kuairec_pop.json >> results/v2/log_split_kuairec.txt 2>&1
python3 experiments/bench_khop.py --threads 4 > results/v2/log_bench_khop.txt 2>&1
