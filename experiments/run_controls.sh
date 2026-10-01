#!/usr/bin/env bash
# Round-4 controls: propagation over the imputation alone (no residual), same grid as DR with cv fixed.
cd "$(dirname "$0")/.."
L=results/v3
export OMP_NUM_THREADS=3
python3 experiments/run_filters.py --dataset coat --prop given --xfit 10 --imps add lr --cvs 1 --methods ImputeProp --out $L/controls_coat_given.json >> $L/log_controls.txt 2>&1
python3 experiments/run_filters.py --dataset yahoo --prop pop --xfit 10 --seeds 5 --alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --cvs 1 --methods ImputeProp --out $L/controls_yahoo_pop.json >> $L/log_controls.txt 2>&1
python3 experiments/run_filters.py --dataset kuairec --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.3 0.5 --floors 0.05 0.1 0.2 --lams 5 20 --imps add lr --cvs 1 --methods ImputeProp --out $L/controls_kuairec_pop.json >> $L/log_controls.txt 2>&1
