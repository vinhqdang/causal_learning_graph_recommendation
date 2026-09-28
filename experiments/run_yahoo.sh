#!/usr/bin/env bash
# Yahoo! R3 training-free estimators. The DR family is searched over the
# control-variate weight cv (0 = IPS end, 1 = DR) for a given propensity model.
# Usage: run_yahoo.sh <prop> "<methods>" <tag>
set -e
GRID="--alphas 0.5 0.7 --floors 0.1 0.2 --lams 20 --imps add lr --degs W Yhat --cvs 0 0.25 0.5 1 --seeds 5"
python3 experiments/run_filters.py --dataset yahoo --prop "$1" --methods $2 $GRID \
  --out results/filters_yahoo_$1$3.json > results/log_filters_yahoo_$1$3.txt 2>&1
