#!/usr/bin/env bash
# KuaiRand audit on CPU (backup for the GPU-runtime run)
cd "$(dirname "$0")/.."
L=results/v3
[ -f $L/fat_kuairand_pop_v3.done ] && exit 0
OMP_NUM_THREADS=3 python3 -u experiments/run_fat.py --dataset kuairand --prop pop --sections fairness intervene intervene_extra --dtype float32 \
  --reps 3 --filters_json $L/filters_kuairand_pop.json --tag _v3 >> $L/log_fat_kuairand.txt 2>&1 && touch $L/fat_kuairand_pop_v3.done
