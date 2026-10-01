#!/usr/bin/env bash
# KuaiRand audit on CPU. fairness and intervene are in fat_kuairand_pop_v3.json; the memory-heavy
# intervene_extra section runs alone into fat_kuairand_pop_v3x.json.
cd "$(dirname "$0")/.."
L=results/v3
[ -f $L/fat_kuairand_pop_v3x.done ] && exit 0
OMP_NUM_THREADS=4 python3 -u experiments/run_fat.py --dataset kuairand --prop pop --sections intervene_extra --dtype float32 \
  --reps 2 --filters_json $L/filters_kuairand_pop.json --tag _v3x >> $L/log_fat_kuairand.txt 2>&1 && touch $L/fat_kuairand_pop_v3x.done
