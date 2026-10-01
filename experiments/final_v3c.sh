#!/usr/bin/env bash
# KuaiRec audit and privacy (CPU); the KuaiRand audit runs on the GPU runtime.
cd "$(dirname "$0")/.."
L=results/v3
[ -f $L/fat_kuairec_pop_v3.done ] && exit 0
OMP_NUM_THREADS=3 python3 experiments/run_fat.py --dataset kuairec --prop pop --sections intervene_extra privacy --dtype float32 \
  --reps 3 --filters_json $L/filters_kuairec_pop.json --tag _v3 >> $L/log_fat_kuairec.txt 2>&1 && touch $L/fat_kuairec_pop_v3.done
