#!/usr/bin/env bash
# Round-3 audit and privacy runs, each once its filter results are complete.
cd "$(dirname "$0")/.."
L=results/v3
fat() {  # dataset prop sections threads
  [ -f $L/fat_$1_$2_v3.done ] && return
  while [ ! -f $L/filters_$1_$2_split4.json ]; do sleep 120; done
  OMP_NUM_THREADS=$4 python3 experiments/run_fat.py --dataset $1 --prop $2 --sections $3 --dtype float32 \
    --reps 3 --filters_json $L/filters_$1_$2.json --tag _v3 >> $L/log_fat_$1.txt 2>&1 && touch $L/fat_$1_$2_v3.done
}
( fat yahoo pop "intervene_extra privacy" 1; fat kuairand pop "fairness intervene intervene_extra" 1 ) &
( while pgrep -f "[r]un_filters.py --dataset kuairec" > /dev/null || [ ! -f $L/filters_kuairec_pop_split4.json ]; do sleep 120; done
  fat kuairec pop "intervene_extra privacy" 2 ) &
wait
