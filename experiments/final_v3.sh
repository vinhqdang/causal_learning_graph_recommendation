#!/usr/bin/env bash
# End of round 3, strictly one training-free run at a time (memory):
# KuaiRec filters, KuaiRand filters, then the KuaiRec and KuaiRand audits.
# Every step skips what is already done, so the script can be restarted.
cd "$(dirname "$0")/.."
L=results/v3
OMP_NUM_THREADS=3 experiments/queue_v3_filters.sh kuairec >> $L/log_filters_kuairec.txt 2>&1
OMP_NUM_THREADS=3 experiments/queue_v3_filters.sh kuairand >> $L/log_filters_kuairand.txt 2>&1
fat() {  # dataset prop sections threads
  [ -f $L/fat_$1_$2_v3.done ] && return
  OMP_NUM_THREADS=$4 python3 experiments/run_fat.py --dataset $1 --prop $2 --sections $3 --dtype float32 \
    --reps 3 --filters_json $L/filters_$1_$2.json --tag _v3 >> $L/log_fat_$1.txt 2>&1 && touch $L/fat_$1_$2_v3.done
}
fat kuairec pop "intervene_extra privacy" 3
fat kuairand pop "fairness intervene intervene_extra" 3
