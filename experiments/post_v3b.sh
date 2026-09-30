#!/usr/bin/env bash
# Serialised end of round 3 (memory): once no training-free run is left, the
# KuaiRand debiased operators, then the KuaiRec and KuaiRand audits.
cd "$(dirname "$0")/.."
L=results/v3
until [ -f $L/filters_kuairec_pop_split4.json ] && [ -f $L/filters_kuairand_pop_split4.json ] \
      && ! pgrep -f "[r]un_filters.py" > /dev/null; do sleep 120; done
R="--dataset kuairand --prop pop --xfit 10 --seeds 5 --dtype float32 --alphas 0.5 0.7 --floors 0.02 0.05 0.1 --lams 20 --imps add lr --cvs 0 0.5 1"
OMP_NUM_THREADS=3 python3 experiments/run_filters.py $R --methods IPS IPS+WC DR DRUP EASE-DR GF-CF-DR BSPM-DR \
  --out $L/filters_kuairand_pop.json >> $L/log_filters_kuairand.txt 2>&1
fat() {  # dataset prop sections threads
  [ -f $L/fat_$1_$2_v3.done ] && return
  OMP_NUM_THREADS=$4 python3 experiments/run_fat.py --dataset $1 --prop $2 --sections $3 --dtype float32 \
    --reps 3 --filters_json $L/filters_$1_$2.json --tag _v3 >> $L/log_fat_$1.txt 2>&1 && touch $L/fat_$1_$2_v3.done
}
fat kuairec pop "intervene_extra privacy" 3
fat kuairand pop "fairness intervene intervene_extra" 3
