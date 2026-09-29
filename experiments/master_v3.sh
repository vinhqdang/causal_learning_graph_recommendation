#!/usr/bin/env bash
# Round-3 scheduler: four lanes that start once earlier jobs have finished.
cd "$(dirname "$0")/.."
wait_for() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
L=results/v3
lane1() { wait_for "[l]earned_kuairec_pop_B"
          experiments/queue_v3_learned.sh yaA > $L/log_learned_yahoo_yaA.txt 2>&1
          experiments/queue_v3_learned.sh yaC > $L/log_learned_yahoo_yaC.txt 2>&1
          experiments/queue_v3_learned.sh rand > $L/log_learned_kuairand.txt 2>&1; }
lane2() { wait_for "[l]earned_kuairec_pop_A"
          experiments/queue_v3_learned.sh yaB > $L/log_learned_yahoo_yaB.txt 2>&1
          experiments/queue_v3_learned.sh yaD > $L/log_learned_yahoo_yaD.txt 2>&1; }
lane3() { wait_for "[s]emisynth_kuairec"
          experiments/queue_v3_learned.sh coat > $L/log_learned_coat.txt 2>&1
          OMP_NUM_THREADS=1 experiments/queue_v3_filters.sh yahoo > $L/log_filters_yahoo.txt 2>&1
          OMP_NUM_THREADS=1 experiments/queue_v3_filters.sh kuairand > $L/log_filters_kuairand.txt 2>&1; }
lane4() { wait_for "[q]ueue_v3_filters.sh coat"
          OMP_NUM_THREADS=1 python3 experiments/run_fat.py --dataset coat --prop given --sections intervene_extra privacy \
              --filters_json $L/filters_coat_given.json --tag _v3 > $L/log_fat_coat.txt 2>&1
          wait_for "[l]earned_kuairec_pop_"
          OMP_NUM_THREADS=2 experiments/queue_v3_filters.sh kuairec > $L/log_filters_kuairec.txt 2>&1; }
lane1 & lane2 & lane3 & lane4 &
wait
