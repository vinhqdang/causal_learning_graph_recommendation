#!/usr/bin/env bash
# Round-3 scheduler. Every step resumes or skips finished work, so the whole
# script can be restarted after a container restart.
cd "$(dirname "$0")/.."
L=results/v3
lane1() { experiments/queue_v3_learned.sh kuB >> $L/log_learned_kuairec_kuB.txt 2>&1
          experiments/queue_v3_learned.sh yaA >> $L/log_learned_yahoo_yaA.txt 2>&1
          experiments/queue_v3_learned.sh yaC >> $L/log_learned_yahoo_yaC.txt 2>&1
          experiments/queue_v3_learned.sh rand >> $L/log_learned_kuairand.txt 2>&1; }
lane2() { experiments/queue_v3_learned.sh kuA >> $L/log_learned_kuairec_kuA.txt 2>&1
          experiments/queue_v3_learned.sh yaB >> $L/log_learned_yahoo_yaB.txt 2>&1
          experiments/queue_v3_learned.sh yaD >> $L/log_learned_yahoo_yaD.txt 2>&1; }
lane3() { experiments/queue_semisynth.sh
          experiments/queue_v3_learned.sh coat >> $L/log_learned_coat.txt 2>&1
          OMP_NUM_THREADS=1 experiments/queue_v3_filters.sh yahoo >> $L/log_filters_yahoo.txt 2>&1
          OMP_NUM_THREADS=1 experiments/queue_v3_filters.sh kuairand >> $L/log_filters_kuairand.txt 2>&1; }
lane4() { OMP_NUM_THREADS=1 experiments/queue_v3_filters.sh coat >> $L/log_filters_coat.txt 2>&1
          [ -f $L/fat_coat_given_v3.done ] || { OMP_NUM_THREADS=1 python3 experiments/run_fat.py --dataset coat \
              --prop given --sections intervene_extra privacy --filters_json $L/filters_coat_given.json --tag _v3 \
              >> $L/log_fat_coat.txt 2>&1 && touch $L/fat_coat_given_v3.done; }
          while pgrep -f "[l]earned_kuairec_pop_" > /dev/null; do sleep 60; done
          OMP_NUM_THREADS=2 experiments/queue_v3_filters.sh kuairec >> $L/log_filters_kuairec.txt 2>&1; }
lane1 & lane2 & lane3 & lane4 &
wait
