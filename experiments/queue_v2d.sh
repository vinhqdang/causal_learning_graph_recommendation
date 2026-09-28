#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while pgrep -f "queue_v2c.sh" > /dev/null; do sleep 60; done
OMP_NUM_THREADS=4 python3 experiments/run_fat.py --dataset kuairec --prop pop --dtype float32 \
  --sections explain privacy --methods DRUP > results/v2/log_fat_kuairec_b.txt 2>&1
