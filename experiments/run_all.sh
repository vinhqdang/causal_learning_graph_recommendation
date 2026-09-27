#!/usr/bin/env bash
# Full pipeline for one dataset: training-free estimators, trained baselines,
# FAT experiments. Usage: experiments/run_all.sh coat given | kuairec pop
set -e
DS=${1:-coat}; PROP=${2:-given}; shift 2 || true
if [ "$DS" = "coat" ]; then
  python3 experiments/run_filters.py --dataset coat --prop "$PROP"
  python3 experiments/run_learned.py --dataset coat --prop "$PROP" --seeds 10 --threads 4
else
  python3 experiments/run_filters.py --dataset kuairec --prop "$PROP" --dtype float32 \
    --alphas 0.3 0.5 --floors 0.02 0.05 0.1 0.2 --lams 5 20 --seeds 5
  python3 experiments/run_learned.py --dataset kuairec --prop "$PROP" --seeds 3 --threads 4 \
    --epochs 15 --batch 65536 --lrs 1e-2 --wds 1e-4 --pairs_per_epoch 10000000 --floor 0.05
fi
python3 experiments/run_fat.py --dataset "$DS" --prop "$PROP" --dtype $([ "$DS" = coat ] && echo float64 || echo float32)
