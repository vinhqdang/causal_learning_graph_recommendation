#!/usr/bin/env bash
# Round-3 trained baselines: per-split early stopping (no split sees another
# split's data), wider grids for graph models, new baselines. Parts run in
# parallel and are merged by experiments/merge_learned.py.
cd "$(dirname "$0")/.."
KU="--dataset kuairec --prop pop --seeds 5 --epochs 30 --patience 3 --batch 32768 --pairs_per_epoch 2000000 \
    --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 --seed_check 2 --wide_layers 1 2 3 --wide_dims 64 128"
YA="--dataset yahoo --prop pop --seeds 5 --epochs 60 --patience 5 --batch 1024 --seed_check 3 \
    --wide_layers 1 2 3 --wide_dims 64 128"
case "$1" in
kuA) OMP_NUM_THREADS=2 python3 experiments/run_learned.py $KU --threads 2 \
      --methods LightGCN-pt iALS SimGCL DR-JL MRDR MACR MF IPS-MF DR-MF --wide LightGCN-pt \
      --out results/v3/learned_kuairec_pop_A.json ;;
kuC) OMP_NUM_THREADS=2 python3 experiments/run_learned.py $KU --threads 2 --methods iALS \
      --out results/v3/learned_kuairec_pop_C.json ;;
kuB) OMP_NUM_THREADS=2 python3 experiments/run_learned.py $KU --threads 2 \
      --methods LightGCN r-AdjNorm NAVIP DR-LightGCN BPR-MF PDA --wide LightGCN \
      --out results/v3/learned_kuairec_pop_B.json ;;
yaA) OMP_NUM_THREADS=1 python3 experiments/run_learned.py $YA --threads 1 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 \
      --methods r-AdjNorm --wide r-AdjNorm --out results/v3/learned_yahoo_pop_A.json ;;
yaB) OMP_NUM_THREADS=1 python3 experiments/run_learned.py $YA --threads 1 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 \
      --methods LightGCN SimGCL --wide LightGCN --out results/v3/learned_yahoo_pop_B.json ;;
yaC) OMP_NUM_THREADS=1 python3 experiments/run_learned.py $YA --threads 1 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 \
      --methods NAVIP LightGCN-pt DR-LightGCN --wide NAVIP --out results/v3/learned_yahoo_pop_C.json ;;
yaD) OMP_NUM_THREADS=1 python3 experiments/run_learned.py $YA --threads 1 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 1e-3 \
      --methods iALS MF IPS-MF DR-MF BPR-MF PDA DR-JL MRDR MACR --out results/v3/learned_yahoo_pop_D.json ;;
rand) OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairand --prop pop --seeds 5 --threads 2 \
      --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 --seed_check 2 \
      --methods LightGCN-pt LightGCN r-AdjNorm SimGCL iALS MF DR-MF BPR-MF NAVIP DR-JL MACR \
      --out results/v3/learned_kuairand_pop.json ;;
coat) OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 \
      --dims 32 64 --epochs 100 --batch 256 --seed_check 3 --wide_layers 1 2 3 --wide_dims 32 64 \
      --methods MF IPS-MF DR-MF BPR-MF PDA LightGCN LightGCN-pt r-AdjNorm NAVIP DR-LightGCN DR-JL MRDR MACR SimGCL iALS \
      --wide LightGCN LightGCN-pt r-AdjNorm NAVIP DR-LightGCN SimGCL --out results/v3/learned_coat_given.json ;;
esac
