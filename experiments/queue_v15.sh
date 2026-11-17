#!/bin/bash
# Third-review experiments: examination-off control, stand-alone rankers, matched-capacity DIN/SIM,
# training-seed replicates of DIN + DAX and PW-LightGCN + DAX.
cd "$(dirname "$0")/.."
mkdir -p results/v15 results/logs_v15
run() { [ -f "$1" ] && return 0; shift; "$@"; }
case "$1" in
  none)     run results/v15/none_noexam.json env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone none --ablate no_exam --seeds 5 --threads 2 --out results/v15/none_noexam.json ;;
  noexam)   run results/v15/dax_ablate_noexam.json env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --ablate no_exam --no_zcache --seeds 5 --threads 2 --out results/v15/dax_ablate_noexam.json ;;
  sim64)    run results/v15/sim_27k_d64.json env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch sim --dim 64 --hidden 64 --mlp 64 --lrs 1e-2 --wds 1e-4 1e-5 --epochs 20 --patience 2 --threads 2 --out results/v15/sim_27k_d64.json ;;
  din64)    run results/v15/din_27k_d64.json env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch din --dim 64 --hidden 64 --mlp 64 --lrs 1e-2 --wds 1e-4 1e-5 --epochs 20 --patience 2 --no_zcache --threads 2 --out results/v15/din_27k_d64.json ;;
  din_s)    for s in 1 2; do run results/v15/din_27k_s$s.json env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch din --train_seed $s --threads 2 --out results/v15/din_27k_s$s.json; done ;;
  pw_s)     for s in 1 2; do run results/v15/pw_s$s.json env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --qp_source stored --select_from results/v11/dax_qp.json --epoch_window 3 --train_seed $s --seeds 5 --threads 4 --out results/v15/pw_s$s.json; done ;;
  laneA)    bash "$0" none > results/logs_v15/none.log 2>&1; bash "$0" noexam > results/logs_v15/noexam.log 2>&1
            bash "$0" din64 > results/logs_v15/din64.log 2>&1; bash "$0" din_s > results/logs_v15/din_s.log 2>&1 ;;
  laneB)    bash "$0" sim64 > results/logs_v15/sim_d64.log 2>&1; bash "$0" pw_s > results/logs_v15/pw_s.log 2>&1 ;;
esac
