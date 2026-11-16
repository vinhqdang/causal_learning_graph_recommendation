#!/bin/bash
# Revision runs: published debiased baselines on the temporal KuaiRand-27K setting, LEAF-LGN controls,
# extended DAX grid and one QP seed replicate. Labels are run one after another by "all".
cd "$(dirname "$0")/.."
D=${V9_DATA:-data/raw/kuairand_t.pt}
case "$1" in
  pub_t)          # published debiased baselines on the same grid as the LightGCN rows (lr x wd, 4 configurations)
    [ -f results/v14/learned_kuairand_t_pub.json ] || OMP_NUM_THREADS=4 V9_DATA=$D python3 experiments/v9_probes/run_learned_t.py --dataset kuairand --prop pop --seeds 5 --threads 4 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods NAVIP IPS-MF DR-JL MRDR --out results/v14/learned_kuairand_t_pub.json ;;
  leaf_lgn_ablate) bash experiments/queue_v10.sh leaf_lgn_ablate ;;
  daxwide_lgn)    V12_SMOKE=1 bash experiments/queue_v12.sh daxwide_lgn ;;
  qp_seed1)       [ -f results/v11/c_seeds/qp_s1.json ] || OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --qp_source stored --train_seed 1 --seeds 5 --threads 4 --out results/v11/c_seeds/qp_s1.json ;;
  all)
    bash "$0" pub_t > results/logs_v14/pub_t.log 2>&1
    bash "$0" leaf_lgn_ablate > results/logs_v14/leaf_lgn_ablate.log 2>&1 &
    bash "$0" daxwide_lgn > results/logs_v14/daxwide_lgn.log 2>&1
    wait
    bash "$0" qp_seed1 > results/logs_v14/qp_seed1.log 2>&1 ;;
  *) echo "usage: $0 {pub_t|leaf_lgn_ablate|daxwide_lgn|qp_seed1|all}" >&2; exit 2 ;;
esac
