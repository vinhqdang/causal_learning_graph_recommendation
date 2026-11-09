SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
# DAX stage of paper/v7/DESIGN.md Sections 6 and 7 on the temporal all-user KuaiRand-27K setting (outputs in results/v11/).
# One label per call, e.g. nohup bash experiments/queue_v11.sh dax_lgn_post > results/v11/log_dax_lgn_post.txt 2>&1 &
# Arguments after the label are appended to the run (later flags override the label's own, as in argparse), for smoke runs.
# Big labels run inside a slot: two slot locks (flock) are shared by all calls, so at most two heavy jobs run at the same
# time (15 GB RAM). LEAF-LGN, joint and learned-backbone jobs take one slot (about 3-5 GB each at full scale); dax_qp
# (which retrains the QP family) takes both slots and runs alone. A call waits until its slot(s) are free. Every job
# resumes from its partial file (OUT.partial.json), so an interrupted label is restarted by the same command.
T0=1650556800000
mkdir -p results/v11 results/v11/.slots results/v11/c_seeds
slot() {   # slot CMD...: runs CMD while holding one of the two slot locks
  while true; do
    for s in 1 2; do
      exec 9>"results/v11/.slots/slot$s"
      if flock -n 9; then
        "$@"
        rc=$?
        flock -u 9
        exec 9>&-
        return $rc
      fi
      exec 9>&-
    done
    sleep 30
  done
}
slot_excl() {   # slot_excl CMD...: runs CMD holding both slot locks (the job runs alone)
  while true; do
    exec 8>"results/v11/.slots/slot1"
    exec 9>"results/v11/.slots/slot2"
    if flock -n 8 && flock -n 9; then
      "$@"
      rc=$?
      flock -u 8
      flock -u 9
      exec 8>&- 9>&-
      return $rc
    fi
    exec 8>&- 9>&-
    sleep 30
  done
}
done_or() {   # done_or OUT CMD...: skips CMD when OUT already exists (a finished sub-run of a label)
  local out="$1"; shift
  if [ -f "$out" ]; then echo "exists, skipped: $out"; return 0; fi
  "$@"
}
case "$1" in
  dax_ease)        # E0 / E1 of the probes with the full EASE grid (lam, w, beta) and the DAX arm post hoc
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone ease --seeds 5 --threads 2 --out results/v11/dax_ease.json "${@:2}" ;;
  dax_lgn_post)    # LEAF-LGN trained alone; DAX (32 configurations) and gamma selected post hoc at every epoch
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --seeds 5 --threads 2 --out results/v11/dax_lgn_post.json "${@:2}" ;;
  dax_lgn_joint_s) # LEAF-LGN + DAX with the DAX weight trained jointly, one DAX configuration (prior 80, k_life 100, tab on)
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode joint --dax_grid small --seeds 5 --threads 2 --out results/v11/dax_lgn_joint_small.json "${@:2}" ;;
  dax_lgn_joint)   # LEAF-LGN + DAX jointly trained, DAX grid prior {40, 80, 160} x k_life {0, 100}
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode joint --dax_grid full --seeds 5 --threads 2 --out results/v11/dax_lgn_joint.json "${@:2}" ;;
  dax_qp)          # the QP family retrained at its stored configurations, with DAX post hoc (the fair baseline); runs alone
    slot_excl env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --seeds 5 --threads 4 --out results/v11/dax_qp.json "${@:2}" ;;
  dax_lgn_gcn)     # LightGCN retrained at its stored configurations, with DAX post hoc
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone lightgcn --mode posthoc --seeds 5 --threads 2 --out results/v11/dax_lightgcn.json "${@:2}" ;;
  dax_ablate)      # full-grid DAX ablations on LEAF-LGN over every epoch (not run: the cut-down label below is the gate B2 run)
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --ablate no_tab no_life_prior no_tag_prior additive shuffle_items shuffle_tags --seeds 5 --threads 2 --out results/v11/dax_ablate_full.json "${@:2}" ;;
  dax_ablate_small) # gate B2, cut down (paper/v7/DESIGN.md Section 7): the LEAF-LGN+DAX (lr, wd, w) trainings chosen per split
                   # in dax_lgn_post.json (2 trainings), each split reads the epochs e_s +- 3 of its chosen epoch; DAX grid of
                   # 32 for the full arm and each ablation (16 for the switched ones, matched 32 for additive), gammas 5 points,
                   # shuffles over 3 permutation seeds; no Z cache (memory)
    done_or results/v11/dax_ablate.json slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --ablate no_tab no_life_prior no_tag_prior additive shuffle_items shuffle_tags --perm_seeds 0 1 2 --no_zcache --seeds 5 --threads 2 --out results/v11/dax_ablate.json "${@:2}" ;;
  c_seeds_lgn)     # Stage C seed replicates of LEAF-LGN+DAX: train seeds 1, 2, 3 at the trainings the main run chose per split
                   # (dax_lgn_post.json); DAX, gamma and epoch re-selected per split on validation (full DAX grid, all epochs)
    for s in 1 2 3; do
      done_or results/v11/c_seeds/lgn_s$s.json slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --train_seed $s --seeds 5 --threads 2 --out results/v11/c_seeds/lgn_s$s.json "${@:2}"
    done ;;
  c_seeds_qp)      # Stage C seed replicates of QP family + DAX: train seeds 1, 2, 3 at the stored configurations (10 trainings
                   # per seed, as dax_qp); runs alone (both slots)
    for s in 1 2 3; do
      done_or results/v11/c_seeds/qp_s$s.json slot_excl env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --qp_source stored --train_seed $s --seeds 5 --threads 4 --out results/v11/c_seeds/qp_s$s.json "${@:2}"
    done ;;
  c_seeds)         # both replicate families in sequence, then the merge into results/v11/c_seeds.json
    bash "$SELF" c_seeds_lgn "${@:2}"; bash "$SELF" c_seeds_qp "${@:2}"; bash "$SELF" c_seeds_merge ;;
  c_seeds_merge)   python3 experiments/v11_merge_seeds.py ;;
  c_1k_data)       # KuaiRand-1K inputs: the lifelong tables on the 1K pool and the temporal cache of the learned runs
    [ -f data/raw/lifelong_1k.npz ] || python3 experiments/v9_build_lifelong.py --archive data/raw/dl_kr1k/KuaiRand-1K.tar.gz --version 1k --pool_dir data/raw/KuaiRand-1K/data --pool_suffix 1k --out data/raw/lifelong_1k.npz
    [ -f data/raw/kuairand_1k_t.pt ] || python3 experiments/v9_make_cache.py --version 1k --train_until $T0 --n_users 0 --out data/raw/kuairand_1k_t.pt ;;
  c_1k_base)       # KuaiRand-1K: LightGCN, r-AdjNorm and the QP methods on the 27K grids (run_learned, unchanged)
    done_or results/v11/c_1k_base.json slot env OMP_NUM_THREADS=2 V9_DATA=data/raw/kuairand_1k_t.pt python3 experiments/v9_probes/run_learned_t.py --dataset kuairand --prop pop --seeds 5 --threads 2 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods LightGCN r-AdjNorm DR-BPR-LGN-1 DR-BPR-LGN-p2 DR-BPR-LGN-p --out results/v11/c_1k_base.json "${@:2}" ;;
  c_1k_lgn)        # KuaiRand-1K: LEAF-LGN and LEAF-LGN+DAX (full grids as dax_lgn_post)
    done_or results/v11/c_1k_lgn.json slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --dataset kuairand1k --seeds 5 --threads 2 --out results/v11/c_1k_lgn.json "${@:2}" ;;
  c_1k_qp)         # KuaiRand-1K: QP family and QP family + DAX over the whole QP grid (no stored 1K selection)
    done_or results/v11/c_1k_qp.json slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone qp_posthoc --qp_source grid --dataset kuairand1k --seeds 5 --threads 2 --out results/v11/c_1k_qp.json "${@:2}" ;;
  c_1k)            # KuaiRand-1K, all sub-labels in sequence
    bash "$SELF" c_1k_data; bash "$SELF" c_1k_base "${@:2}"; bash "$SELF" c_1k_lgn "${@:2}"; bash "$SELF" c_1k_qp "${@:2}" ;;
  stageB)          # gates B1, B2 and C (experiments/final_v11.py); reads the outputs above
    python3 experiments/final_v11.py ;;
  tests)           python3 -m pytest -q tests/test_dax.py ;;
  *)
    echo "usage: $0 {dax_ease|dax_lgn_post|dax_lgn_joint_s|dax_lgn_joint|dax_qp|dax_lgn_gcn|dax_ablate|dax_ablate_small|c_seeds|c_seeds_lgn|c_seeds_qp|c_seeds_merge|c_1k|c_1k_data|c_1k_base|c_1k_lgn|c_1k_qp|stageB|tests}" >&2
    exit 2 ;;
esac
