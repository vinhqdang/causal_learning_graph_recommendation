cd "$(dirname "$0")/.."
# Stage 1 of paper/v6/DESIGN.md Section 7 on the temporal all-user KuaiRand-27K setting (outputs in results/v10/).
# One label per call, e.g. nohup bash experiments/queue_v10.sh leaf_lgn > results/v10/log_leaf_lgn.txt 2>&1 &
# Arguments after the label are appended to the run (later flags override the label's own, as in argparse), for smoke runs.
# Big labels run inside a slot: two slot locks (flock) are shared by all calls, so at most two big jobs run at the
# same time (15 GB RAM). LEAF jobs take one slot (about 4-5 GB each at full scale); base_t, qp_t and leaf_full
# (about 7-8 GB, 6-9 GB) take both slots and run alone. A call waits until its slot(s) are free. Every job resumes from its
# partial file (run_learned: OUT.<method>.partial.json; run_leaf: OUT.partial.json), so an interrupted label is
# restarted by the same command.
mkdir -p results/v10 results/v10/.slots
slot() {   # slot CMD...: runs CMD while holding one of the two slot locks
  while true; do
    for s in 1 2; do
      exec 9>"results/v10/.slots/slot$s"
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
slot_excl() {   # slot_excl CMD...: runs CMD holding both slot locks (the job runs alone; used by leaf_full)
  while true; do
    exec 8>"results/v10/.slots/slot1"
    exec 9>"results/v10/.slots/slot2"
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
T0=1650556800000
case "$1" in
  cache_t)        # temporal all-user setting in the dict format of run_learned (exists after the first build)
    [ -f data/raw/kuairand_t.pt ] || python3 experiments/v9_make_cache.py --train_until $T0 --n_users 0 --out data/raw/kuairand_t.pt ;;
  base_t)         # fully tuned LightGCN, r-AdjNorm, iALS and iALS + additive prior (run_learned, unchanged code;
                  # about 7-8 GB at full scale, so it runs alone)
    slot_excl env OMP_NUM_THREADS=2 V9_DATA=${V9_DATA:-data/raw/kuairand_t.pt} python3 experiments/v9_probes/run_learned_t.py --dataset kuairand --prop pop --prior add --seeds 5 --threads 2 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods LightGCN r-AdjNorm iALS iALS-p --out results/v10/learned_kuairand_t_base.json "${@:2}" ;;
  qp_t)           # QP family (DR-BPR-LGN-1 / -p2 / -p) on the same grid as the LightGCN rows (runs alone, as base_t)
    slot_excl env OMP_NUM_THREADS=2 V9_DATA=${V9_DATA:-data/raw/kuairand_t.pt} python3 experiments/v9_probes/run_learned_t.py --dataset kuairand --prop pop --seeds 5 --threads 2 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods DR-BPR-LGN-1 DR-BPR-LGN-p2 DR-BPR-LGN-p --out results/v10/learned_kuairand_t_qp.json "${@:2}" ;;
  leaf_tied)      # closed-form LEAF (probe form: validation-selected channel weights)
    slot env OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand_t --variant tied --gamma grid --ablate full --seeds 5 --threads 2 --w 0 0.25 0.5 1 --lam 1000 3000 10000 --beta 0 0.25 0.5 --s 5 20 --out results/v10/leaf_tied.json "${@:2}" ;;
  leaf_full)      # closed-form LEAF with free lifelong-key weights W; runs alone (about 6-9 GB at full scale)
    slot_excl env OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand_t --variant full --seeds 5 --threads 2 --w 0 0.25 0.5 1 --lam 1000 3000 10000 --beta 0 0.25 0.5 --s 5 20 --lam_phi 1e2 1e3 1e4 --families tag tag+author_id tag+author_id+music_id --min_imp 20 --out results/v10/leaf_full.json "${@:2}" ;;
  leaf_lgn)       # LEAF-LGN, channel weights trained jointly with the BPR objective
    slot env OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand_t --variant lgn --ablate full --seeds 5 --threads 2 --w 0 0.25 0.5 --s 5 --lrs 1e-2 3e-3 --wds 1e-4 1e-5 --dim 64 --layers 2 --epochs 40 --patience 3 --batch 8192 --out results/v10/leaf_lgn.json "${@:2}" ;;
  leaf_ablate)    # closed-form ablations: shuffled users, no tag channel, no author channel, and the design's others
    slot env OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand_t --variant tied --gamma grid --ablate shuffle_users no_tag no_author no_lifelong no_policy lift_only familiarity_only no_music --seeds 5 --threads 2 --w 0 0.25 0.5 1 --lam 1000 3000 10000 --beta 0 0.25 0.5 --s 5 20 --out results/v10/leaf_tied_ablate.json "${@:2}" ;;
  leaf_lgn_ablate) # LEAF-LGN without lifelong channels, with shuffled users, and without the policy channel
    slot env OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand_t --variant lgn --ablate no_lifelong shuffle_users no_policy --seeds 5 --threads 2 --w 0 0.25 0.5 --s 5 --lrs 1e-2 3e-3 --wds 1e-4 1e-5 --dim 64 --layers 2 --epochs 40 --patience 3 --batch 8192 --out results/v10/leaf_lgn_ablate.json "${@:2}" ;;
  tests)          python3 -m pytest -q tests ;;
  *)
    echo "usage: $0 {cache_t|base_t|qp_t|leaf_tied|leaf_full|leaf_lgn|leaf_ablate|leaf_lgn_ablate|tests}" >&2
    exit 2 ;;
esac
