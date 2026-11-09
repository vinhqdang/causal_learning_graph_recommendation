cd "$(dirname "$0")/.."
# DAX stage of paper/v7/DESIGN.md Sections 6 and 7 on the temporal all-user KuaiRand-27K setting (outputs in results/v11/).
# One label per call, e.g. nohup bash experiments/queue_v11.sh dax_lgn_post > results/v11/log_dax_lgn_post.txt 2>&1 &
# Arguments after the label are appended to the run (later flags override the label's own, as in argparse), for smoke runs.
# Big labels run inside a slot: two slot locks (flock) are shared by all calls, so at most two heavy jobs run at the same
# time (15 GB RAM). LEAF-LGN, joint and learned-backbone jobs take one slot (about 3-5 GB each at full scale); dax_qp
# (which retrains the QP family) takes both slots and runs alone. A call waits until its slot(s) are free. Every job
# resumes from its partial file (OUT.partial.json), so an interrupted label is restarted by the same command.
mkdir -p results/v11 results/v11/.slots
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
  dax_ablate)      # LEAF-LGN with the DAX ablations (the full arm and the LEAF-LGN base are run in the same trainings)
    slot env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --ablate no_tab no_life_prior no_tag_prior additive shuffle_items shuffle_tags --seeds 5 --threads 2 --out results/v11/dax_ablate.json "${@:2}" ;;
  stageB)          # gates B1, B2 and C (experiments/final_v11.py); reads the outputs above
    python3 experiments/final_v11.py ;;
  tests)           python3 -m pytest -q tests/test_dax.py ;;
  *)
    echo "usage: $0 {dax_ease|dax_lgn_post|dax_lgn_joint_s|dax_lgn_joint|dax_qp|dax_lgn_gcn|dax_ablate|stageB|tests}" >&2
    exit 2 ;;
esac
