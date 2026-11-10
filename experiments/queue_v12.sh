SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
# v12 (paper/v8/OUTLINE.md limitation 8 and the DAX grid edges): lifelong-aware deep baselines (DIN, SIM-hard, each with
# DAX post hoc), the extended DAX grid, and the final table. Outputs in results/v12/; the v11 queue is not edited.
# One label per call, e.g. nohup bash experiments/queue_v12.sh din_27k > results/v12/log_din_27k.txt 2>&1 &
# Arguments after the label are appended to the run (later flags override the label's own, as in argparse), for smoke runs.
# Memory: the v11 seed replicates use 6-10 GB. Every v12 heavy job first waits until no run_dax.py process is alive
# (pgrep loop below), then takes the single v12 slot (flock), so v12 jobs never overlap each other or the v11 runs and the
# total stays under 15 GB (V12_SMOKE=1 skips that wait, for the tiny smoke runs only). Each job resumes from its partial file (OUT.partial.json): rerun the same label to continue.
mkdir -p results/v12 results/v12/.slots data/raw
SEQ_DIR="${V12_SEQ_DIR:-data/raw}"   # directory of seq_27k.npz / seq_1k.npz (V12_SEQ_DIR: smoke runs on a tiny file)
heavy() {   # heavy CMD...: runs CMD in the v12 slot, after no run_dax.py process is alive
  while true; do
    exec 9>"results/v12/.slots/slot1"
    if flock -n 9; then
      while [ -z "$V12_SMOKE" ] && pgrep -f "[r]un_dax\.py" >/dev/null; do sleep 60; done   # V12_SMOKE=1: tiny smoke runs only
      "$@"
      rc=$?
      flock -u 9
      exec 9>&-
      return $rc
    fi
    exec 9>&-
    sleep 30
  done
}
done_or() {   # done_or OUT CMD...: skips CMD when OUT already exists (a finished sub-run of a label)
  local out="$1"; shift
  if [ -f "$out" ]; then echo "exists, skipped: $out"; return 0; fi
  "$@"
}
need_seq() {  # need_seq 27k|1k: the per-event sequence file of experiments/v12_build_seq.py (built once)
  case "$1" in
    27k) [ -f "$SEQ_DIR/seq_27k.npz" ] || heavy python3 experiments/v12_build_seq.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz --version 27k --out "$SEQ_DIR/seq_27k.npz" ;;
    1k)  [ -f "$SEQ_DIR/seq_1k.npz" ] || heavy python3 experiments/v12_build_seq.py --data_dir data/raw/KuaiRand-1K/data --version 1k --pool_dir data/raw/KuaiRand-1K/data --pool_suffix 1k --out "$SEQ_DIR/seq_1k.npz" ;;
  esac
}
case "$1" in
  build_seq_27k)  need_seq 27k ;;
  build_seq_1k)   need_seq 1k ;;
  din_27k)        # DIN (target attention over the 200 lifelong events) and DIN + DAX (DAX grid of 32, gamma 10 points)
    need_seq 27k; done_or results/v12/din_27k.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch din --dataset kuairand27k --seq "$SEQ_DIR/seq_27k.npz" --seeds 5 --threads 2 --out results/v12/din_27k.json "${@:2}" ;;
  sim_27k)        # SIM-hard (the 50 most recent events sharing the candidate's tag or author) and SIM + DAX
    need_seq 27k; done_or results/v12/sim_27k.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch sim --dataset kuairand27k --seq "$SEQ_DIR/seq_27k.npz" --seeds 5 --threads 2 --out results/v12/sim_27k.json "${@:2}" ;;
  din_1k)         # KuaiRand-1K (1000 users, the outpool events of the 1K archive)
    need_seq 1k; done_or results/v12/din_1k.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch din --dataset kuairand1k --seq "$SEQ_DIR/seq_1k.npz" --seeds 5 --threads 2 --out results/v12/din_1k.json "${@:2}" ;;
  sim_1k)
    need_seq 1k; done_or results/v12/sim_1k.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_din.py --arch sim --dataset kuairand1k --seq "$SEQ_DIR/seq_1k.npz" --seeds 5 --threads 2 --out results/v12/sim_1k.json "${@:2}" ;;
  daxwide_lgn)    # LEAF-LGN + DAX on the extended grid (prior 10-160 x k_life 0-1000; 200 configurations), at the stored
                  # trainings and epochs of dax_lgn_post.json (+- 3 epochs), gammas as dax_ablate_small, no Z cache. daxref_lgn is
                  # the default grid under the same reduced setting, so the grid effect is not confounded with the gammas
    done_or results/v12/daxref_lgn.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --seeds 5 --threads 2 --out results/v12/daxref_lgn.json "${@:2}"
    done_or results/v12/daxwide_lgn.json heavy env OMP_NUM_THREADS=2 python3 experiments/run_dax.py --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --prior_grid 10 20 40 80 160 --klife_grid 0 100 200 400 1000 --seeds 5 --threads 2 --out results/v12/daxwide_lgn.json "${@:2}" ;;
  daxwide_qp)     # QP family + DAX on the extended grid, at the configurations and epochs (+- 3) chosen in dax_qp.json (runs
                  # alone: 4 threads); daxref_qp is the default grid under the same setting
    done_or results/v12/daxref_qp.json heavy env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --select_from results/v11/dax_qp.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --seeds 5 --threads 4 --out results/v12/daxref_qp.json "${@:2}"
    done_or results/v12/daxwide_qp.json heavy env OMP_NUM_THREADS=4 python3 experiments/run_dax.py --backbone qp_posthoc --select_from results/v11/dax_qp.json --epoch_window 3 --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --prior_grid 10 20 40 80 160 --klife_grid 0 100 200 400 1000 --seeds 5 --threads 4 --out results/v12/daxwide_qp.json "${@:2}" ;;
  final12)        # the final table and the Holm family (experiments/final_v12.py); reads the outputs above, missing ones are notes
    python3 experiments/final_v12.py ;;
  *)
    echo "usage: $0 {build_seq_27k|build_seq_1k|din_27k|sim_27k|din_1k|sim_1k|daxwide_lgn|daxwide_qp|final12}" >&2
    exit 2 ;;
esac
