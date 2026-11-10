#!/usr/bin/env bash
# Runs one heavy label of the v11 / v12 queues on a Colab GPU VM (--device cuda). The arguments are those of
# experiments/queue_v12.sh and experiments/queue_v11.sh; only the device and the output directory differ.
#
# Labels:
#   din_27k      DIN and DIN + DAX                                          (queue_v12 din_27k)
#   sim_27k      SIM-hard and SIM + DAX                                     (queue_v12 sim_27k)
#   c_seeds_qp   QP family + DAX, train seeds 1, 2, 3                        (queue_v11 c_seeds_qp)
#   daxwide_lgn  LEAF-LGN + DAX: default-grid reference, then extended grid  (queue_v12 daxwide_lgn)
#   daxwide_qp   QP family + DAX: default-grid reference, then extended grid (queue_v12 daxwide_qp)
#
# Outputs go to /content/out/ (OUT), under the same relative names as results/ locally (v11/..., v12/...).
# Resume: run the same label again. A sub-run whose output exists is skipped; an unfinished one continues from
# OUT.partial.json, which the run writes next to its output.
#
# Uploads, from the local repo root, to the paths on the VM (REPO defaults to /content/repo):
#   1. code: tar -czf code.tgz experiments drup tests
#      -> upload code.tgz to /content, then: mkdir -p /content/repo && tar -xzf /content/code.tgz -C /content/repo
#   2. data/raw/seq_27k.npz                     -> /content/repo/data/raw/seq_27k.npz               (din_27k, sim_27k)
#   3. data/raw/lifelong_27k.npz                -> /content/repo/data/raw/lifelong_27k.npz           (all labels)
#   4. data/raw/KuaiRand-Pure/data/ (directory) -> /content/repo/data/raw/KuaiRand-Pure/data/   (all labels: the
#      loader reads the five CSV files of this directory, and the video features of the DAX content codes)
#   5. data/raw/kuairand_t.pt                   -> /content/repo/data/raw/kuairand_t.pt
#      (optional: only a consistency check of the test lists in run_dax.py, skipped when absent; 1.6 GB)
#   6. results/v11/dax_lgn_post.json            -> /content/repo/results/v11/dax_lgn_post.json
#      (daxwide_lgn, --select_from)
#   7. results/v11/dax_qp.json                  -> /content/repo/results/v11/dax_qp.json
#      (daxwide_qp, --select_from)
#   8. results/v10/learned_kuairand_t_qp.json   -> /content/repo/results/v10/learned_kuairand_t_qp.json
#      (c_seeds_qp: the stored QP-family configurations)
#   9. optional, to continue an interrupted local run: results/v12/<name>.json.partial.json
#      -> /content/out/v12/<name>.json.partial.json (or results/v11/c_seeds/qp_s<k>.json.partial.json ->
#      /content/out/v11/c_seeds/qp_s<k>.json.partial.json). The partial files do not depend on the device.
#
# Run, in a Colab cell after the uploads and with a GPU runtime (the script stops if no CUDA device is found):
#   !bash /content/repo/experiments/colab_job.sh din_27k
# Extra arguments after the label are appended to each run (later flags override the label's own), e.g. a tiny
# smoke run before the full one (it writes to its own output path, not the label's):
#   !bash /content/repo/experiments/colab_job.sh din_27k --n_users 40 --epochs 1 --out /content/out/smoke/din.json
#
# Note: the label's runs write to OUT, and the device is cuda in every run; the CPU results of results/ are not
# reproduced on the GPU, only the same protocol (the float arithmetic differs).

REPO="${REPO:-/content/repo}"
OUT="${OUT:-/content/out}"
LABEL="${1:-}"

usage() {
  echo "usage: $0 {din_27k|sim_27k|c_seeds_qp|daxwide_lgn|daxwide_qp} [extra arguments]" >&2
  exit 2
}
case "$LABEL" in
  din_27k|sim_27k|c_seeds_qp|daxwide_lgn|daxwide_qp) ;;
  *) usage ;;
esac
cd "$REPO" || { echo "repo not found: $REPO (see the uploads in the header)" >&2; exit 1; }

python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "no CUDA device: set the Colab runtime to a GPU and rerun" >&2; exit 1; }
python3 -c "import torch; print('device:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

need() {   # need PATH...: stops with the upload to make when a file or directory is missing
  for p in "$@"; do
    [ -e "$p" ] || { echo "missing upload: $REPO/$p (see the header of $0)" >&2; exit 1; }
  done
}
done_or() {   # done_or OUT CMD...: skips CMD when OUT already exists; stops the script when CMD fails
  local out="$1"; shift
  if [ -f "$out" ]; then echo "exists, skipped: $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  "$@" || { echo "failed: $out" >&2; exit 1; }
}

case "$LABEL" in
  din_27k)
    need data/raw/seq_27k.npz data/raw/lifelong_27k.npz data/raw/KuaiRand-Pure/data
    done_or "$OUT/v12/din_27k.json" env OMP_NUM_THREADS=2 python3 experiments/run_din.py \
      --arch din --dataset kuairand27k --seq data/raw/seq_27k.npz --seeds 5 --threads 2 --device cuda \
      --out "$OUT/v12/din_27k.json" "${@:2}" ;;
  sim_27k)
    need data/raw/seq_27k.npz data/raw/lifelong_27k.npz data/raw/KuaiRand-Pure/data
    done_or "$OUT/v12/sim_27k.json" env OMP_NUM_THREADS=2 python3 experiments/run_din.py \
      --arch sim --dataset kuairand27k --seq data/raw/seq_27k.npz --seeds 5 --threads 2 --device cuda \
      --out "$OUT/v12/sim_27k.json" "${@:2}" ;;
  c_seeds_qp)
    need data/raw/lifelong_27k.npz data/raw/KuaiRand-Pure/data results/v10/learned_kuairand_t_qp.json
    for s in 1 2 3; do
      done_or "$OUT/v11/c_seeds/qp_s$s.json" env OMP_NUM_THREADS=4 python3 experiments/run_dax.py \
        --backbone qp_posthoc --qp_source stored --train_seed $s --seeds 5 --threads 4 --device cuda \
        --out "$OUT/v11/c_seeds/qp_s$s.json" "${@:2}"
    done ;;
  daxwide_lgn)
    need data/raw/lifelong_27k.npz data/raw/KuaiRand-Pure/data results/v11/dax_lgn_post.json
    done_or "$OUT/v12/daxref_lgn.json" env OMP_NUM_THREADS=2 python3 experiments/run_dax.py \
      --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 \
      --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --seeds 5 --threads 2 --device cuda \
      --out "$OUT/v12/daxref_lgn.json" "${@:2}"
    done_or "$OUT/v12/daxwide_lgn.json" env OMP_NUM_THREADS=2 python3 experiments/run_dax.py \
      --backbone leaf_lgn --mode posthoc --select_from results/v11/dax_lgn_post.json --epoch_window 3 \
      --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --prior_grid 10 20 40 80 160 --klife_grid 0 100 200 400 1000 \
      --seeds 5 --threads 2 --device cuda --out "$OUT/v12/daxwide_lgn.json" "${@:2}" ;;
  daxwide_qp)
    need data/raw/lifelong_27k.npz data/raw/KuaiRand-Pure/data results/v11/dax_qp.json
    done_or "$OUT/v12/daxref_qp.json" env OMP_NUM_THREADS=4 python3 experiments/run_dax.py \
      --backbone qp_posthoc --select_from results/v11/dax_qp.json --epoch_window 3 \
      --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --seeds 5 --threads 4 --device cuda \
      --out "$OUT/v12/daxref_qp.json" "${@:2}"
    done_or "$OUT/v12/daxwide_qp.json" env OMP_NUM_THREADS=4 python3 experiments/run_dax.py \
      --backbone qp_posthoc --select_from results/v11/dax_qp.json --epoch_window 3 \
      --gammas 0.1 0.25 0.5 0.7 1.0 --no_zcache --prior_grid 10 20 40 80 160 --klife_grid 0 100 200 400 1000 \
      --seeds 5 --threads 4 --device cuda --out "$OUT/v12/daxwide_qp.json" "${@:2}" ;;
  *)
    usage ;;
esac
