cd "$(dirname "$0")/.."
mkdir -p results/v6
# copies the partial file of an earlier run next to the new output, so that the identical command resumes it
resume() { [ -f "$2.$3.partial.json" ] || cp "$1.$3.partial.json" "$2.$3.partial.json"; }   # args: earlier out, new out, method
case "$1" in
  x1_coat_add_noips|x1_coat_pop|x1_coat_shuf_item)   # prior anatomy on Coat, grids of the stored add runs
    pr=${1#x1_coat_}
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.1 --prior $pr --methods iALS-p --out results/v6/learned_coat_${pr}_ials.json || exit 1
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --floor 0.05 --prior $pr --wide LightGCN-p --wide_layers 1 2 3 --wide_dims 32 64 --methods LightGCN-p --out results/v6/learned_coat_${pr}_lgn.json ;;
  x1_kuairec_add|x1_kuairec_add_noips|x1_kuairec_pop|x1_kuairec_shuf_item)   # prior anatomy on KuaiRec, grid of iALS-p-small
    pr=${1#x1_kuairec_}
    OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 2 --dims 64 --epochs 15 --patience 3 --seed_check 0 --floor 0.2 --prior $pr --methods iALS-p-small --out results/v6/learned_kuairec_${pr}_ials_small.json ;;
  x1_alone_coat|x1_alone_yahoo|x1_alone_kuairand|x1_alone_kuairec)   # prior alone rows for every variant
    ds=${1#x1_alone_}
    case $ds in
      coat)     args="--prop given --seeds 10 --floor 0.1" ;;
      yahoo)    args="--prop pop --seeds 5 --floor 0.1" ;;
      kuairand) args="--prop pop --seeds 5 --floor 0.02" ;;
      kuairec)  args="--prop pop --seeds 5 --floor 0.2" ;;
    esac
    for v in add add_noips pop shuf_item user_only; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset $ds $args --threads 1 --seed_check 0 --prior $v --methods Prior-alone --out results/v6/learned_${ds}_alone_${v}.json || exit 1
    done ;;
  x2_coat)   # Coat with estimated propensities (--prop pop), stored grids
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop pop --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.1 --prior add --methods iALS-p --out results/v6/learned_coat_pop_add_ials.json || exit 1
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop pop --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --floor 0.05 --wide LightGCN-p --wide_layers 1 2 3 --wide_dims 32 64 --methods LightGCN-p --out results/v6/learned_coat_pop_lgn_p.json || exit 1
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop pop --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.05 --methods iALS-wide --out results/v6/learned_coat_pop_ialswide.json || exit 1
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop pop --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --floor 0.05 --methods LightGCN --out results/v6/learned_coat_pop_lgn.json || exit 1
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop pop --seeds 10 --threads 1 --seed_check 0 --floor 0.1 --prior add --methods Prior-alone --out results/v6/learned_coat_pop_alone_add.json ;;
  x3_yahoo)   # finish offs_yahoo (queue_v5) from its partial file
    resume results/v4/learned_yahoo_offonly.json results/v6/learned_yahoo_offonly.json LightGCN-p
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset yahoo --prop pop --seeds 5 --threads 1 --epochs 60 --patience 5 --batch 1024 --lrs 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods LightGCN-p --wide LightGCN-p --wide_layers 1 2 3 --wide_dims 64 128 --out results/v6/learned_yahoo_offonly.json ;;
  x3_kuairand)   # finish offs_kuairand (queue_v5) from its partial file
    resume results/v4/learned_kuairand_offonly.json results/v6/learned_kuairand_offonly.json LightGCN-p
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairand --prop pop --seeds 5 --threads 1 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 1e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods LightGCN-p --out results/v6/learned_kuairand_offonly.json ;;
  x4_coat)   # equal-budget iALS grid on Coat
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.05 --methods iALS-wide2 --out results/v6/learned_coat_ialswide2.json ;;
  *)
    echo "usage: $0 {x1_coat_add_noips|x1_coat_pop|x1_coat_shuf_item|x1_kuairec_add|x1_kuairec_add_noips|x1_kuairec_pop|x1_kuairec_shuf_item|x1_alone_coat|x1_alone_yahoo|x1_alone_kuairand|x1_alone_kuairec|x2_coat|x3_yahoo|x3_kuairand|x4_coat}" >&2
    echo "unknown label: $1" >&2
    exit 2 ;;
esac
