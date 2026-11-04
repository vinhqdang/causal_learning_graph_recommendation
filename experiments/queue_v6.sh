cd "$(dirname "$0")/.."
C=results/v5/cache
case "$1" in
  coat_ials)   # DRGP-iALS, iALS + additive prior, matched iALS grid
    for pr in drgp add; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --floor 0.1 --prior $pr --gf_buckets 2 --prior_cache $C --methods iALS-p --out results/v5/learned_coat_${pr}_ials.json || exit 1
    done
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 30 --seed_check 0 --methods iALS-wide --out results/v5/learned_coat_ialswide.json ;;
  coat_prior)  # prior alone for every floor (cheap)
    for f in 0.05 0.1 0.2; do for b in 1 2; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --seed_check 0 --floor $f --prior drgp --gf_buckets $b --prior_cache $C --methods DRGP --out results/v5/learned_coat_drgp_f${f}_b${b}.json || exit 1
    done; done ;;
  coat_lgn)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --floor 0.1 --prior drgp --gf_buckets 2 --prior_cache $C --methods DRGP-LGN-bpr DRGP-LGN-dr DRGP-LGN-hyb --out results/v5/learned_coat_drgp_lgn.json ;;
  kuairec_prior)  # builds and caches the prior (about 15 minutes per floor), then scores it
    for f in 0.1 0.2; do
      OMP_NUM_THREADS=4 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 4 --seed_check 0 --floor $f --prior drgp --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods DRGP --out results/v5/learned_kuairec_drgp_f${f}.json || exit 1
    done ;;
  kuairec_ials)
    for pr in drgp add; do
      OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 2 --dims 64 --epochs 15 --patience 3 --seed_check 0 --floor 0.2 --prior $pr --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods iALS-p --out results/v5/learned_kuairec_${pr}_ials.json || exit 1
    done ;;
  kuairec_lgn)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 1 --epochs 30 --patience 3 --batch 32768 --pairs_per_epoch 2000000 --lrs 3e-3 1e-3 --wds 1e-5 --seed_check 0 --floor 0.2 --prior drgp --gf_buckets 4 --gf_alphas 0.5 --prior_cache $C --methods DRGP-LGN-dr DRGP-LGN-bpr --out results/v5/learned_kuairec_drgp_lgn.json ;;
  yahoo_all|kuairand_all)
    ds=${1%_all}; fl=$([ $ds = yahoo ] && echo 0.1 || echo 0.02)
    for m in DRGP iALS-p DRGP-LGN-bpr DRGP-LGN-hyb; do
      OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset $ds --prop pop --seeds 5 --threads 1 --epochs 60 --patience 5 --batch $([ $ds = yahoo ] && echo 1024 || echo 8192) --lrs 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --floor $fl --prior drgp --gf_buckets 4 --prior_cache $C --methods $m --out results/v5/learned_${ds}_drgp_${m}.json || exit 1
    done ;;
  *) echo "usage: $0 {coat_ials|coat_prior|coat_lgn|kuairec_prior|kuairec_ials|kuairec_lgn|yahoo_all|kuairand_all}" >&2
     echo "unknown label: $1" >&2; exit 2 ;;
esac
