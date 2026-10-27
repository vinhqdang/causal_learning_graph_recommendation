cd "$(dirname "$0")/.."
case "$1" in
  ens_coat)
    for st in drup trained blend; do OMP_NUM_THREADS=1 python3 experiments/run_ensemble.py --dataset coat --stage $st || exit 1; done ;;
  ens_kuairec_drup)
    OMP_NUM_THREADS=2 python3 experiments/run_ensemble.py --dataset kuairec --stage drup --threads 2 ;;
  ens_kuairec_rest)
    for st in trained blend; do OMP_NUM_THREADS=1 python3 experiments/run_ensemble.py --dataset kuairec --stage $st || exit 1; done ;;
  ens_kuairand|ens_yahoo)
    ds=${1#ens_}
    for st in drup trained blend; do OMP_NUM_THREADS=1 python3 experiments/run_ensemble.py --dataset $ds --stage $st || exit 1; done ;;
  seeds_*)
    OMP_NUM_THREADS=1 python3 experiments/run_seeds.py --dataset ${1#seeds_} --seeds 1 2 3 ;;
  adv_coat)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --wide_layers 1 2 3 --wide_dims 32 64 --methods AdvInfoNCE-LGN --wide AdvInfoNCE-LGN --adv_taus 0.2 0.75 --adv_etas 4 12 --adv_lr 1e-2 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_coat_adv.json ;;
  advdr_coat)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset coat --prop given --seeds 10 --threads 1 --dims 32 64 --epochs 100 --batch 256 --seed_check 0 --wide_layers 1 2 3 --wide_dims 32 64 --methods DR-AdvInfoNCE-LGN-p --adv_taus 0.75 --adv_etas 4 12 --adv_lr 1e-2 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_coat_advdr.json ;;
  adv_yahoo)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset yahoo --prop pop --seeds 5 --threads 1 --epochs 60 --patience 5 --batch 1024 --lrs 3e-3 1e-3 --wds 1e-5 1e-4 --dims 128 --layers 2 --seed_check 0 --methods AdvInfoNCE-LGN --adv_taus 0.1 0.28 --adv_etas 4 12 --adv_lr 1e-3 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_yahoo_adv.json ;;
  adv_kuairand)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairand --prop pop --seeds 5 --threads 1 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-4 --seed_check 0 --methods AdvInfoNCE-LGN --adv_taus 0.1 0.28 --adv_etas 4 12 --adv_lr 1e-3 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_kuairand_adv.json ;;
  adv_kuairec)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 1 --epochs 30 --patience 3 --batch 4096 --pairs_per_epoch 2000000 --lrs 1e-2 3e-3 --wds 1e-5 --seed_check 0 --methods AdvInfoNCE-LGN --adv_taus 0.2 2.0 --adv_etas 4 12 --adv_lr 5e-4 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_kuairec_adv.json ;;
  advdr_kuairec)
    OMP_NUM_THREADS=1 python3 experiments/run_learned.py --dataset kuairec --prop pop --seeds 5 --threads 1 --epochs 30 --patience 3 --batch 4096 --pairs_per_epoch 2000000 --lrs 1e-2 3e-3 --wds 1e-5 --seed_check 0 --methods DR-AdvInfoNCE-LGN-p --adv_taus 2.0 --adv_etas 4 12 --adv_lr 5e-4 --adv_interval 1 --adv_min_steps 100 --n_neg 64 --out results/v4/learned_kuairec_advdr.json ;;
  *)
    echo "usage: $0 {ens_coat|ens_kuairec_drup|ens_kuairec_rest|ens_kuairand|ens_yahoo|seeds_<dataset>|adv_coat|advdr_coat|adv_yahoo|adv_kuairand|adv_kuairec|advdr_kuairec}" >&2
    echo "unknown label: $1" >&2
    exit 2 ;;
esac
