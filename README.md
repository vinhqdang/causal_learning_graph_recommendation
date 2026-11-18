# Examination before attraction: lifelong affinities and detection-adjusted item appeal under randomised exposure

Code and result files for the manuscript in `paper/v8/` (LaTeX source and PDF).

The method has two parts:

- **LEAF**: lifelong, exposure-conditional click statistics on tags, authors and music, fed to a LightGCN recommender (`drup/leaf.py`).
- **DAX**: a click factorised into user examination, scenario examination and item attractiveness, fitted by approximate expectation-maximisation and added post hoc to any backbone (`drup/dax.py`).

Evaluation is on KuaiRand-27K (replication on KuaiRand-1K) with a temporal protocol: training uses only data before a cut-off; validation and test use only the randomised-exposure log after it.

## Layout

| path | content |
|---|---|
| `drup/` | models, data loading, metrics, DAX and LEAF |
| `experiments/` | `run_dax.py`, `run_leaf.py`, `run_din.py`, `v9_probes/run_learned_t.py` (training and evaluation); `queue_v10.sh` to `queue_v15.sh` (the job queues behind each table); `final_v12.py` (all statistical comparisons); `v13_tables.py`, `v13_figs.py` (tables and figures of the paper) |
| `results/` | result files (`v10` to `v17`) and the printed analysis `v14/final14.txt` |
| `tests/` | unit tests (`python3 -m pytest -q tests`) |
| `paper/v8/` | manuscript sources; `refs.bib` in `paper/` |

## Reproducing

1. Download KuaiRand (Pure and 1K) into `data/raw/` and build the caches with `experiments/v9_make_cache.py`, `v9_build_lifelong.py` and `v12_build_seq.py`.
2. Run the queues in `experiments/` (for example `bash experiments/queue_v14.sh all`). Runs resume from `*.partial.json` files.
3. `python3 experiments/final_v12.py` prints every comparison; `python3 experiments/v13_tables.py` and `v13_figs.py` regenerate the paper tables and figures.

Runs use CPU (four cores, 15 GB memory suffice); `--device cuda` is supported by the main scripts.

## Licence

See `LICENSE`.
