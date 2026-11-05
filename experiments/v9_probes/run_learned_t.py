"""Run experiments/run_learned.py unchanged on a cached drup/data2.py setting: every torch.load of
"data/raw/kuairand.pt" is redirected to the file named by the environment variable V9_DATA
(e.g. data/raw/kuairand_t.pt from experiments/v9_make_cache.py). Same arguments as run_learned.py with
--dataset kuairand.
V9_DATA=data/raw/kuairand_t.pt python3 experiments/v9_probes/run_learned_t.py --dataset kuairand ...
"""
import os
import runpy
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.environ["V9_DATA"]
_load = torch.load


def _patched(path, *a, **k):
    if isinstance(path, str) and os.path.normpath(path) == os.path.normpath("data/raw/kuairand.pt"):
        path = TARGET
    return _load(path, *a, **k)


torch.load = _patched
sys.argv[0] = os.path.join(HERE, "..", "run_learned.py")
sys.path.insert(0, os.path.join(HERE, ".."))
runpy.run_path(sys.argv[0], run_name="__main__")
