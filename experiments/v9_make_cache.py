"""Cache a KuaiRand setting from drup/data2.py in the drup/data.py dict format (O, Y, P_given, test) so that
experiments/run_learned.py can load it with torch.load, like data/raw/kuairand.pt.
python3 experiments/v9_make_cache.py --train_until 1650556800000 --n_users 0 --out data/raw/kuairand_t.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drup.data2 import load_kuairand_rich  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_until", type=int, default=None)
    ap.add_argument("--n_users", type=int, default=0, help="0 = all users")
    ap.add_argument("--label", default="is_click")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load_kuairand_rich(version="pure", n_users=(a.n_users or None), label=a.label, train_until=a.train_until,
                           dense_dtype=torch.float32, with_signals=False)
    out = {"name": f"kuairand_t{a.train_until}", "O": d["O"], "Y": d["Y"], "P_given": None, "test": d["test"],
           "users_raw": d["users_raw"], "meta": d["meta"]}
    torch.save(out, a.out)
    print(f"saved {a.out}: O {tuple(d['O'].shape)}, test users {len(d['test'])}, leak {d['meta']['leak']}")


if __name__ == "__main__":
    main()
