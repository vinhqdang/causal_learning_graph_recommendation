"""Wall-clock cost of the corrected K-hop operator against the plug-in one.

Random dense edge estimates of a given shape; scores are computed for a
subset of rows, as in the experiments. Reports the median of a few runs.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drup.khop import khop  # noqa: E402
from drup.propagation import three_hop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["290x300x290", "7176x10728x1411"],
                    help="users x items x scored rows")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/v2/bench_khop.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    g = torch.Generator().manual_seed(0)
    out = {}
    for sh in a.shapes:
        m, n, r = map(int, sh.split("x"))
        W = torch.rand(m, n, generator=g, dtype=torch.float32)
        C = torch.rand(m, n, generator=g, dtype=torch.float32) / (m * n) ** 0.25
        rows = torch.arange(r)
        res = {}
        for name, fn in (("K3_plugin", lambda: three_hop(W, C, rows=rows, correct=False)),
                         ("K3_corrected", lambda: three_hop(W, C, rows=rows, correct=True)),
                         ("K5_plugin", lambda: khop(W, C, 5, rows=rows, correct=False)),
                         ("K5_corrected", lambda: khop(W, C, 5, rows=rows, correct=True))):
            ts = []
            for _ in range(a.reps):
                t0 = time.time()
                fn()
                ts.append(time.time() - t0)
            res[name] = sorted(ts)[len(ts) // 2]
            print(sh, name, f"{res[name]:.2f}s", flush=True)
        out[sh] = res
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "seconds": out}, f, indent=1)


if __name__ == "__main__":
    main()
