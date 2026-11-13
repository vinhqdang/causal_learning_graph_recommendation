"""Data statistics of the temporal protocol (Table tab:data, Section 3 median impressions per pool item).

python3 experiments/v13_data_stats.py      (about 2 GB peak memory; reads three columns of the CSV logs in chunks)

Inputs: the standard and random CSV logs of KuaiRand-Pure (27K) and KuaiRand-1K, the cached test lists and
matrices data/raw/kuairand_t.pt and data/raw/kuairand_1k_t.pt (memory-mapped, used for shapes, test lists and
distinct pairs only), and the out-of-pool tables data/raw/lifelong_27k.npz and data/raw/lifelong_1k.npz.
Output: results/v13_data_stats.txt
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from drup.data2 import _files  # noqa: E402
from drup.leaf import protocol_split  # noqa: E402

CUT = 1650556800000                       # 2022-04-22 00:00 UTC+8, the cut-off T0
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results", "v13_data_stats.txt")
CHUNK = 2_000_000


def scan_standard(paths, users, pool):
    """Pre-cut standard impressions of the dataset users. Returns counts per pool video, the number of
    impressions on pool and on non-pool videos, and the distinct (user, video) pairs on pool videos."""
    counts = np.zeros(len(pool), np.int64)
    n_in = n_out = 0
    keys = []
    for p in paths:
        for ch in pd.read_csv(p, usecols=["user_id", "video_id", "time_ms"],
                                                dtype="int64", chunksize=CHUNK):
            ch = ch[(ch.time_ms.values < CUT) & np.isin(ch.user_id.values, users)]
            v = ch.video_id.values
            pos = np.searchsorted(pool, v)
            pos = np.minimum(pos, len(pool) - 1)
            inp = pool[pos] == v
            n_in += int(inp.sum())
            n_out += int((~inp).sum())
            counts += np.bincount(pos[inp], minlength=len(pool))
            u = np.searchsorted(users, ch.user_id.values[inp])
            keys.append(np.unique(u.astype(np.int64) * len(pool) + pos[inp]))
    pairs = int(np.unique(np.concatenate(keys)).size) if keys else 0
    return counts, n_in, n_out, pairs


def scan_random_pool(paths):
    vids = []
    for p in paths:
        vids.append(np.unique(pd.read_csv(p, usecols=["video_id"], dtype="int64").video_id.values))
    return np.unique(np.concatenate(vids))


def cache_info(path):
    C = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    O = C["O"]
    test = C["test"]
    nc = np.array([len(it) for _, it, _ in test])
    evaluated = {}
    pooled = set()
    for s in range(5):
        val, tst = protocol_split(test, s, 0.3)
        ev = {u for u, _, r in tst if np.asarray(r).sum() > 0}
        evaluated[s] = (len(val), len(tst), len(ev))
        pooled |= ev
    users_with_pos = {u for u, _, r in test if np.asarray(r).sum() > 0}
    return {"users": np.asarray(C["users_raw"]), "O_shape": tuple(O.shape), "O_pairs": int(torch.count_nonzero(O)),
            "test_users": len(test), "cand_mean": float(nc.mean()), "cand_median": float(np.median(nc)),
            "cand_min": int(nc.min()), "cand_max": int(nc.max()), "evaluated": evaluated,
            "users_with_pos": len(users_with_pos), "pooled_eval": len(pooled)}


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    res = {}
    for name, version, npz in (("27K", "pure", "lifelong_27k.npz"), ("1K", "1k", "lifelong_1k.npz")):
        cache = cache_info(os.path.join(RAW, "kuairand_t.pt" if version == "pure" else "kuairand_1k_t.pt"))
        d, f = _files(os.path.join(ROOT, "data", "raw"), version)
        pool = scan_random_pool([f["rnd"]])
        counts, n_in, n_out, pairs = scan_standard(f["std"], cache["users"], pool)
        L = np.load(os.path.join(RAW, npz), allow_pickle=True)
        npz_out = int(round(float(L["tot_pre"][:, 0].astype(np.float64).sum())))
        npz_users = int(L["tot_pre_user"].shape[0])
        res[name] = dict(cache=cache, pool=len(pool), counts=counts, n_in=n_in, n_out=n_out, pairs=pairs,
                         npz_out=npz_out, npz_users=npz_users)
        r = res[name]
        log(f"== KuaiRand-{name} ({version})")
        log(f"dataset users {len(cache['users'])}; random-log pool videos {len(pool)}; cached O shape {cache['O_shape']}")
        log(f"pre-cut standard impressions of dataset users, pool videos (CSV): {n_in}")
        log(f"pre-cut standard impressions of dataset users, non-pool videos (CSV): {n_out}")
        log(f"distinct pre-cut pairs (CSV) {pairs}; nonzero entries of cached O {cache['O_pairs']}; "
            f"match: {pairs == cache['O_pairs']}")
        log(f"pre-cut out-of-pool impressions (lifelong npz, {npz_users} users with history): {npz_out}")
        log(f"median pre-cut impressions per pool item (over {len(pool)} pool videos, zeros included): "
            f"{np.median(counts):.1f}; mean {counts.mean():.1f}; share with zero {np.mean(counts == 0):.4f}")
        log(f"test users (candidate lists): {cache['test_users']}; candidates per user mean {cache['cand_mean']:.1f}, "
            f"median {cache['cand_median']:.0f}, min {cache['cand_min']}, max {cache['cand_max']}")
        for s, (nv, nt, ne) in cache["evaluated"].items():
            log(f"split {s} (seed {s}, frac_val 0.3): validation users {nv}; test users {nt}; "
                f"test users with a click {ne}")
        log(f"test users with a click in at least one split, pooled: {cache['pooled_eval']}; "
            f"users with a click in any candidate list: {cache['users_with_pos']}")

        log("")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
