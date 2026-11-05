"""Build lifelong (out-of-pool) exposure tables of KuaiRand users from the KuaiRand-27K (or -1K) archive
without extracting the 40+ GB of standard logs: every standard-log member of the tar.gz is streamed and
aggregated chunk by chunk.

Output (data/raw/lifelong_<version>.npz): for each content key family (tag, author, music) and for two
time windows ("pre": impressions strictly before 2022-04-22 00:00 UTC+8, the first day of the random
exposure period; "all": every standard impression), the user x key table of impressions, clicks and long
views restricted to videos OUTSIDE the random pool, plus per-user totals. Only key values that occur on
pool videos are kept (they are the only ones that can transfer to the test items). The random log is
never read by the aggregation.

python3 experiments/v9_build_lifelong.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz --version 27k
"""
import argparse
import io
import os
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd

CUT = 1650556800000          # 2022-04-22 00:00 UTC+8
KEYS = ("tag", "author_id", "music_id")


def keystr(col, k):
    if k == "tag":
        return col.astype(str).str.split(",").str[0]
    return pd.to_numeric(col, errors="coerce").astype("Int64").astype(str)


def members(archive):
    with tarfile.open(archive, "r:gz") as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def stream(archive, member):
    p = subprocess.Popen(["tar", "xzOf", archive, member], stdout=subprocess.PIPE, bufsize=1 << 24)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--version", default="27k")
    ap.add_argument("--pool_dir", default="data/raw/KuaiRand-Pure/data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunksize", type=int, default=4_000_000)
    ap.add_argument("--members", nargs="*", default=None, help="member names (skips the slow listing)")
    a = ap.parse_args()
    out = a.out or f"data/raw/lifelong_{a.version}.npz"
    t0 = time.time()
    pool_feat = pd.read_csv(os.path.join(a.pool_dir, "video_features_basic_pure.csv"))
    pool_ids = np.sort(pd.read_csv(os.path.join(a.pool_dir, "log_random_4_22_to_5_08_pure.csv"),
                                   usecols=["video_id"]).video_id.unique())
    for k in KEYS:
        pool_feat[k] = keystr(pool_feat[k], k)
    pool_vals = {k: pd.Index(pool_feat[k][pool_feat[k] != "<NA>"].unique()) for k in KEYS}
    names = a.members or members(a.archive)
    feat_m = [x for x in names if "video_features_basic" in x][0]
    std_m = sorted(x for x in names if "log_standard" in x)
    print("members:", feat_m, std_m, flush=True)
    # 1. video -> key code (only key values present in the pool; -1 otherwise), streamed
    maxid = 0
    vid, codes = [], {k: [] for k in KEYS}
    p = stream(a.archive, feat_m)
    for ch in pd.read_csv(p.stdout, usecols=["video_id", "author_id", "music_id", "tag"], chunksize=a.chunksize,
                          dtype={"tag": str}):
        vid.append(ch.video_id.values)
        for k in KEYS:
            codes[k].append(pool_vals[k].get_indexer(keystr(ch[k], k)).astype(np.int32))
    p.wait()
    vid = np.concatenate(vid)
    maxid = int(vid.max()) + 1
    vmap = {}
    for k in KEYS:
        arr = np.full(maxid, -1, np.int32)
        arr[vid] = np.concatenate(codes[k])
        vmap[k] = arr
    inpool = np.zeros(maxid, bool)
    inpool[pool_ids[pool_ids < maxid]] = True
    print(f"video features: {len(vid)} videos, {time.time()-t0:.0f}s", flush=True)
    # 2. stream the standard logs and aggregate (user, key value) x window
    acc = {(k, w): {} for k in KEYS for w in ("pre", "all")}
    tot = {}
    cols = ["user_id", "video_id", "time_ms", "is_click", "long_view"]
    nrows = 0
    for mname in std_m:
        p = stream(a.archive, mname)
        for ch in pd.read_csv(p.stdout, usecols=cols, chunksize=a.chunksize):
            nrows += len(ch)
            v = ch.video_id.values
            ok = (v < maxid)
            ch = ch[ok & ~inpool[np.minimum(v, maxid - 1)]]
            pre = ch.time_ms.values < CUT
            for w, sel in (("pre", pre), ("all", np.ones(len(ch), bool))):
                c = ch[sel]
                g = c.groupby("user_id").agg(n=("is_click", "size"), c=("is_click", "sum"), l=("long_view", "sum"))
                t = tot.setdefault(w, None)
                tot[w] = g if t is None else t.add(g, fill_value=0)
                for k in KEYS:
                    code = vmap[k][c.video_id.values]
                    m = code >= 0
                    if not m.any():
                        continue
                    df = pd.DataFrame({"u": c.user_id.values[m], "k": code[m], "c": c.is_click.values[m],
                                       "l": c.long_view.values[m]})
                    gg = df.groupby(["u", "k"]).agg(n=("c", "size"), c=("c", "sum"), l=("l", "sum"))
                    prev = acc[(k, w)].get("df")
                    acc[(k, w)]["df"] = gg if prev is None else prev.add(gg, fill_value=0)
            print(f"  {mname}: {nrows} rows, {time.time()-t0:.0f}s", flush=True)
        p.wait()
    res = {"cut_ms": CUT, "n_rows": nrows}
    for w in ("pre", "all"):
        t = tot[w]
        res[f"tot_{w}_user"] = t.index.values
        res[f"tot_{w}"] = t[["n", "c", "l"]].values.astype(np.float32)
        for k in KEYS:
            df = acc[(k, w)]["df"].reset_index()
            res[f"{k}_{w}_u"] = df.u.values.astype(np.int64)
            res[f"{k}_{w}_k"] = df.k.values.astype(np.int32)
            res[f"{k}_{w}_ncl"] = df[["n", "c", "l"]].values.astype(np.float32)
    for k in KEYS:
        res[f"{k}_values"] = pool_vals[k].values.astype(str)
    np.savez_compressed(out, **res)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
