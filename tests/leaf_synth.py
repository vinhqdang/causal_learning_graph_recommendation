"""Small synthetic KuaiRand-like world for the LEAF tests. Every user has a favourite tag. Out-of-pool history
before the cut-off is informative about it (clicks on the favourite tag are more frequent); the in-pool training
log is not (clicks independent of the tag); the random-exposure test candidates are labelled by the favourite tag.
So a lifelong tag channel carries all the signal, and a shuffle of the users' tables must remove it."""

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drup.leaf import KEYS, CUT_MS, key_strings  # noqa: E402


def world(seed=0, m=300, n=120, n_out=60, n_tags=6, n_authors=20, n_music=30, hist=200, log_per=40, test_per=30):
    rng = np.random.default_rng(seed)
    fav = rng.integers(0, n_tags, m)
    feat = pd.DataFrame({"tag": [f"{i % n_tags}" for i in range(n)],
                         "author_id": np.arange(n) % n_authors, "music_id": np.arange(n) % n_music})
    out_tag = rng.integers(0, n_tags, n_out)
    out_author = rng.integers(0, n_authors, n_out)
    out_music = rng.integers(0, n_music, n_out)
    users_raw = 1000 + np.arange(m)
    # lifelong events: user, out-of-pool video, time (before the cut-off), click
    ev_u, ev_v, ev_t, ev_c = [], [], [], []
    for u in range(m):
        v = rng.integers(0, n_out, hist)
        p = np.where(out_tag[v] == fav[u], 0.6, 0.1)
        ev_u.append(np.full(hist, users_raw[u])), ev_v.append(v)
        ev_t.append(rng.integers(CUT_MS - 10_000_000, CUT_MS - 1000, hist))
        ev_c.append((rng.random(hist) < p).astype(np.float32))
    ev_u, ev_v, ev_t, ev_c = map(np.concatenate, (ev_u, ev_v, ev_t, ev_c))
    npz = _aggregate(ev_u, ev_v, ev_t, ev_c, out_tag, out_author, out_music, feat)
    # training log (in-pool, uninformative clicks)
    O = torch.zeros(m, n)
    Y = torch.zeros(m, n)
    for u in range(m):
        it = rng.choice(n, log_per, replace=False)
        O[u, it] = 1.0
        Y[u, it] = torch.as_tensor((rng.random(log_per) < 0.25).astype(np.float32))
    # random-exposure candidates (never impressed in training), relevance from the favourite tag
    test = []
    tag_of = np.array([int(t) for t in feat.tag])
    for u in range(m):
        free = np.flatnonzero(O[u].numpy() == 0)
        it = rng.choice(free, test_per, replace=False)
        p = np.where(tag_of[it] == fav[u], 0.6, 0.1)
        rel = (rng.random(test_per) < p).astype(np.float64)
        if rel.sum() == 0:
            rel[0] = 1.0
        test.append((u, it, rel))
    return {"O": O, "Y": Y, "test": test, "npz": npz, "users_raw": users_raw, "item_feat": feat, "fav": fav}


def _aggregate(u, v, t, c, out_tag, out_author, out_music, feat):
    """Tables in the layout of experiments/v9_build_lifelong.py, window "pre" (events before the cut-off)."""
    pre = t < CUT_MS
    u, v, c = u[pre], v[pre], c[pre]
    npz = {}
    tot = pd.DataFrame({"u": u, "c": c}).groupby("u").agg(n=("c", "size"), c=("c", "sum"))
    npz["tot_pre_user"] = tot.index.values.astype(np.int64)
    npz["tot_pre"] = np.stack([tot.n.values, tot.c.values, np.zeros(len(tot))], 1).astype(np.float32)
    for k, arr in zip(KEYS, (out_tag, out_author, out_music)):
        pool_vals = pd.Index(key_strings(feat[k], k).astype(str).unique())
        npz[f"{k}_values"] = pool_vals.values.astype(str)
        key_of_video = pd.Series(pool_vals.get_indexer(pd.Series(arr).astype(str).values))
        kk = key_of_video.values[v]
        ok = kk >= 0
        df = pd.DataFrame({"u": u[ok], "k": kk[ok], "c": c[ok]})
        g = df.groupby(["u", "k"]).agg(n=("c", "size"), c=("c", "sum")).reset_index()
        npz[f"{k}_pre_u"] = g.u.values.astype(np.int64)
        npz[f"{k}_pre_k"] = g.k.values.astype(np.int32)
        npz[f"{k}_pre_ncl"] = np.stack([g.n.values, g.c.values, np.zeros(len(g))], 1).astype(np.float32)
    return npz


def ndcg10(S, test, row_of=None):
    """Mean nDCG@10 over users with a positive (drup.metrics.evaluate)."""
    from drup.metrics import evaluate
    return evaluate(S, test, (10,), row_of)["ndcg@10"]
