"""KuaiRand with all of its signals: impressions, 12 feedback types, scenario (tab), timestamps,
user and video features, and (for KuaiRand-1K) the users' histories outside the random item pool.

Same conventions as drup/data.py:
    O      (m, n)  1 if the pair was impressed in the standard (policy-driven) log used for training
    Y      (m, n)  binary training label on impressed pairs (0 elsewhere)
    P_given        None (KuaiRand ships no propensities)
    test   list of (user, item_array, relevance_array) over the randomly exposed impressions;
           pairs that were impressed in the training log are never ranked (as in load_kuairand)
Use drup.data.split_test(test, frac_val, seed, by="user") for validation / test splits.

With version="pure", n_users=10000, seed=0, label="is_click" and train_until=None the matrices O, Y and
the test list are identical to drup.data.load_kuairand (checked in experiments/v9_dataset_stats.py), so
results are paired with the stored KuaiRand rows of results/v3-v6.

Extra keys (all built from the standard log only, never from the random log):
    signals     {name: (m, n) float32} per exposed pair: is_click, long_view, is_like, is_follow,
                is_comment, is_forward, is_hate, is_profile_enter (max over repeats), play_ratio (mean
                play_time / duration, clipped to [0, 5]), n_imp (number of impressions)
    log         DataFrame of the training impressions: u, i, time_ms, tab, the signals, play_ratio
    test_log    DataFrame of the test impressions (random log) with the same columns, for
                impression-level metrics (GAUC) and time-aware models; its outcomes are test labels
    user_feat   (m, F) float32, numeric user features (categorical ranges one-hot encoded)
    item_feat   DataFrame indexed 0..n-1: author_id, music_id, tag (first tag), video_duration, video_type,
                upload_type
    aux         optional (extra_users=True): DataFrame u_aux, i, time_ms, signals of the standard log of
                users NOT in the sample (training data only; their random log is never loaded)
    outpool     version="1k" only: DataFrame of the standard log on videos outside the random pool
                (u, video_id, time_ms, tab, signals) plus outpool_item_feat for those videos
    meta        cutoffs, label, and the result of the leak checks
The platform statistics files (video_features_statistic_*) are month aggregates that cover the test
period; they are loaded only with include_platform_stats=True and must not be used by a method.
"""

import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

SIGNALS = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
           "is_profile_enter"]
LOG_COLS = ["user_id", "video_id", "time_ms", "tab", "play_time_ms", "duration_ms"] + SIGNALS
_DT = {c: "int8" for c in SIGNALS}
_DT.update({"user_id": "int64", "video_id": "int64", "time_ms": "int64", "tab": "int8",
            "play_time_ms": "float32", "duration_ms": "float32"})


def _files(root, version):
    if version == "pure":
        d = os.path.join(root, "KuaiRand-Pure", "data")
        s = "pure"
    elif version == "1k":
        d = os.path.join(root, "KuaiRand-1K", "data")
        s = "1k"
    else:
        raise ValueError(version)
    return d, {"std": [os.path.join(d, f"log_standard_4_08_to_4_21_{s}.csv"),
                       os.path.join(d, f"log_standard_4_22_to_5_08_{s}.csv")],
               "rnd": os.path.join(d, f"log_random_4_22_to_5_08_{s}.csv"),
               "user": os.path.join(d, f"user_features_{s}.csv"),
               "video": os.path.join(d, f"video_features_basic_{s}.csv"),
               "stat": os.path.join(d, f"video_features_statistic_{s}.csv")}


def _read_log(path):
    x = pd.read_csv(path, usecols=LOG_COLS, dtype=_DT)
    x["play_ratio"] = (x.play_time_ms / x.duration_ms.clip(lower=1.0)).clip(0.0, 5.0).astype("float32")
    return x.drop(columns=["play_time_ms", "duration_ms"])


def _user_features(path, users):
    uf = pd.read_csv(path).set_index("user_id").reindex(users)
    num = uf.select_dtypes(include=[np.number]).astype("float64")
    num = num.drop(columns=[c for c in num.columns if c.endswith("_range")], errors="ignore")
    cat = uf.select_dtypes(exclude=[np.number])
    parts = [num.fillna(num.median())]
    if len(cat.columns):
        parts.append(pd.get_dummies(cat.astype(str), dtype="float64"))
    X = pd.concat(parts, axis=1)
    # heavy-tailed counts: log scale, then standardise
    X = np.sign(X) * np.log1p(np.abs(X))
    X = (X - X.mean()) / X.std().replace(0, 1.0)
    return X.values.astype(np.float32), list(X.columns)


def _video_features(path, items, chunks=False):
    cols = ["video_id", "author_id", "music_id", "tag", "video_duration", "video_type", "upload_type"]
    if chunks:   # the 1K file has 4.4M rows; keep only the requested videos
        want = pd.Index(items)
        parts = [c[c.video_id.isin(want)] for c in pd.read_csv(path, usecols=cols, chunksize=1_000_000)]
        vf = pd.concat(parts)
    else:
        vf = pd.read_csv(path, usecols=cols)
    vf = vf.drop_duplicates("video_id").set_index("video_id").reindex(items)
    vf["tag"] = vf["tag"].astype(str).str.split(",").str[0]
    return vf.reset_index(drop=True)


def _dense(u, i, v, m, n, agg):
    df = pd.DataFrame({"u": u, "i": i, "v": v})
    g = df.groupby(["u", "i"]).v.agg(agg)
    M = torch.zeros(m, n, dtype=torch.float32)
    M[torch.as_tensor(g.index.get_level_values(0).values), torch.as_tensor(g.index.get_level_values(1).values)] = \
        torch.as_tensor(g.values.astype(np.float32))
    return M


def load_kuairand_rich(root=ROOT, version="pure", n_users=10000, seed=0, label="is_click",
                       train_until=None, extra_users=False, include_platform_stats=False, dense_dtype=torch.float64,
                       with_signals=True):
    """See the module docstring. label: training and test label (is_click or long_view or any of SIGNALS).
    train_until: optional time_ms cutoff; only standard impressions strictly before it are kept for
    training (a temporal protocol); the test is unchanged. n_users=None keeps every user in both logs."""
    d, f = _files(root, version)
    std = pd.concat([_read_log(p) for p in f["std"]], ignore_index=True)
    rnd = _read_log(f["rnd"])
    pool = np.sort(rnd.video_id.unique())
    outpool = None
    if version == "1k":
        inpool = std.video_id.isin(pool)
        outpool = std[~inpool].reset_index(drop=True)
        std = std[inpool].reset_index(drop=True)
    both = np.intersect1d(std.user_id.unique(), rnd.user_id.unique())
    rng = np.random.default_rng(seed)
    if n_users is None or n_users >= len(both):
        users = both
    else:
        users = np.sort(rng.choice(both, size=n_users, replace=False))
    if version == "pure":   # same item index as drup.data.load_kuairand
        items = np.union1d(std.video_id.unique(), rnd.video_id.unique())
    else:
        items = pool
    aux = None
    if extra_users:
        aux = std[~std.user_id.isin(users)].copy()
    std = std[std.user_id.isin(users)].copy()
    rnd = rnd[rnd.user_id.isin(users)].copy()
    if train_until is not None:
        std = std[std.time_ms < train_until]
        if aux is not None:
            aux = aux[aux.time_ms < train_until]
    m, n = len(users), len(items)
    std["u"] = np.searchsorted(users, std.user_id.values)
    std["i"] = np.searchsorted(items, std.video_id.values)
    rnd["u"] = np.searchsorted(users, rnd.user_id.values)
    rnd["i"] = np.searchsorted(items, rnd.video_id.values)

    if with_signals:
        signals = {s: _dense(std.u.values, std.i.values, std[s].values, m, n, "max") for s in SIGNALS}
        signals["play_ratio"] = _dense(std.u.values, std.i.values, std.play_ratio.values, m, n, "mean")
        signals["n_imp"] = _dense(std.u.values, std.i.values, np.ones(len(std), np.float32), m, n, "sum")
    else:   # lean mode (all 27K users): only the impression count and the label
        signals = {"n_imp": _dense(std.u.values, std.i.values, np.ones(len(std), np.float32), m, n, "sum"),
                   label: _dense(std.u.values, std.i.values, std[label].values, m, n, "max")}
    O = (signals["n_imp"] > 0).to(dense_dtype)
    Y = signals[label].to(dense_dtype) * O
    if not with_signals:
        signals = {}

    Onp = O.numpy()
    rg = rnd.groupby(["u", "i"])[label].max().reset_index()
    test = []
    for u, grp in rg.groupby("u"):
        it = grp.i.values
        keep = Onp[u, it] == 0
        test.append((int(u), it[keep], grp[label].values[keep].astype(np.float64)))

    user_feat, user_cols = _user_features(f["user"], users)
    item_feat = _video_features(f["video"], items)
    out = {"name": f"kuairand_{version}", "O": O, "Y": Y, "P_given": None, "test": test,
           "signals": signals, "log": std.drop(columns=["user_id", "video_id"]).reset_index(drop=True),
           "test_log": rnd.drop(columns=["user_id", "video_id"]).reset_index(drop=True),
           "user_feat": user_feat, "user_feat_cols": user_cols, "item_feat": item_feat,
           "users_raw": users, "items_raw": items}
    if aux is not None:
        aux_users = np.sort(aux.user_id.unique())
        aux["u_aux"] = np.searchsorted(aux_users, aux.user_id.values)
        aux["i"] = np.searchsorted(items, aux.video_id.values)
        out["aux"] = aux.drop(columns=["user_id", "video_id"]).reset_index(drop=True)
    if outpool is not None:
        outpool = outpool[outpool.user_id.isin(users)].copy()
        if train_until is not None:
            outpool = outpool[outpool.time_ms < train_until]
        outpool["u"] = np.searchsorted(users, outpool.user_id.values)
        out["outpool"] = outpool.drop(columns=["user_id"]).reset_index(drop=True)
        out["outpool_item_feat_path"] = f["video"]
    if include_platform_stats:
        out["platform_stats_LEAKS_TEST_PERIOD"] = pd.read_csv(f["stat"]).set_index("video_id").reindex(items)
    out["meta"] = {"version": version, "label": label, "train_until": train_until, "n_users": m, "n_items": n,
                   "leak": leak_checks(out)}
    return out


def leak_checks(d):
    """Checks that no test outcome can reach training. Returns a dict of counts (all must be 0 / True)."""
    O = d["O"].numpy()
    tl = d["test_log"]
    lg = d["log"]
    n_ranked_logged = int(sum(int(O[u, it].sum()) for u, it, _ in d["test"]))
    key = lambda df: set(zip(df.u.values.tolist(), df.i.values.tolist(), df.time_ms.values.tolist()))
    same_event = len(key(lg) & key(tl))
    pair_overlap = len(set(zip(lg.u.values.tolist(), lg.i.values.tolist())) &
                       set(zip(tl.u.values.tolist(), tl.i.values.tolist())))
    res = {"ranked_pairs_present_in_training_log": n_ranked_logged,
           "test_events_present_in_training_log": same_event,
           "test_pairs_also_impressed_in_training_log_(excluded_from_ranking)": pair_overlap,
           "training_uses_random_log": False,
           "max_train_time_ms": int(lg.time_ms.max()) if len(lg) else None,
           "min_test_time_ms": int(tl.time_ms.min()) if len(tl) else None}
    return res


def load_kuairand_pure_rich(**kw):
    return load_kuairand_rich(version="pure", **kw)


def load_kuairand_1k_rich(**kw):
    kw.setdefault("n_users", None)
    return load_kuairand_rich(version="1k", **kw)
