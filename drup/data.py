"""Datasets with an MNAR training log and an unbiased (MAR / fully observed) test.

Each loader returns a dict with dense float tensors
    O      (m, n)  1 if the pair was exposed in the training log
    Y      (m, n)  binary outcome on exposed pairs (0 elsewhere)
    P_given        optional (m, n) propensities shipped with the dataset
    test   list of (user, item_array, relevance_array) over unbiased candidates
"""

import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_coat(root=ROOT, threshold=4):
    d = os.path.join(root, "coat")
    tr = np.loadtxt(os.path.join(d, "train.ascii"))
    te = np.loadtxt(os.path.join(d, "test.ascii"))
    pr = np.loadtxt(os.path.join(d, "propensities.ascii"))
    O = (tr > 0).astype(np.float64)
    Y = (tr >= threshold).astype(np.float64)
    test = []
    for u in range(te.shape[0]):
        # 7.9% of the random test ratings hit items the user also rated in the
        # MNAR log; as for KuaiRec, only never-logged items are ranked.
        items = np.nonzero((te[u] > 0) & (tr[u] == 0))[0]
        test.append((u, items, (te[u, items] >= threshold).astype(np.float64)))
    return {
        "name": "coat",
        "O": torch.tensor(O),
        "Y": torch.tensor(Y),
        "P_given": torch.tensor(pr),
        "test": test,
    }


def load_kuairec(root=ROOT, threshold=2.0):
    d = os.path.join(root, "KuaiRec 2.0", "data")
    if not os.path.isdir(d):
        d = os.path.join(root, "KuaiRec", "data")
    cols = ["user_id", "video_id", "watch_ratio"]
    big = pd.read_csv(os.path.join(d, "big_matrix.csv"), usecols=cols)
    small = pd.read_csv(os.path.join(d, "small_matrix.csv"), usecols=cols)
    users = np.union1d(big.user_id.unique(), small.user_id.unique())
    items = np.union1d(big.video_id.unique(), small.video_id.unique())
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {i: k for k, i in enumerate(items)}
    m, n = len(users), len(items)
    bu = big.user_id.map(uidx).values
    bi = big.video_id.map(iidx).values
    # A pair can be logged several times; keep the mean watch ratio.
    agg = pd.DataFrame({"u": bu, "i": bi, "w": big.watch_ratio.values}).groupby(["u", "i"]).w.mean()
    O = torch.zeros(m, n, dtype=torch.float64)
    Y = torch.zeros(m, n, dtype=torch.float64)
    uu = torch.as_tensor(agg.index.get_level_values(0).values)
    ii = torch.as_tensor(agg.index.get_level_values(1).values)
    O[uu, ii] = 1.0
    Y[uu, ii] = torch.as_tensor((agg.values >= threshold).astype(np.float64))
    su = small.user_id.map(uidx).values
    si = small.video_id.map(iidx).values
    sg = pd.DataFrame({"u": su, "i": si, "w": small.watch_ratio.values}).groupby(["u", "i"]).w.mean().reset_index()
    Onp = O.numpy()
    test = []
    for u, grp in sg.groupby("u"):
        it = grp.i.values
        keep = Onp[u, it] == 0          # never rank what was already logged
        it = it[keep]
        rel = (grp.w.values[keep] >= threshold).astype(np.float64)
        test.append((int(u), it, rel))
    return {"name": "kuairec", "O": O, "Y": Y, "P_given": None, "test": test}


def split_test(test, frac_val, seed, by="entry"):
    """Split the unbiased data into validation (tuning) and test parts.

    by="entry": per user, a random fraction of candidates goes to validation
                (Coat, where every user has only 16 MAR ratings);
    by="user" : a random fraction of users goes to validation (KuaiRec).
    """
    rng = np.random.default_rng(seed)
    val, tst = [], []
    if by == "user":
        perm = rng.permutation(len(test))
        nv = int(round(frac_val * len(test)))
        vs = set(perm[:nv].tolist())
        for k, t in enumerate(test):
            (val if k in vs else tst).append(t)
        return val, tst
    for u, it, rel in test:
        mask = rng.random(len(it)) < frac_val
        val.append((u, it[mask], rel[mask]))
        tst.append((u, it[~mask], rel[~mask]))
    return val, tst


def load(name):
    if name == "coat":
        return load_coat()
    if name == "kuairec":
        return load_kuairec()
    raise ValueError(name)
