"""Leakage tests of the lifelong channels (DESIGN.md Sections 5.1, 6.1, 7 stage 0).

(a) the lifelong tables use only out-of-pool history strictly before the random-exposure period (checked on the
    real builder experiments/v9_build_lifelong.py, run on a tiny synthetic archive);
(b) test and validation labels never enter a channel or the closed-form model;
(c) a shuffled-users control removes the lifelong gain on a small synthetic example.
"""

import inspect
import io
import os
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drup.leaf import (CUT_MS, LifelongTables, collective_gram, collective_solve, compact_base,  # noqa: E402
                       fit_tied, key_strings)
from leaf_synth import ndcg10, world  # noqa: E402

CUT = CUT_MS


def _archive(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, df in members.items():
            data = df.to_csv(index=False).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


FEATS = pd.DataFrame({
    "video_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "author_id": [10, 11, 11, 10, 10, 11, 12, 13],
    "music_id": [100, 101, 100, 101, 100, 101, 102, 103],
    "tag": ["7", "8", "7", "8", "7", "8", "9", "7"],
})
POOL = [1, 2, 3, 4]


def _events(variant):
    """Standard-log rows. variant "b" changes every row after the cut-off and every pool-video row; the tables of
    the pre-cut-off window must not move."""
    rows = [  # user, video, time_ms, click, long_view
        (1, 5, CUT - 1000, 1, 0), (1, 5, CUT - 900, 0, 0), (1, 6, CUT - 800, 0, 0), (1, 7, CUT - 700, 1, 0),
        (1, 8, CUT - 600, 0, 0), (1, 1, CUT - 500, 1, 0), (2, 6, CUT - 400, 1, 0), (1, 5, CUT + 1000, 1, 0),
        (2, 5, CUT + 1100, 0, 0),
    ]
    if variant == "b":
        rows = [(u, v, t, 1 - c if (t >= CUT or v in POOL) else c, lv) for u, v, t, c, lv in rows]
        rows += [(1, 6, CUT + 5000, 1, 1), (2, 1, CUT - 300, 1, 1), (2, 8, CUT + 9000, 1, 0)]
    return pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms", "is_click", "long_view"])


def _build(tmp, variant):
    pool_dir = tmp / f"pool_{variant}"
    pool_dir.mkdir(exist_ok=True)
    FEATS[FEATS.video_id.isin(POOL)].to_csv(pool_dir / "video_features_basic_pure.csv", index=False)
    pd.DataFrame({"video_id": POOL}).to_csv(pool_dir / "log_random_4_22_to_5_08_pure.csv", index=False)
    arc = tmp / f"arc_{variant}.tar.gz"
    _archive(arc, {"kr/video_features_basic_pure.csv": FEATS, "kr/log_standard_a.csv": _events(variant)})
    out = tmp / f"out_{variant}.npz"
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "v9_build_lifelong.py"), "--archive", str(arc),
                    "--version", "tiny", "--pool_dir", str(pool_dir), "--out", str(out), "--chunksize", "1000",
                    "--members", "kr/video_features_basic_pure.csv", "kr/log_standard_a.csv"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    return dict(np.load(out, allow_pickle=True))


def _pairs(npz, window, key):
    df = pd.DataFrame({"u": npz[f"{key}_{window}_u"], "k": npz[f"{key}_{window}_k"]})
    df[["n", "c", "l"]] = npz[f"{key}_{window}_ncl"]
    vals = np.asarray(npz[f"{key}_values"]).astype(str)
    return {(int(r.u), vals[int(r.k)]): (int(r.n), int(r.c)) for r in df.itertuples()}


def test_builder_uses_out_of_pool_history_strictly_before_cut(tmp_path):
    a = _build(tmp_path, "a")
    tot = dict(zip(a["tot_pre_user"].tolist(), a["tot_pre"].tolist()))
    # user 1 before the cut: out-of-pool rows 5, 5, 6, 7, 8 (pool row 1 and the post-cut row 5 excluded)
    assert tot[1] == [5.0, 2.0, 0.0]
    assert tot[2] == [1.0, 1.0, 0.0]          # user 2: video 6 before the cut; video 5 and 1 excluded
    tot_all = dict(zip(a["tot_all_user"].tolist(), a["tot_all"].tolist()))
    assert tot_all[1][0] == 6.0               # the post-cut row on video 5 is in the all-window only
    tag = _pairs(a, "pre", "tag")
    assert tag[(1, "7")] == (3, 1)            # videos 5, 5 and 8 (tag 7); video 7 (tag 9) is not a pool value
    assert tag[(1, "8")] == (1, 0)
    assert (1, "9") not in tag
    author = _pairs(a, "pre", "author_id")
    assert author[(1, "10")] == (2, 1)
    assert author[(1, "11")] == (1, 0)
    assert (1, "13") not in author            # author 13 has no pool item
    music = _pairs(a, "pre", "music_id")
    assert music[(1, "100")] == (2, 1)
    assert music[(2, "101")] == (1, 1)


def test_pre_window_tables_ignore_post_cut_and_pool_rows(tmp_path):
    a = _build(tmp_path, "a")
    b = _build(tmp_path, "b")
    for key in a:
        if "_pre" in key:
            assert np.array_equal(a[key], b[key]), key


def test_tables_ignore_npz_rows_of_users_outside_the_dataset():
    d = world(seed=3, m=40)
    feat = d["item_feat"]
    npz = dict(d["npz"])
    base = LifelongTables(npz, d["users_raw"], feat, window="pre", s=5.0)
    extra = dict(npz)
    for k in ("tag", "author_id", "music_id"):
        extra[f"{k}_pre_u"] = np.concatenate([npz[f"{k}_pre_u"], [999999]])
        extra[f"{k}_pre_k"] = np.concatenate([npz[f"{k}_pre_k"], [0]]).astype(np.int32)
        extra[f"{k}_pre_ncl"] = np.concatenate([npz[f"{k}_pre_ncl"], [[50.0, 40.0, 0.0]]]).astype(np.float32)
    extra["tot_pre_user"] = np.concatenate([npz["tot_pre_user"], [999999]])
    extra["tot_pre"] = np.concatenate([npz["tot_pre"], [[50.0, 40.0, 0.0]]]).astype(np.float32)
    other = LifelongTables(extra, d["users_raw"], feat, window="pre", s=5.0)
    for k in ("tag", "author_id", "music_id"):
        assert torch.equal(base.tab[k], other.tab[k])


def test_lift_and_familiarity_match_the_formula():
    d = world(seed=4, m=50)
    feat, npz = d["item_feat"], d["npz"]
    tabs = LifelongTables(npz, d["users_raw"], feat, window="pre", s=5.0)
    tot = dict(zip(npz["tot_pre_user"].tolist(), npz["tot_pre"].tolist()))
    u, k_val = 3, "tag"
    raw = int(d["users_raw"][u])
    C, N = tot[raw][1], tot[raw][0]
    b = (C + 1.0) / (N + 2.0)
    vals = np.asarray(npz["tag_values"]).astype(str)
    cnt = {int(kk): (n, c) for uu, kk, (n, c, _) in zip(npz["tag_pre_u"], npz["tag_pre_k"], npz["tag_pre_ncl"])
           if uu == raw}
    for i in range(0, len(feat), 7):
        code = pd.Index(vals).get_indexer([key_strings(feat[k_val], k_val).iloc[i]])[0]
        n, c = cnt.get(code, (0.0, 0.0))
        lift = (c + 5.0 * b) / (n + 5.0) - b if n > 0 else 0.0
        fam = np.log1p(n)
        got = tabs.pairs(torch.tensor([u]), torch.tensor([i]))[0, :2].numpy()
        scale = tabs.scale["tag"].numpy()
        assert abs(got[0] - lift / scale[0]) < 2e-3 * max(1.0, abs(lift / scale[0])), i
        assert abs(got[1] - fam / scale[1]) < 2e-3 * max(1.0, abs(fam / scale[1])), i


def test_labels_of_test_and_validation_never_enter_the_channels():
    forbidden = {"test", "tst", "val", "valid", "validation", "rel", "labels_test", "uniform"}
    for fn in (LifelongTables.__init__, collective_gram, collective_solve, fit_tied, compact_base):
        assert not forbidden & set(inspect.signature(fn).parameters), fn.__name__
    d = world(seed=5, m=60)
    d2 = dict(d)
    d2["test"] = [(u, it, 1.0 - rel) for u, it, rel in d["test"]]     # every test label flipped

    def pipeline(dd):
        tabs = LifelongTables(dd["npz"], dd["users_raw"], dd["item_feat"], window="pre", s=5.0)
        G, C = collective_gram(dd["O"], dd["Y"], 0.5)
        B, _ = collective_solve(G, C, dd["O"].shape[1], 1000.0)
        rows = torch.arange(dd["O"].shape[0])
        return tabs.dense_rows(rows, torch.tensor([1.0, 0, 0, 0, 0, 0])), B

    S1, B1 = pipeline(d)
    S2, B2 = pipeline(d2)
    assert torch.equal(S1, S2) and torch.equal(B1, B2)
    # a changed label of a test candidate changes the metric, never the model
    assert ndcg10(S1, d["test"]) != ndcg10(S1, d2["test"])


def test_shuffled_users_remove_the_lifelong_gain():
    d = world(seed=1)
    m, n = d["O"].shape
    rows = torch.arange(m)
    w_tag = torch.tensor([1.0, 0, 0, 0, 0, 0])
    real = LifelongTables(d["npz"], d["users_raw"], d["item_feat"], window="pre", s=5.0)
    shuf = LifelongTables(d["npz"], d["users_raw"], d["item_feat"], window="pre", s=5.0, shuffle=True)
    torch.manual_seed(0)
    noise = 1e-6 * torch.randn(m, n)
    none = ndcg10(noise, d["test"])
    gain_real = ndcg10(real.dense_rows(rows, w_tag) + noise, d["test"]) - none
    gain_shuf = ndcg10(shuf.dense_rows(rows, w_tag) + noise, d["test"]) - none
    assert gain_real > 0.10, gain_real
    assert abs(gain_shuf) < 0.03, gain_shuf
    assert gain_shuf < gain_real / 4
