"""Tests of the lifelong-aware deep baselines (drup/din.py, experiments/v12_build_seq.py, experiments/run_din.py).

(a) leakage: the sequence builder keeps only pre-cut events on videos outside the random pool (run on a tiny archive);
(b) a shuffle of the users' histories removes DIN's advantage on a small synthetic world;
(c) SIM-hard attends only to history events that share the candidate's tag or author;
(d) the default DAX grid of experiments/run_dax.py is unchanged, the extension only adds configurations.
"""

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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from drup.din import DIN, fit_bpr, load_sequences, score_rows  # noqa: E402
from drup.leaf import CUT_MS  # noqa: E402
from drup.metrics import evaluate  # noqa: E402

CUT = CUT_MS
POOL = [1, 2, 3, 4]


def _archive(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, df in members.items():
            data = df.to_csv(index=False).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_sequences_contain_only_pre_cut_out_of_pool_events(tmp_path):
    feats = pd.DataFrame({"video_id": [1, 2, 3, 4, 5, 6, 7, 8],
                          "author_id": [10, 11, 11, 10, 10, 11, 12, 13],
                          "music_id": [100, 101, 100, 101, 100, 101, 102, 103],
                          "tag": ["7", "8", "7", "8", "7", "8", "9", "7"]})
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pd.DataFrame({"video_id": POOL}).to_csv(pool_dir / "log_random_4_22_to_5_08_pure.csv", index=False)
    # user, video, time_ms, is_click, long_view; pool videos and post-cut rows carry marker clicks that must not appear
    rows = [(1, 5, CUT - 1000, 1, 0), (1, 6, CUT - 800, 0, 1), (1, 1, CUT - 500, 1, 0), (1, 5, CUT + 1000, 1, 1),
            (2, 6, CUT - 400, 1, 0), (2, 8, CUT - 300, 0, 0), (2, 7, CUT + 1100, 1, 1), (2, 2, CUT - 200, 1, 1),
            (3, 7, CUT - 100, 1, 0)]
    log = pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms", "is_click", "long_view"])
    arc = tmp_path / "arc.tar.gz"
    _archive(arc, {"kr/features.csv": feats, "kr/log_a.csv": log})
    out = tmp_path / "seq.npz"
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "v12_build_seq.py"), "--archive", str(arc),
                    "--members", "kr/features.csv", "kr/log_a.csv", "--pool_dir", str(pool_dir), "--version", "tiny",
                    "--L", "5", "--chunksize", "4", "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
    Z = np.load(out, allow_pickle=True)
    users = Z["users"].tolist()
    assert users == [1, 2, 3]
    kept = {}
    for r, u in enumerate(users):
        L = Z["L"].item()
        k = int(Z["length"][r])
        cols = slice(L - k, L)
        assert np.all(Z["time_ms"][r, :L - k] == 0)           # left padding only
        assert np.all(Z["time_ms"][r, cols] < CUT), "a post-cut event reached the sequences"
        assert not np.isin(Z["video"][r, cols], POOL).any(), "a pool video reached the sequences"
        kept[u] = list(zip(Z["video"][r, cols].tolist(), Z["time_ms"][r, cols].tolist()))
        assert np.all(np.diff(Z["time_ms"][r, cols]) >= 0), "events not time-ordered"
    assert kept[1] == [(5, CUT - 1000), (6, CUT - 800)]        # pool video 1 and the post-cut row are absent
    assert kept[2] == [(6, CUT - 400), (8, CUT - 300)]         # pool video 2 and the post-cut row are absent
    assert kept[3] == [(7, CUT - 100)]
    assert Z["n_events"].tolist() == [2, 2, 1]                 # counts of the pre-cut out-of-pool events
    # the model's loader aligns the rows with the dataset users; a user without a sequence has length 0
    item_feat = pd.DataFrame({"tag": ["7", "8", "9", "nan"], "author_id": [10, 11, 12, np.nan],
                              "music_id": [100, 101, 102, np.nan]})
    seq = load_sequences(out, np.array([3, 4, 1]), item_feat, L=5)
    assert seq["length"].tolist() == [1, 0, 2]
    assert not bool(seq["valid"][1].any())
    assert int(seq["cand"][3].abs().sum()) == 0                # the item without key values maps to code 0


def _world(seed=0, m=200, n=80, n_tags=5, n_auth=7, n_music=9, hist=60, log_per=1, test_per=40):
    """Synthetic world for (b). The in-pool log is nearly empty: each user has one click, on an item with the user's
    favourite tag, so the preference is not learnable from the in-pool data (and the ID embeddings are switched off in
    the test, use_id=False). The lifelong history (out-of-pool events) carries the favourite tag as its frequent tag,
    so the history is the only source of the preference. The random test candidates are relevant with a high probability
    on the favourite tag."""
    rng = np.random.default_rng(seed)
    fav = rng.integers(0, n_tags, m)
    item_tag = np.arange(n) % n_tags
    item_auth = np.arange(n) % n_auth
    item_music = np.arange(n) % n_music
    X = np.zeros((m, n), bool)
    for u in range(m):
        cand = np.flatnonzero(item_tag == fav[u])
        X[u, rng.choice(cand, log_per, replace=False)] = True
    # lifelong history (right-aligned, time-ordered): tags drawn mostly from the favourite tag
    L = hist
    tag = np.zeros((m, L), np.int32)
    auth = np.zeros((m, L), np.int32)
    music = np.zeros((m, L), np.int32)
    sig = np.zeros((m, L), np.int8)
    for u in range(m):
        t = np.where(rng.random(L) < 0.6, fav[u], rng.integers(0, n_tags, L))
        tag[u] = t + 1
        auth[u] = rng.integers(0, n_auth, L) + 1
        music[u] = rng.integers(0, n_music, L) + 1
        sig[u] = (2 * (rng.random(L) < 0.4)).astype(np.int8)
    npz = {"users": np.arange(m, dtype=np.int64), "L": np.array(L), "cut_ms": np.array(CUT),
           "vocab_tag": np.array([str(i) for i in range(n_tags)]), "vocab_author": np.arange(n_auth, dtype=np.int64),
           "vocab_music": np.arange(n_music, dtype=np.int64), "tag": tag, "author": auth, "music": music, "sig": sig,
           "video": np.zeros((m, L), np.int64), "time_ms": np.zeros((m, L), np.int64),
           "length": np.full(m, L, np.int32), "n_events": np.full(m, L, np.int64)}
    test = []
    for u in range(m):
        it = rng.choice(np.flatnonzero(~X[u]), test_per, replace=False)
        p = np.where(item_tag[it] == fav[u], 0.7, 0.05)
        rel = (rng.random(test_per) < p).astype(np.float64)
        if rel.sum() == 0:
            rel[0] = 1.0
        test.append((u, it, rel))
    item_feat = pd.DataFrame({"tag": item_tag.astype(str), "author_id": item_auth, "music_id": item_music})
    return npz, X, test, item_feat


def _train_eval(npz, X, test, item_feat, path, seed=0, epochs=30, use_id=False):
    np.savez(path, **npz)
    seq = load_sequences(path, np.arange(X.shape[0]), item_feat, L=npz["L"].item())
    torch.manual_seed(seed)
    m, n = X.shape
    model = DIN(m, n, seq["n_tag"], seq["n_author"], seq["n_music"], seq["cand"], d=8, hidden=8, mlp=16,
                use_id=use_id)
    fit_bpr(model, torch.as_tensor(X), seq["tok"], seq["valid"], lr=1e-2, wd=1e-5, epochs=epochs, batch=32,
            on_epoch=lambda mod, ep: False, seed=seed)
    rows = torch.arange(m)
    cand = torch.arange(n)[None, :].repeat(m, 1)
    S = score_rows(model, rows, cand, seq["tok"], seq["valid"]).numpy()
    return evaluate(S, test, (10,))["ndcg@10"]


def test_shuffling_histories_removes_din_advantage(tmp_path):
    npz, X, test, item_feat = _world()
    true_val = _train_eval(npz, X, test, item_feat, tmp_path / "true.npz")
    rng = np.random.default_rng(1)
    perm = rng.permutation(X.shape[0])
    shuffled = dict(npz)
    for k in ("tag", "author", "music", "sig"):
        shuffled[k] = npz[k][perm]
    shuf_val = _train_eval(shuffled, X, test, item_feat, tmp_path / "shuf.npz")
    # measured on this world (use_id=False, 30 epochs): true 0.741, shuffled 0.208, gap 0.533
    assert true_val - shuf_val > 0.2, (true_val, shuf_val)


def test_sim_attends_only_to_matching_events():
    torch.manual_seed(0)
    m, n, L = 4, 10, 12
    cand = np.stack([np.arange(n) % 4 + 1, np.arange(n) % 3 + 1, np.arange(n) % 2 + 1], 1)  # codes (tag, author, music)
    cand = torch.as_tensor(cand.astype(np.int64))
    sim = DIN(m, n, 5, 5, 5, cand, d=8, hidden=8, mlp=8, sim_k=4)
    din = DIN(m, n, 5, 5, 5, cand, d=8, hidden=8, mlp=8, sim_k=0)
    din.load_state_dict(sim.state_dict())
    g = torch.Generator().manual_seed(3)
    tok = torch.randint(1, 5, (1, L, 4), generator=g)
    tok[..., 3] = torch.randint(0, 4, (1, L), generator=g)     # signal codes 0..3
    valid = torch.ones(1, L, dtype=torch.bool)
    rows = torch.tensor([0])
    c = torch.tensor([[2]])                                  # candidate item 2: tag code 3, author code 2
    ctag, cauth = int(cand[2, 0]), int(cand[2, 1])
    base_sim = sim.score(rows, c, tok, valid)
    base_din = din.score(rows, c, tok, valid)
    # overwrite every event that shares neither the candidate's tag nor its author
    other = (tok[0, :, 0] != ctag) & (tok[0, :, 1] != cauth)
    tok2 = tok.clone()
    tok2[0, other, :] = torch.randint(1, 5, (int(other.sum()), 4), generator=g)
    tok2[0, other, 3] = torch.randint(0, 4, (int(other.sum()),), generator=g)
    tok2[0, other, 0] = torch.where(tok2[0, other, 0] == ctag, ctag % 4 + 1, tok2[0, other, 0])
    tok2[0, other, 1] = torch.where(tok2[0, other, 1] == cauth, cauth % 4 + 1, tok2[0, other, 1])
    assert torch.allclose(sim.score(rows, c, tok2, valid), base_sim, atol=1e-6), "SIM read a non-matching event"
    assert not torch.allclose(din.score(rows, c, tok2, valid), base_din, atol=1e-6), "DIN ignored the history"


def test_default_dax_grid_is_unchanged_and_extensions_only_add():
    import run_dax as RD
    expected = [dict(prior=p, centre=c, k_life=k, tab=t, prior_s=ps)
                for p in (80.0, 160.0) for c in ("tag", "content") for k in (0.0, 100.0) for t in (False, True)
                for ps in (None, 80.0)]
    assert RD.post_grid() == expected
    assert len(RD.post_grid()) == 32
    assert RD.post_grid(small=True) == [dict(prior=80.0, centre="tag", k_life=100.0, tab=True, prior_s=None)]
    wide = RD.post_grid(False, (10.0, 20.0, 40.0, 80.0, 160.0), (0.0, 100.0, 200.0, 400.0, 1000.0))
    assert len(wide) == 5 * 2 * 5 * 2 * 2 == 200
    assert all(c in wide for c in expected)
    assert {c["prior"] for c in wide} == {10.0, 20.0, 40.0, 80.0, 160.0}
    assert {c["k_life"] for c in wide} == {0.0, 100.0, 200.0, 400.0, 1000.0}
