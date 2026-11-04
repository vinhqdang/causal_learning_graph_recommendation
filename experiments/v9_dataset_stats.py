"""Statistics and leak checks of the KuaiRand loaders of drup/data2.py.

python3 experiments/v9_dataset_stats.py --version pure [--n_users 10000] [--label is_click] [--check_v1]
python3 experiments/v9_dataset_stats.py --version 1k
--check_v1 also verifies that O, Y and the test list equal drup.data.load_kuairand (the stored setting).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from drup.data import split_test  # noqa: E402
from drup.data2 import SIGNALS, load_kuairand_rich  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="pure", choices=["pure", "1k"])
    ap.add_argument("--n_users", type=int, default=10000, help="0 = all users")
    ap.add_argument("--label", default="is_click")
    ap.add_argument("--train_until", type=int, default=None)
    ap.add_argument("--check_v1", action="store_true")
    a = ap.parse_args()
    d = load_kuairand_rich(version=a.version, n_users=(a.n_users or None), label=a.label,
                           train_until=a.train_until)
    O, Y = d["O"], d["Y"]
    m, n = O.shape
    lg, tl = d["log"], d["test_log"]
    print(f"== KuaiRand-{a.version}  label={a.label}  train_until={a.train_until}")
    print(f"users {m}  items {n}  training impressions {len(lg)}  distinct exposed pairs {int(O.sum())}"
          f"  density {O.mean().item():.4%}")
    print(f"impressions per user: mean {len(lg)/m:.1f}  median {lg.groupby('u').size().median():.0f}"
          f"  repeated-pair share {1 - O.sum().item()/max(len(lg),1):.3f}")
    print("training label rates per impression: " + ", ".join(f"{s} {lg[s].mean():.4f}" for s in SIGNALS)
          + f", play_ratio mean {lg.play_ratio.mean():.3f}")
    print("training tab shares: " + str(lg.tab.value_counts(normalize=True).round(3).head(8).to_dict()))
    print(f"random (test) impressions {len(tl)}  users {tl.u.nunique()}  items {tl.i.nunique()}")
    print("test label rates per impression: " + ", ".join(f"{s} {tl[s].mean():.4f}" for s in SIGNALS))
    print("test tab shares: " + str(tl.tab.value_counts(normalize=True).round(3).to_dict()))
    cand = [len(it) for _, it, _ in d["test"]]
    pos = [r.sum() for _, _, r in d["test"]]
    ev = sum(1 for c, p in zip(cand, pos) if c > 0 and p > 0)
    print(f"test users {len(d['test'])}  candidates/user mean {np.mean(cand):.1f}  positives/user mean {np.mean(pos):.2f}"
          f"  users with >= 1 positive (evaluated) {ev}")
    va, te = split_test(d["test"], 0.3, 0, by="user")
    print(f"split 0 (by user, frac_val 0.3): validation users {len(va)}  test users {len(te)}")
    print(f"user features {d['user_feat'].shape}  item features {list(d['item_feat'].columns)}"
          f"  items with author {d['item_feat'].author_id.notna().mean():.3f}")
    print("time range train", lg.time_ms.min(), lg.time_ms.max(), " test", tl.time_ms.min(), tl.time_ms.max())
    if "outpool" in d:
        op = d["outpool"]
        print(f"out-of-pool training impressions {len(op)}  videos {op.video_id.nunique()}  per user {len(op)/m:.0f}")
    print("leak checks:")
    for k, v in d["meta"]["leak"].items():
        print(f"   {k}: {v}")
    assert d["meta"]["leak"]["ranked_pairs_present_in_training_log"] == 0
    assert d["meta"]["leak"]["test_events_present_in_training_log"] == 0
    if a.check_v1:
        from drup.data import load_kuairand
        v1 = load_kuairand()
        same = (v1["O"].shape == O.shape and bool((v1["O"] == O).all()) and bool((v1["Y"] == Y).all())
                and len(v1["test"]) == len(d["test"])
                and all(u1 == u2 and np.array_equal(i1, i2) and np.array_equal(r1, r2)
                        for (u1, i1, r1), (u2, i2, r2) in zip(v1["test"], d["test"])))
        print(f"identical to drup.data.load_kuairand (stored setting): {same}")
        assert same


if __name__ == "__main__":
    main()
