"""Ranking metrics: UserMetrics (nDCG@k, recall@k, MRR, AUC) and the per-metric test values of BucketSelector against a
brute-force implementation on toy candidate lists, including ties and unequal list lengths."""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drup.leaf import EXTRA_METRICS, BucketSelector, Layout  # noqa: E402
from drup.metrics import UserMetrics  # noqa: E402

KS = (5, 10, 20)


def brute(s, rel):
    """Reference metrics of one candidate list: scores s, binary relevance rel (at least one relevant). Ties are
    broken by candidate-list position (stable descending sort), as in drup.metrics.evaluate."""
    order = np.argsort(-s, kind="stable")
    r = rel[order]
    npos = int(rel.sum())
    out = {}
    for k in KS:
        top = r[:k]
        disc = 1.0 / np.log2(np.arange(2, len(top) + 2))
        ideal = (1.0 / np.log2(np.arange(2, min(npos, k) + 2))).sum()
        out[f"ndcg@{k}"] = float((top * disc).sum() / ideal)
        out[f"recall@{k}"] = float(top.sum() / npos)
    out["mrr"] = 1.0 / (int(np.flatnonzero(r > 0)[0]) + 1)
    return out


def test_mrr_and_ndcg_hand_cases():
    """Hand-computed: scores 3, 1, 2 with the relevant item last in the order -> rank 3; a tie at the top."""
    for s, rel, want in [([3.0, 1.0, 2.0], [0.0, 1.0, 0.0], 1 / 3),
                         ([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], 1 / 3),      # stable: ties keep list order
                         ([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 1.0),
                         ([0.5, 0.2, 0.9, 0.1], [0.0, 0.0, 0.0, 1.0], 1 / 4)]:
        um = UserMetrics([(0, np.arange(len(s)), np.asarray(rel))], KS, None, "cpu")
        got = um.compute(torch.tensor([s], dtype=torch.float64))["mrr"].item()
        assert abs(got - want) < 1e-12, (s, rel, got, want)
    # nDCG@5 of relevance (0, 1, 0, 1) scored 4, 3, 2, 1: DCG = 1/log2(3) + 1/log2(5), ideal = 1 + 1/log2(3)
    um = UserMetrics([(0, np.arange(4), np.array([0.0, 1.0, 0.0, 1.0]))], KS, None, "cpu")
    got = um.compute(torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float64))["ndcg@5"].item()
    want = (1 / np.log2(3) + 1 / np.log2(5)) / (1 + 1 / np.log2(3))
    assert abs(got - want) < 1e-12, (got, want)


def test_user_metrics_match_brute_force():
    rng = np.random.default_rng(0)
    n = 70
    scores = torch.tensor(rng.integers(0, 4, size=(25, n)), dtype=torch.float64)    # integer scores: many ties
    test = []
    for u in range(25):
        L = int(rng.integers(1, 60))
        items = rng.choice(n, L, replace=False)
        rel = (rng.random(L) < 0.2).astype(float)
        rel[0] = 1.0                                                                  # at least one relevant
        test.append((u, items, rel))
    out = UserMetrics(test, KS, None, "cpu").compute(scores)
    for i, (u, items, rel) in enumerate(test):
        ref = brute(scores[u, items].numpy(), rel)
        for m in ref:
            assert abs(out[m][i].item() - ref[m]) < 1e-12, (u, m, out[m][i].item(), ref[m])


def test_bucket_selector_test_metrics_match_brute_force():
    """BucketSelector: the chosen (validation nDCG@10) evaluation's test values of every metric, per split and pooled,
    equal the brute-force values of the test users of that split; per_user_extra[ndcg@10] is per_user."""
    rng = np.random.default_rng(3)
    test = []
    for u in range(40):
        L = int(rng.integers(1, 90))                       # spans several buckets (edges 32, 64, 128)
        test.append((100 + u, rng.choice(10000, L, replace=False), (rng.random(L) < 0.25).astype(float)))
        test[-1][2][0] = 1.0
    layout = Layout(test)
    ctest = [(k, np.arange(len(it)), rel) for k, (_, it, rel) in enumerate(test)]
    splits = []
    for s in range(2):
        perm = rng.permutation(len(ctest))
        splits.append(([ctest[k] for k in perm[:14]], [ctest[k] for k in perm[14:]]))
    row_ids = np.array([u for u, _, _ in test])
    sel = BucketSelector(layout, splits, (10,), "ndcg@10", row_ids)
    # two evaluations with random bucketed scores; the second one is the better on validation for some splits
    evals = []
    for c in range(2):
        S = [torch.tensor(rng.integers(0, 3, size=(len(bk["k"]), bk["cand"].shape[1])), dtype=torch.float64)
             for bk in layout.buckets]
        evals.append(S)
        sel.add(S, dict(id=c), active=[True, True])

    def full_scores(S, k):
        return S[layout.bucket_of[k]][layout.local_of[k]].numpy()

    for s, (val, tst) in enumerate(splits):
        vals = []
        for c in range(2):
            vals.append(np.mean([brute(full_scores(evals[c], k)[: len(it)], rel)["ndcg@10"]
                                 for k, it, rel in val]))
        best = int(np.argmax(vals))
        assert sel.best[s]["cfg"]["id"] == best, (s, vals)
        want = [brute(full_scores(evals[best], k)[: len(it)], rel) for k, it, rel in tst]
        res = sel.result()
        for m in EXTRA_METRICS:
            mean_m = np.mean([w[m] for w in want])
            assert abs(res["per_split"][s][m] - mean_m) < 1e-12, (s, m)
            for (k, _, _), w in zip(tst, want):
                got = res["per_user_extra"][m][s][str(int(row_ids[k]))]
                assert abs(got - w[m]) < 1e-12, (s, m, k)
    res = sel.result()
    assert set(res["test_metrics"]) == set(EXTRA_METRICS)
    for m in EXTRA_METRICS:
        assert abs(res["test_metrics"][m][0] - np.mean([p[m] for p in res["per_split"]])) < 1e-15
    assert res["per_user"] == res["per_user_extra"]["ndcg@10"]
    assert res["test_mean"] == res["test_metrics"]["ndcg@10"][0]
