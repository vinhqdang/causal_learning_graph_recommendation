"""Ranking metrics on unbiased candidate sets."""

import numpy as np
import torch


def evaluate(scores, test, ks=(5, 10), row_of=None):
    """scores: (rows, n) tensor; test: list of (user, items, relevance).

    row_of maps a user id to its row in ``scores`` (identity if None).
    Users without any relevant candidate are skipped (NDCG undefined).
    """
    out = {f"ndcg@{k}": [] for k in ks}
    out.update({f"recall@{k}": [] for k in ks})
    S = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    for u, items, rel in test:
        if len(items) == 0 or rel.sum() == 0:
            continue
        r = u if row_of is None else row_of[u]
        s = S[r, items]
        order = np.argsort(-s, kind="stable")
        rs = rel[order]
        npos = rel.sum()
        for k in ks:
            top = rs[:k]
            disc = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = (top * disc).sum()
            ideal = disc[: int(min(npos, k))].sum()
            out[f"ndcg@{k}"].append(dcg / ideal)
            out[f"recall@{k}"].append(top.sum() / npos)
    return {k: float(np.mean(v)) for k, v in out.items()}
