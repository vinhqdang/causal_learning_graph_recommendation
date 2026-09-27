"""End-to-end construction of a propagation recommender from one config.

method in {"Obs", "IPS", "IPS+WC", "DR", "DRUP"}; the configuration keys are
those selected by experiments/run_filters.py (alpha, floor, lam, deg, beta).
"""

import torch

from .estimation import baseline_imputation, clip_propensity, lowrank_imputation
from .propagation import degree_weights, edge_estimate, three_hop


def impute(O, Y, P, cfg):
    """Outcome imputation: additive ('add', default) or low-rank ('lr')."""
    if cfg.get("imp", "add") == "lr":
        return lowrank_imputation(O, Y, P, rank=cfg.get("rank", 32), lam=cfg.get("lr_lam", 10.0),
                                  ridge=cfg["lam"])
    return baseline_imputation(O, Y, P, lam=cfg["lam"])


def build(O, Y, P_raw, method, cfg):
    if method == "Obs":
        P = torch.ones_like(O)
        Yhat = None
    else:
        P = clip_propensity(P_raw, cfg["floor"])
        Yhat = impute(O, Y, P, cfg) if method.startswith("DR") else None
    W = edge_estimate(O, Y, P, Yhat)
    D = Yhat if cfg.get("deg", "W") == "Yhat" and Yhat is not None else None
    C = degree_weights(W, cfg["alpha"], D=D)
    return {"method": method, "cfg": cfg, "P": P, "Yhat": Yhat, "W": W, "C": C,
            "correct": method in ("IPS+WC", "DRUP")}


def raw_parts(M, rows):
    s1 = (M["C"] * M["W"])[rows]
    s3 = three_hop(M["W"], M["C"], rows=rows, correct=M["correct"])
    return s1, s3


def scale_constants(M, rows):
    """Global (not per-user) scale constants so that the final score is a
    fixed linear combination of the 1-hop and 3-hop terms. Fixed constants
    keep every score affine in the user's own row (needed for exact
    explanations); rankings are within-user so this only sets beta's unit."""
    s1, s3 = raw_parts(M, rows)
    return float(s1.abs().mean().clamp_min(1e-12)), float(s3.abs().mean().clamp_min(1e-12))


def scores(M, rows, consts=None):
    s1, s3 = raw_parts(M, rows)
    beta = M["cfg"].get("beta", 1.0)
    if consts is None:
        # per-user unit normalisation, as used for model selection
        s1 = s1 / s1.abs().mean(1, keepdim=True).clamp_min(1e-12)
        s3 = s3 / s3.abs().mean(1, keepdim=True).clamp_min(1e-12)
    else:
        s1, s3 = s1 / consts[0], s3 / consts[1]
    if beta >= 1e3:
        return s3
    return s1 + beta * s3


def scores_with_consts(M, rows):
    """scores(M, rows, scale_constants(M, rows)) with a single propagation."""
    s1, s3 = raw_parts(M, rows)
    c1 = float(s1.abs().mean().clamp_min(1e-12))
    c3 = float(s3.abs().mean().clamp_min(1e-12))
    beta = M["cfg"].get("beta", 1.0)
    S = s3 / c3 if beta >= 1e3 else s1 / c1 + beta * s3 / c3
    return S, (c1, c3)
