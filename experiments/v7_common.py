"""Shared helpers for the analysis-only scripts v7_reconcile.py, v7_sensitivity.py, v7_tests.py,
v7_bootstrap_rank.py and v7_tables.py (design: paper/v4/DESIGN.md, Section 3.4).

Nothing here trains a model. The helpers read the stored result files (results/v*/learned_*.json,
results/v2/filters_*.json) and the stored validation/test banks (results/v*/bank_*.json).

- replay_bank() re-runs the training-time selection of experiments/run_learned.py (record() and the
  per-split patience stop) over the stored (configuration, epoch) rows of a bank.
- combine(), peru(), cmp(), holm() are copies of the helpers of experiments/final_analysis.py. That file
  runs its analysis when imported, so it cannot be imported here; its code is not modified.
- The method registry below lists the methods of the paper's main table with the result files that hold
  them. A file that is missing (for example an X-run that has not been executed yet) is skipped with a
  printed note, so the scripts can be re-run later.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/v6"
V2, V3, V4, V5, V6 = "results/v2/", "results/v3/", "results/v4/", "results/v5/", "results/v6/"

KEY = {"coat": "ndcg@5", "yahoo": "ndcg@5", "kuairand": "ndcg@10", "kuairec": "ndcg@20"}
NAME = {"coat": "Coat", "yahoo": "Yahoo", "kuairand": "KuaiRand", "kuairec": "KuaiRec"}
# equivalence margins of DESIGN.md Section 2.3 (provisional until TODO-L2 is closed)
MARGIN = {"coat": 0.010, "yahoo": 0.010, "kuairand": 0.003, "kuairec": 0.003}
# run_learned.py default; used only where a bank records no patience (never for the registry below)
PATIENCE_DEFAULT = 5
FRAC_VAL = 0.3


class Report:
    """Prints lines and keeps them for the summary file in results/v6/."""

    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)

    def note(self, s):
        self(f"NOTE: {s}")

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def exists(path, rep=None):
    ok = os.path.exists(path)
    if not ok and rep is not None:
        rep.note(f"missing input {path}; skipped")
    return ok


def bank_path(res_path):
    d, b = os.path.split(res_path)
    return os.path.join(d, "bank_" + b)


def cfg_key(cfg):
    return json.dumps({k: v for k, v in cfg.items() if k != "epoch"}, sort_keys=True)


def same_cfg(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def bank_rows(res_path, method):
    """(bank args, rows) of `method` from the bank written next to the result file, or None."""
    p = bank_path(res_path)
    if not os.path.exists(p):
        return None
    j = load_json(p)
    if method not in j.get("banks", {}):
        return None
    return j.get("args", {}), j["banks"][method]


def replay_bank(rows, S, patience):
    """Replay the training-time selection over bank rows.

    rows: evaluations (cfg with epoch, per-split val and test) in training order; the epochs of one
    configuration are consecutive. Replays run_learned.py record(): for each configuration and split,
    the best validation epoch is kept (strict improvement, so the first epoch wins a tie); a split stops
    after `patience` epochs without improvement (patience None: no stop). Returns per configuration:
    groups (cfg without epoch, list of rows), V (N x S best val), T (N x S test at that epoch),
    E (N x S selected epoch).
    """
    groups, last = [], None
    for r in rows:
        k = cfg_key(r["cfg"])
        if k != last:
            cfg = {kk: vv for kk, vv in r["cfg"].items() if kk != "epoch"}
            groups.append((cfg, []))
            last = k
        groups[-1][1].append(r)
    N = len(groups)
    V = np.full((N, S), -np.inf)
    T = np.full((N, S), np.nan)
    E = np.zeros((N, S), dtype=int)
    for n, (_, grp) in enumerate(groups):
        cur = [-1.0] * S
        bad = [0] * S
        done = [False] * S
        for r in grp:
            for s in range(S):
                if done[s]:
                    continue
                v = r["val"][s]
                if v > cur[s]:
                    cur[s], bad[s] = v, 0
                    V[n, s], T[n, s], E[n, s] = v, r["test"][s], r["cfg"].get("epoch", 0)
                else:
                    bad[s] += 1
                    if patience is not None and bad[s] >= patience:
                        done[s] = True
    return groups, V, T, E


def select(V, T):
    """Validation selection per split over configurations (first maximum wins, as in run_learned.py)."""
    idx = np.argmax(V, axis=0)
    cols = np.arange(V.shape[1])
    return idx, V[idx, cols], T[idx, cols]


def stored(res_path, method):
    j = load_json(res_path)
    return j["results"][method]


def combine(results, key):
    """Copy of final_analysis.combine: per split, the job with the largest stored val_best."""
    pu, ps, ch = [], [], []
    for s in range(len(results[0]["per_split"])):
        j = int(np.argmax([r["val_best"][s] for r in results]))
        pu.append(results[j]["per_user"][s])
        ps.append(results[j]["per_split"][s][key])
        ch.append(results[j]["chosen"][s])
    return {"per_user": pu, "per_split": ps, "chosen": ch}


def peru(r):
    """Copy of final_analysis.peru: mean over splits of each user's metric."""
    a = {}
    for q in r["per_user"]:
        for u, v in q.items():
            a.setdefault(u, []).append(v)
    return {u: np.mean(v) for u, v in a.items()}


def cmp(a, b):
    """Copy of final_analysis.cmp (mean difference x1e-3, t half-width x1e-3, two-sided p)."""
    x, y = peru(a), peru(b)
    us = sorted(set(x) & set(y))
    d = np.array([x[u] - y[u] for u in us])
    return 1000 * d.mean(), 1000 * stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)), stats.ttest_1samp(d, 0).pvalue


def holm(ps):
    """Copy of final_analysis.holm."""
    ps = np.asarray(ps, dtype=float)
    if len(ps) == 0:
        return ps
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def user_diffs(a_pu, b_pu):
    """Per-user difference, averaged over the splits where both have the user (significance.user_diffs)."""
    acc = {}
    for pa, pb in zip(a_pu, b_pu):
        for u, va in pa.items():
            if u in pb:
                acc.setdefault(u, []).append(va - pb[u])
    return np.array([np.mean(v) for v in acc.values()])


def paired_stats(d, margin):
    """Mean, t half-width, p, MDE (80% power, two-sided 5%) and TOST p at a margin, for user differences d."""
    n = len(d)
    dof = n - 1
    if np.abs(d).max() < 1e-5:
        # identical rankings (e.g. both select the same configuration): floating-point noise, not a difference
        return {"n_users": n, "diff": 0.0, "half": 0.0, "p": 1.0, "mde": 0.0, "tost_p": 0.0, "identical": True}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, dof))
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else 1.0
    mde = float((tcrit + stats.t.ppf(0.80, dof)) * se)
    lo = float(1 - stats.t.cdf((mean + margin) / se, dof)) if se > 0 else 0.0
    hi = float(stats.t.cdf((mean - margin) / se, dof)) if se > 0 else 0.0
    return {"n_users": n, "diff": mean, "half": tcrit * se, "p": p, "mde": mde, "tost_p": max(lo, hi)}


def nadeau_bengio_p(a_split, b_split, key, frac_val=FRAC_VAL):
    """Nadeau-Bengio corrected resampled t-test over split means (experiments/significance.py)."""
    from significance import nadeau_bengio
    ra = {"per_split": a_split}
    rb = {"per_split": b_split}
    return nadeau_bengio(ra, rb, key, frac_val)


# ---------------------------------------------------------------------------------------------
# Method registry. Each entry: display name, role, and the (result file, method name) jobs that hold it.
# Jobs of one method with several configurations (the DR-BPR-LGN family) are pooled as in final_analysis.
# ---------------------------------------------------------------------------------------------
TRAINED_BASE = ["MF", "IPS-MF", "DR-MF", "BPR-MF", "PDA", "LightGCN", "LightGCN-pt", "r-AdjNorm", "NAVIP",
                "DR-LightGCN", "DR-JL", "MRDR", "MACR", "SimGCL", "iALS"]
DEBIASERS = ["r-AdjNorm", "NAVIP", "DR-LightGCN", "SimGCL", "MACR", "AdvInfoNCE-LGN"]


def role_of(n):
    if n == "LightGCN":
        return "backbone"
    return "debiaser" if n in DEBIASERS else "baseline"


def _m(name, role, *jobs):
    return {"name": name, "role": role, "jobs": list(jobs)}


def _family_jobs(ds):
    """Final DR-BPR family (final_config.FINAL_JOBS), pooled over configurations."""
    if ds == "coat":
        return [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN"), (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p"),
                (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")]
    if ds == "yahoo":
        return [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] + \
               [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")]
    if ds == "kuairand":
        return [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")]
    return [(V4 + f"learned_kuairec_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3")] + \
           [(V4 + f"learned_kuairec_p2_lr{lr}.json", "DR-BPR-LGN-p2") for lr in ("1e-2", "3e-3", "1e-3")] + \
           [(V4 + "learned_kuairec_p_lr3e-3.json", "DR-BPR-LGN-p")]


def registry(ds):
    """Trained methods of the main table for a dataset (order = display order)."""
    R = []
    if ds == "coat":
        R += [_m(n, role_of(n), (V3 + "learned_coat_given.json", n)) for n in TRAINED_BASE]
        R += [_m("AdvInfoNCE-LGN", "debiaser", (V4 + "learned_coat_adv.json", "AdvInfoNCE-LGN")),
              _m("iALS-wide", "baseline", (V5 + "learned_coat_ialswide.json", "iALS-wide")),
              _m("iALS-p (add prior)", "prior", (V5 + "learned_coat_add_ials.json", "iALS-p")),
              _m("LightGCN-p (add prior)", "prior", (V4 + "learned_coat_prior_bpr.json", "LightGCN-p")),
              _m("DR-BPR-LGN", "refinement", (V4 + "learned_coat_drbpr.json", "DR-BPR-LGN")),
              _m("DR-BPR-LGN-1 (Q, lrQ)", "refinement", (V4 + "learned_coat_lrQ.json", "DR-BPR-LGN-1")),
              _m("DR-BPR-LGN-p", "refinement", (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p")),
              _m("DR-BPR-LGN-p2", "refinement", (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")),
              _m("DR-iALS", "refinement", (V4 + "learned_coat_driALS.json", "DR-iALS")),
              _m("QP family (pooled)", "final", *_family_jobs("coat"))]
        R += [_m("control: shuffled Q", "control", (V4 + "learned_coat_abl_DR-BPR-shuf.json", "DR-BPR-shuf")),
              _m("control: popularity Q", "control", (V4 + "learned_coat_abl_DR-BPR-pop.json", "DR-BPR-pop"))]
    elif ds == "yahoo":
        src = {"LightGCN": V3 + "learned_yahoo_pop_B.json", "SimGCL": V3 + "learned_yahoo_pop_B.json",
               "r-AdjNorm": V3 + "learned_yahoo_pop_A.json", "NAVIP": V3 + "learned_yahoo_pop_C.json",
               "LightGCN-pt": V3 + "learned_yahoo_pop_C.json", "DR-LightGCN": V3 + "learned_yahoo_pop_C.json"}
        for n in TRAINED_BASE:
            R.append(_m(n, role_of(n), (src.get(n, V3 + "learned_yahoo_pop_D.json"), n)))
        R += [_m("AdvInfoNCE-LGN", "debiaser", (V4 + "learned_yahoo_adv.json", "AdvInfoNCE-LGN")),
              _m("LightGCN-p (add prior)", "prior", (V6 + "learned_yahoo_offonly.json", "LightGCN-p")),
              _m("DR-BPR-LGN-1 (Q)", "refinement", *_family_jobs("yahoo")[:3]),
              _m("QP family (pooled)", "final", *_family_jobs("yahoo")),
              _m("control: shuffled Q", "control", (V4 + "learned_yahoo_abl_DR-BPR-shuf.json", "DR-BPR-shuf")),
              _m("control: popularity Q", "control", (V4 + "learned_yahoo_abl_DR-BPR-pop.json", "DR-BPR-pop"))]
    elif ds == "kuairand":
        R += [_m(n, role_of(n), (V3 + "learned_kuairand_pop.json", n))
              for n in ["MF", "DR-MF", "BPR-MF", "LightGCN", "LightGCN-pt", "r-AdjNorm", "NAVIP", "DR-JL", "MACR", "SimGCL", "iALS"]]
        R += [_m("AdvInfoNCE-LGN", "debiaser", (V4 + "learned_kuairand_adv.json", "AdvInfoNCE-LGN")),
              _m("LightGCN-p (add prior)", "prior", (V6 + "learned_kuairand_offonly.json", "LightGCN-p")),
              _m("QP family (pooled)", "final", *_family_jobs("kuairand"))]
    else:
        src = {"LightGCN-pt": V3 + "learned_kuairec_pop_A.json", "SimGCL": V3 + "learned_kuairec_pop_A.json",
               "DR-JL": V3 + "learned_kuairec_pop_A.json", "MRDR": V3 + "learned_kuairec_pop_A.json",
               "MACR": V3 + "learned_kuairec_pop_A.json", "iALS": V3 + "learned_kuairec_pop_A.json",
               "MF": V3 + "learned_kuairec_pop_A.json", "IPS-MF": V3 + "learned_kuairec_pop_A.json",
               "DR-MF": V3 + "learned_kuairec_pop_A.json"}
        for n in ["LightGCN", "LightGCN-pt", "r-AdjNorm", "NAVIP", "DR-LightGCN", "BPR-MF", "SimGCL", "DR-JL", "MRDR", "MACR", "iALS", "MF", "IPS-MF", "DR-MF"]:
            R.append(_m(n, role_of(n), (src.get(n, V3 + "learned_kuairec_pop_B.json"), n)))
        R += [_m("AdvInfoNCE-LGN", "debiaser", (V4 + "learned_kuairec_adv.json", "AdvInfoNCE-LGN")),
              _m("LightGCN-p (add prior)", "prior", (V4 + "learned_kuairec_offonly.json", "LightGCN-p")),
              # X1 run (8 configurations, same prior settings as the stored Coat add prior: floor 0.2, IPS fit)
              _m("iALS-p (add prior)", "prior", (V6 + "learned_kuairec_add_ials_small.json", "iALS-p-small")),
              # matched-grid additive row of the DRGP queue (45 configurations); reported as an alternative only
              _m("iALS-p (add, DRGP grid)", "alternate", (V5 + "learned_kuairec_add_ials.json", "iALS-p")),
              _m("QP family (pooled)", "final", *_family_jobs("kuairec"))]
    return R


# training-free operators: (display name, bank file, result file, method name)
def training_free(ds):
    """Operators without training. A bank exists only for some (v2 bank_filters_*); the others are
    stored-only (their per-user values support A2, not A1)."""
    names = {"coat": ["Obs", "EASE", "GF-CF", "IPS", "DR", "DRUP", "EASE-DR", "GF-CF-DR", "Pop", "Impute"],
             "yahoo": ["Obs", "DR", "DRUP", "Pop", "Impute"],
             "kuairec": ["Obs", "DRUP", "DR", "Pop", "Impute"],
             "kuairand": []}[ds]
    bank = {"coat": "bank_filters_coat.json", "yahoo": "bank_filters_yahoo.json", "kuairec": "bank_filters_kuairec.json"}
    res = {"coat": "filters_coat_given.json", "yahoo": "filters_yahoo_pop.json", "kuairec": "filters_kuairec_pop.json"}
    if ds not in res:
        return []
    return [(n, V2 + bank[ds], V2 + res[ds], n) for n in names]


# ---------------------------------------------------------------------------------------------
# A loaded method: stored results (per split, per user, chosen) and, where a bank exists, the replay.
# ---------------------------------------------------------------------------------------------
class Loaded:
    def __init__(self, ds, name, role, S, per_split, per_user, n_configs, chosen, val_best, tables, notes):
        self.ds, self.name, self.role, self.S = ds, name, role, S
        self.per_split, self.per_user, self.n_configs = per_split, per_user, n_configs
        self.chosen, self.val_best = chosen, val_best
        self.tables = tables          # list of (groups, V, T, E) per job; None if no bank
        self.notes = notes

    @property
    def key(self):
        return KEY[self.ds]

    def stored_mean(self, key=None):
        key = key or self.key
        return float(np.mean([r[key] for r in self.per_split]))

    def table(self):
        """Pooled candidate set: union of the configurations of all jobs (None if a job has no bank)."""
        if self.tables is None or any(t is None for t in self.tables):
            return None
        V = np.vstack([t[1] for t in self.tables])
        T = np.vstack([t[2] for t in self.tables])
        groups = [g for t in self.tables for g in t[0]]
        return groups, V, T


def load_method(ds, entry, rep, patience_override=None):
    """Load one registry entry. Pools several jobs per split as final_analysis.combine does (by validation).
    Returns None (with a note) when any job file is missing."""
    key = KEY[ds]
    jobs = entry["jobs"]
    results, tables, notes = [], [], []
    for res_path, mname in jobs:
        if not os.path.exists(res_path):
            rep.note(f"{entry['name']}: missing {res_path}; method skipped")
            return None
        js = load_json(res_path)
        if mname not in js.get("results", {}):
            rep.note(f"{entry['name']}: {mname} not in {res_path}; method skipped")
            return None
        r = js["results"][mname]
        if "per_user" not in r or not r["per_user"]:
            rep.note(f"{entry['name']}: no per-user values in {res_path}; method skipped")
            return None
        results.append(r)
        bk = bank_rows(res_path, mname)
        if bk is None:
            tables.append(None)
        else:
            bargs, rows = bk
            S = len(r["per_split"])
            if patience_override is not None:
                P = patience_override
            elif bargs.get("patience") is not None:
                P = bargs["patience"]
            else:
                P = None
                notes.append(f"{res_path}: bank records no patience; replayed without a stop")
            groups, V, T, E = replay_bank(rows, S, P)
            tables.append((groups, V, T, E, P))
    S = len(results[0]["per_split"])
    # per split, the job with the largest validation score supplies the selection (final_analysis.combine)
    per_split, per_user, chosen, val_best = [], [], [], []
    for s in range(S):
        j = int(np.argmax([rr["val_best"][s] for rr in results]))
        per_user.append(results[j]["per_user"][s])
        per_split.append(results[j]["per_split"][s])
        chosen.append(results[j]["chosen"][s])
        val_best.append(results[j]["val_best"][s])
    n_cfg = sum(rr.get("n_configs", 0) for rr in results)
    return Loaded(ds, entry["name"], entry["role"], S, per_split, per_user, n_cfg, chosen, val_best,
                  tables if all(t is not None for t in tables) else None, notes)


def load_training_free(ds, item, rep):
    name, bfile, rfile, mname = item
    if not os.path.exists(rfile):
        rep.note(f"{name}: missing {rfile}; skipped")
        return None
    r = load_json(rfile)["results"][mname]
    S = len(r["per_split"])
    tables = None
    if os.path.exists(bfile):
        bj = load_json(bfile)
        if mname in bj["banks"]:
            groups, V, T, E = replay_bank(bj["banks"][mname], S, None)
            tables = [(groups, V, T, E, None)]
    return Loaded(ds, name, "training-free", S, r["per_split"], r["per_user"], r.get("n_configs"),
                  r["chosen"], r["val_best"], tables, [])


# ---------------------------------------------------------------------------------------------
# All methods of a dataset (trained registry + training-free operators) and the primary comparisons.
# ---------------------------------------------------------------------------------------------
def load_all(ds, rep):
    """name -> Loaded, in display order. Missing inputs are skipped with a note."""
    M = {}
    for entry in registry(ds):
        m = load_method(ds, entry, rep)
        if m is not None:
            M[m.name] = m
    for item in training_free(ds):
        m = load_training_free(ds, item, rep)
        if m is not None:
            M[m.name] = m
    return M


def val_mean(m):
    """Mean validation score of the validation-selected configuration (replayed when a bank exists)."""
    t = m.table()
    if t is None:
        return float(np.mean(m.val_best))
    _, V, _ = t
    return float(select(V, np.zeros_like(V))[1].mean())


def primary_pairs(ds, M):
    """Comparisons of DESIGN.md Section 2.3 that exist for this dataset: (family, A, B).
    P1 uses the third-party debiaser with the highest mean validation score (not test); P4 uses the trained
    model with the highest mean validation score. A comparison whose method is missing is dropped by the caller."""
    pairs = []
    deb = [m for m in M.values() if m.role == "debiaser"]
    if deb:
        best = max(deb, key=val_mean)
        pairs.append(("P1", best.name, "LightGCN"))
    if ds == "coat":
        pairs += [("P2", "iALS-p (add prior)", "iALS"), ("P2", "LightGCN-p (add prior)", "LightGCN"),
                  ("P3", "QP family (pooled)", "LightGCN-p (add prior)"), ("P3", "DR-iALS", "iALS-p (add prior)")]
    if ds in ("yahoo", "kuairand", "kuairec"):
        pairs += [("P2", "LightGCN-p (add prior)", "LightGCN")]
    if ds == "kuairec":
        pairs += [("P2", "iALS-p (add prior)", "iALS"), ("P3", "QP family (pooled)", "LightGCN-p (add prior)")]
        trained = [m for m in M.values() if m.role in ("backbone", "baseline", "debiaser")]
        if trained and "DRUP" in M:
            pairs.append(("P4", "DRUP", max(trained, key=val_mean).name))
    if ds in ("coat", "yahoo", "kuairec") and "DRUP" in M and "DR" in M:
        pairs.append(("W", "DRUP", "DR"))
    return pairs
