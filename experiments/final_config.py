"""Final-method job lists, propensity choice and analysis sets shared by the experiment scripts.

Imported by run_ensemble.py, run_seeds.py, ensemble_analysis.py and seeds_analysis.py.
The values are copied literally from experiments/final_analysis.py (which runs its analysis on
import and therefore cannot be imported). experiments/final_analysis.py is not modified; the
consistency of these values with its SETS is checked by a snippet, not at run time.
"""
import glob

V4, V3 = "results/v4/", "results/v3/"

PROP = {"coat": "given", "yahoo": "pop", "kuairand": "pop", "kuairec": "pop"}

# metric per dataset (lower-case key)
KEY = {"coat": "ndcg@5", "yahoo": "ndcg@5", "kuairand": "ndcg@10", "kuairec": "ndcg@20"}

# (file, method name) of the final method, per dataset
FINAL_JOBS = {
    "coat": [(V4 + "learned_coat_drbpr.json", "DR-BPR-LGN"), (V4 + "learned_coat_prior_dr.json", "DR-BPR-LGN-p"),
             (V4 + "learned_coat_p2.json", "DR-BPR-LGN-p2")],
    "yahoo": [(V4 + f"learned_yahoo_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3", "1e-3")] +
             [(V4 + "learned_yahoo_p2_lr3e-3.json", "DR-BPR-LGN-p2"), (V4 + "learned_yahoo_p_lr3e-3.json", "DR-BPR-LGN-p")],
    "kuairand": [(V4 + "learned_kuairand_drbpr.json", "DR-BPR-LGN-1"), (V4 + "learned_kuairand_p2.json", "DR-BPR-LGN-p2"),
                 (V4 + "learned_kuairand_p.json", "DR-BPR-LGN-p")],
    "kuairec": [(V4 + f"learned_kuairec_drbpr_lr{lr}.json", "DR-BPR-LGN-1") for lr in ("1e-2", "3e-3")] +
               [(V4 + f"learned_kuairec_p2_lr{lr}.json", "DR-BPR-LGN-p2") for lr in ("1e-2", "3e-3", "1e-3")] +
               [(V4 + "learned_kuairec_p_lr3e-3.json", "DR-BPR-LGN-p")],
}

# per dataset display name: (metric, baseline part files, baseline names), as SETS of final_analysis.py
# (the final jobs of each set are FINAL_JOBS[dataset] above)
SETS = {
    "Coat": ("ndcg@5",
             [V3 + "learned_coat_given.json", V4 + "learned_coat_adv.json"],
             ["LightGCN", "r-AdjNorm", "NAVIP", "DR-LightGCN", "iALS", "SimGCL", "MACR", "AdvInfoNCE-LGN"]),
    "Yahoo": ("ndcg@5",
              glob.glob(V3 + "learned_yahoo_pop*.json") + [V4 + "learned_yahoo_adv.json"],
              ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "SimGCL", "AdvInfoNCE-LGN"]),
    "KuaiRand": ("ndcg@10",
                 [V3 + "learned_kuairand_pop.json", V4 + "learned_kuairand_adv.json"],
                 ["LightGCN", "r-AdjNorm", "NAVIP", "iALS", "BPR-MF", "SimGCL", "AdvInfoNCE-LGN"]),
    "KuaiRec": ("ndcg@20",
                glob.glob(V3 + "learned_kuairec_pop*.json") + [V4 + "learned_kuairec_adv.json"],
                ["DR-JL", "MRDR", "LightGCN-pt", "DR-LightGCN", "NAVIP", "iALS", "LightGCN", "r-AdjNorm", "AdvInfoNCE-LGN"]),
}
DS_NAME = {"coat": "Coat", "yahoo": "Yahoo", "kuairand": "KuaiRand", "kuairec": "KuaiRec"}

# baselines of the seed runs: method name -> part file with its "chosen" per split
SEED_SRC = {
    "coat": {"iALS": V3 + "learned_coat_given.json", "DR-LightGCN": V3 + "learned_coat_given.json"},
    "yahoo": {"r-AdjNorm": V3 + "learned_yahoo_pop_A.json", "LightGCN": V3 + "learned_yahoo_pop_B.json"},
    "kuairand": {"r-AdjNorm": V3 + "learned_kuairand_pop.json", "LightGCN": V3 + "learned_kuairand_pop.json"},
    "kuairec": {"DR-JL": V3 + "learned_kuairec_pop_A.json", "MRDR": V3 + "learned_kuairec_pop_A.json"},
}
