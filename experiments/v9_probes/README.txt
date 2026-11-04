v9 probes (KuaiRand). Outputs go to results/v9_probes/. Design and results: paper/v6/DESIGN.md.

Raw-log diagnostics, run from the repository root
(argument 1: data/raw/KuaiRand-Pure/data, argument 2: output text file):
  diag_raw_stats.py      log statistics (argument 1 only)
  diag_headroom.py       D1/D2: leave-one-user-out item oracle, uniform-data low-rank residual (H2 design)
  diag_item_signals.py   D4: log / platform item statistics vs the uniform item click rate
  diag_item_trend.py     D4: temporal exposure dynamics
  diag_item_early.py     D4: early-life, tab-specific and user-activity-stratified item CTR
  diag_context.py        D3: within-user temporal state before a random impression

Matrix diagnostics / probes that use experiments/v8_probes/harness.py, run from the repository root
(argument 1: experiments/v8_probes, argument 2: output text file):
  diag_decomp.py, diag_oracle_plus_personal.py, probe_controlfn.py

Probes on drup/data2.py with per-split validation selection; run inside experiments/v9_probes
(argument: output file):
  probe_signals.py OUT [base mb imp imp2 aux author until]   EASE backbone + signal channels
  probe_appeal.py OUT                                        item-appeal factor (I2), all-user pooling (I7)
  probe_ctr.py OUT [--aux] [--epochs E]                      impression-level response models (I6)
  diag_outpool.py OUT                                        KuaiRand-1K out-of-pool diagnostic
  probe_1k.py OUT [--until]                                  KuaiRand-1K lifelong channels (I10)
  probe_27k.py OUT [--window pre|all]                        all Pure users + KuaiRand-27K lifelong tables
  pair.py A.json B.json                                      paired per-user difference of two probe runs
common.py holds the shared loader, EASE and a vectorised selector (FastSelector, equal to Selector).

Data builders (repository root): experiments/v9_build_lifelong.py, experiments/v9_link_users.py,
experiments/v9_make_cache.py, experiments/v9_dataset_stats.py.
