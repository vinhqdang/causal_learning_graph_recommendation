# v6 design: data with richer signals, headroom measurement, probes, and the method decision

Status: design document, 2026-11-04. New files only: `drup/data2.py`, `experiments/v9_dataset_stats.py`,
`experiments/v9_probes/*`, `results/v9_probes/*`, this file. No existing file was edited. Raw data added
under the git-ignored `data/raw/` (KuaiRand-1K and KuaiRand-27K archives, Open Bandit Dataset archive,
derived `lifelong_*.npz` tables). New scripts: `experiments/v9_build_lifelong.py`,
`experiments/v9_link_users.py`, `experiments/v9_make_cache.py`.

Number provenance:
- "stored": copied from a file in `results/` (named where it appears).
- "probe": computed for this design by a script in `experiments/v9_probes/`, output in
  `results/v9_probes/`. Same data (the stored KuaiRand setting, reproduced exactly by `drup/data2.py`),
  splits (`split_test`, by user, frac_val 0.3, seeds 0..4), metric (`drup.metrics.evaluate`, nDCG@10),
  per-split validation selection over every configuration of the grid; test read only for the selected
  configuration. Paired differences are per-user means over splits, x1e-3 with the 95% half-width in
  brackets, unadjusted p (the `compare` of `experiments/v8_probes/harness.py`).
- "diagnostic": uses random-exposure (test) labels to measure how much the benchmark can reward a method.
  Never a method, never used for selection.

---

## 0. Verdict (read this first)

1. **Data.** KuaiRand is the only public dataset found that has randomised exposure, real impressions
   and rich side information together (16 candidates checked, Section 1). It is *not* saturated in the way
   KuaiRec is: the leave-one-user-out uniform item rate (a diagnostic oracle) reaches nDCG@10 0.4889 in the
   stored 10K-user setting against 0.4613 for the best stored method (QP family) - **+28e-3 of headroom**
   (KuaiRec: +1.8e-3). But that headroom is item-level and **no log item statistic recovers it** (Section 3,
   D4); probes that re-estimate item appeal (latent factor, control function, early-life, tab-matched) all
   fail, and so do multi-behaviour inputs or heads, user features, scenario (tab) adjustment and plain
   impression-level CTR models. Item side information (author, tag, duration) helps only inside a
   combined model (+7.26e-3 [1.82] over the policy-channel model).
2. **Two signals pass their probes** (validation-selected, per-user paired):
   - **Policy channel.** Impressions without a click used as fractional positive input of an item-item
     model: EASE(click + 0.5 shown) vs EASE(click), both + popularity offset: **+12.31e-3 [2.89]**,
     p = 7e-17 (stored setting); the plain impression channel holds under the temporal protocol
     (+8.97e-3 [2.19]). This lifts a closed-form model to the level of tuned LightGCN (-1.85e-3 [3.39],
     p = 0.29) but not above the QP family (-5.78e-3 [3.44]).
   - **Lifelong channel (the new signal).** KuaiRand-1K ships each user's ~11.7K impressions on 4.4M videos
     *outside* the random pool. Exposure-conditional tag lift and author familiarity computed from that
     history add **+24.04e-3 [9.18]**, p = 3e-7 (842 users, temporal protocol: all training data before
     the random period), on top of a tuned in-pool EASE + popularity model; selected with non-zero weight
     in 5/5 splits. Under the concurrent protocol the same gain is +39.02e-3: about 40% of it is a
     feedback-loop leak of the logging policy (later impressions caused by the random exposure), which the
     temporal protocol removes.
3. **Recommended method (Section 5): LEAF - lifelong exposure-conditional affinity transfer**, a
   closed-form collective linear autoencoder that predicts in-pool clicks from three inputs - in-pool
   clicks, in-pool impressions (policy channel) and the user's lifelong exposure-conditional affinities on
   shared content keys (tag, author, music) - trained only on data before the random-exposure period and
   evaluated on the uniformly inserted videos. It is not a prior and does not re-weight negatives: it adds
   an information source that no stored method sees.
4. **Novelty is moderate** (closest: SIM, arXiv 2006.05639, and TWIN, arXiv 2302.02352, for lifelong
   behaviour; EASE side-information extensions, Jeunen et al. RecSys 2020; impression-aware
   recommendation, arXiv 2308.07857). The defensible contribution is the mechanism (exposure-conditional
   transfer across disjoint item sets via content keys, closed form) plus the first evidence on randomised
   exposures, with a leakage-safe temporal protocol, that it carries real gains.
5. **Honest probability** of a publishable method paper with real gains: **about 30%** (Section 8). The
   main risks are (i) the 27K-scale replication (pending in Section 4.4), and (ii) a reviewer-requested
   lifelong-aware deep baseline (SIM/DIN-style CTR model fed the same history), which LEAF must match.

---

## 1. Dataset survey

Requirements: (a) randomised / uniform exposure usable as an unbiased test, or off-policy data with known
propensities; (b) real impressions (shown, not clicked); (c) rich side information. "Reachable" was tested
from this sandbox on 2026-11-04 (HTTP status of a HEAD or 1 kB range request; "downloaded" = full file
fetched and opened).

| # | Dataset (reference, verified) | Size | Licence | Download from this sandbox | Unbiased test construction | Signals a method could use | Closest prior work using it |
|---|---|---|---|---|---|---|---|
| 1 | **KuaiRand-Pure** (Gao et al., CIKM 2022, arXiv 2208.08696) | 27,285 users x 7,583 videos; 1,436,609 standard impressions; 1,186,059 random impressions | CC BY-SA 4.0 (README) / CC BY 4.0 (Zenodo record) | **downloaded** (Zenodo 10439422, 47 MB; already in `data/raw`) | Each feed slot was replaced with probability ~0.37% by a video drawn uniformly from the 7,583-video pool (99.3% in tab 1); random impressions are the test, never ranked if the pair was impressed in the standard log | impressions with no click; 12 feedback signals (click, like, follow, comment, forward, hate, long view, play time, profile enter, ...); 15 scenarios (tab); ms timestamps; 30 user features; video author / music / tag / duration / type; month-aggregate platform statistics (cover the test period: leak, excluded) | Our v2-v5 work (click only, 10K users); Hu et al. TKDE 2025 (arXiv 2303.12973, propensity calibration, AUC); Zhao et al. RecSys 2023 D2Co (arXiv 2308.08120, watch time); Wang et al. WWW 2025 (arXiv 2409.04810, evaluation on random exposure); Zhang et al. WSDM 2027 (arXiv 2611.05006, category-mean audit with random insertions) |
| 2 | **KuaiRand-1K** (same paper) | 1,000 users (993 in both logs); 11,709,994 standard impressions on 4.37M videos (11,741 per user outside the pool); 43,028 random | as 1 | **downloaded** (1.14 GB tar.gz, 7 min; logs extracted; 3.4 GB statistics file not extracted) | same random insertion; user ids, video ids and timestamps coincide with Pure (`v9_link_users.py`: 1,000 of 1,000 users one-to-one with identical ids) | everything of 1 plus long out-of-pool histories (lifelong behaviour on other videos of the same authors / tags / music) | none found that uses the out-of-pool history for the random test |
| 3 | **KuaiRand-27K** (same paper) | 27,285 users, 32M videos, ~322M standard impressions (four CSV parts, 24.5 GB uncompressed); random log identical to Pure | as 1 | **downloaded** (Zenodo, 9.9 GB, 31 min at ~5.5 MB/s; one full decompression pass of the archive takes 5 min); used by streaming, never extracted (disk) | identical to Pure; user and video ids are the Pure ids (verified on 1K: 43,030 / 43,028 random events joined on (time, video), 1,000 users one-to-one, identical ids; 27K check in `results/v9_probes/link_users.txt`) | full lifelong histories of every Pure user; 2.8 GB video features for 32M videos | none found that uses the out-of-pool history for the random test |
| 4 | KuaiRec 2.0 (Gao et al., CIKM 2022, arXiv 2202.10842) | 7,176 x 10,728 log (12.5M); fully observed 1,411 x 3,327 | CC BY-SA 4.0 | on disk | fully observed small matrix | watch ratio, captions / categories, daily item features, user features, social network | v2-v5 here; v5 H1/H2 show it is saturated by item effects (best methods within 2-3e-3 of the item oracle) |
| 5 | Open Bandit Dataset (Saito et al., NeurIPS 2021 D&B, arXiv 2008.07146) | 80 items, 3 positions; random policy ~1.37M rounds (all campaign), Bernoulli-TS ~12.2M rounds | CC BY 4.0 | **downloaded** (research.zozo.com, 413 MB zip) | uniform-random logging policy with known propensities: unbiased off-policy evaluation of any new policy | context (hashed user features, user-item affinity), item features, position; click only | the OBP benchmark itself; OPE/OPL papers. **No persistent user ids**: not a user-item CF benchmark |
| 6 | Coat (Schnabel et al., ICML 2016) | 290 x 300 | research use | on disk (features not in our copy) | 16 random ratings per user | ratings, small user/item features | saturated / underpowered (v4/v5) |
| 7 | Yahoo! R3 | 15,400 x 1,000 | Webscope agreement | on disk | 10 random songs per test user | ratings only | v2-v5 here |
| 8 | Yahoo! R6A/B (news, uniform random) | - | Webscope agreement | **not reachable** (webscope.sandbox.yahoo.com: no response) | uniform random article display | article / user features | Li et al. 2010-2011 (bandits) |
| 9 | KuaiSAR (Sun et al., CIKM 2023 resource, arXiv 2306.07705) | ~25K users, search + recommendation | non-commercial ("other-nc") | reachable (Zenodo 8181109, 488 MB) | **none** (no random exposure) | search queries, rec behaviours | - |
| 10 | Tenrec (Yuan et al., NeurIPS 2022 D&B, arXiv 2210.10629) | 5M users, 140M interactions | research | reachable (static.qblv.qq.com) | **none** | multi-behaviour, negatives | - |
| 11 | MIND (Wu et al., ACL 2020) | 1M users, 160K news, impression lists | MS research licence | **blocked** (Azure blob 409 "public access not permitted") | **none** (impressions logged under the production policy) | impressions with clicks, title / abstract text | - |
| 12 | Yambda-5B (Ploshkin et al., arXiv 2505.22238) | 1M users, 9.4M tracks, 4.79B events | Apache-2.0 | reachable (Hugging Face) | **none**: an `is_organic` flag separates organic from recommended events, but organic exposure is self-selected, not uniform | listens, likes, dislikes, timestamps, audio embeddings | - |
| 13 | ContentWise Impressions (Pérez Maurera et al., CIKM 2020, arXiv 2008.01212) | 42K users | CC BY-NC-SA 4.0 | **blocked** (github.com 403 from the proxy) | none | impressions, series/episode metadata | - |
| 14 | FINN.no Slates (Eide et al., RecSys 2021 workshop, arXiv 2111.03340) | 2.3M users, slates | CC BY 4.0 | github blocked; not tried elsewhere | none | full slates shown, search vs rec | - |
| 15 | MicroLens / PixelRec (content-driven, raw media) | up to 1M users | research | Hugging Face 401 (gated) | none | images, text, video | - |
| 16 | Spotify MSSD (shuffle sessions) | 160M sessions | non-commercial, login | not downloadable (AIcrowd login) | shuffle-mode order is random within a context, not item exposure | skips, track audio features | - |

Search for 2025-2026 releases with random exposure and rich signals (arXiv API search `abs:KuaiRand`, web
search): no new public dataset with uniform exposure was found; 2025-2026 Kuaishou releases found
(KuaiLive, KuaiComt, RecFlow) do not document random exposure. KuaiRand remains the only public dataset that
satisfies (a), (b) and (c) together.

**Ranking.** 1. KuaiRand (Pure-scale test, all signals) - the only dataset with all three properties and
power (8,347 evaluated users in the stored 10K-user setting, 22,847 with all users). 2. KuaiRand-1K -
same test protocol, plus long out-of-pool histories; low power (842 evaluated users). 3. KuaiRec - has
features but is saturated (v5 H1/H2); only a non-inferiority dataset. 4. Open Bandit Dataset - the right
OPE testbed but no user ids and 80 items; it only fits a contextual-bandit method, not this line of work.
Everything else fails (a).

**Chosen.** Primary: **KuaiRand (Pure)**, loader `drup/data2.py`, stored 10K-user setting for paired
comparison with every stored KuaiRand row, all-user setting (`n_users=None`) for the paper. Secondary:
**KuaiRand-1K** (out-of-pool histories; linked to the same random pool) and **KuaiRec** (non-inferiority
with its side features). KuaiRand-27K is the scale-up if the 1K probe shows that out-of-pool history
carries signal.

---

## 2. Loader and statistics

`drup/data2.py::load_kuairand_rich(version="pure"|"1k", n_users=10000, seed=0, label="is_click",
train_until=None, extra_users=False, include_platform_stats=False)` returns the `drup/data.py` dict
(`O`, `Y`, `P_given=None`, `test`) plus `signals` (dense per-pair matrices of the 8 binary feedback
signals, mean play ratio and impression count), `log` / `test_log` (impression-level DataFrames with
timestamps and tab), `user_feat` (standardised numeric + one-hot user features), `item_feat` (author,
music, first tag, duration, type, upload type), `aux` (standard log of the users outside the sample;
training only), `outpool` (1K only) and `meta["leak"]`. With the default arguments, `O`, `Y` and `test` are
**identical** to `drup.data.load_kuairand` (checked by `experiments/v9_dataset_stats.py --check_v1`:
True), so every probe is paired with the stored KuaiRand rows.

Statistics (`results/v9_probes/stats_pure_10k.txt`, `stats_1k.txt`, `diag_raw_stats.py` output):

| | KuaiRand-Pure, all users | Pure, stored 10K setting | KuaiRand-1K (pool part) |
|---|---|---|---|
| users / items | 27,285 / 7,583 | 10,000 / 7,583 | 993 / 7,388 |
| training impressions (standard log) | 1,436,609 | 534,550 (511,863 distinct pairs, density 0.675%) | 51,042 in-pool + 11,658,952 out-of-pool |
| impressions per user (mean / median) | 53.1 / 39 | 53.5 / 40 | 51.4 / 38 in-pool; 11,741 out-of-pool |
| training click / long-view / like rate per impression | 0.460 / 0.332 / 0.019 | 0.456 / 0.329 / 0.019 | 0.462 / 0.337 / 0.016 |
| training tabs | tab 1 73.7%, tab 0 12.4%, tab 4 6.5%, ... | same | same |
| random (test) impressions | 1,186,059 (43.5 per user, 7,583 items) | 436,664 | 42,944 |
| test click / long-view / like rate | 0.176 / 0.085 / 0.005 | 0.175 / 0.084 / 0.004 | 0.174 / 0.084 / 0.006 |
| test tab | 99.3% tab 1 | 99.3% tab 1 | 99.1% tab 1 |
| test users evaluated (>= 1 positive) | 22,847 | 8,347 (5 splits: ~5,840 test users each) | 842 |
| candidates / positives per user | 43.5 / 7.6 | 43.6 / 7.64 | 43.2 / 7.51 |

Leak checks (all from `meta["leak"]`): ranked pairs that occur in the training log = 0; test events
(user, video, ms timestamp) that occur in the training log = 0; (user, video) pairs impressed in both logs
= 267 in the 10K setting (0.06%; excluded from ranking, as in the stored loader); the random log is never
read into training matrices; the platform statistics file is not loaded unless asked and is flagged
`..._LEAKS_TEST_PERIOD` because it aggregates the whole month, including the test weeks. The standard log
overlaps the test weeks in time (concurrent protocol, as stored); `train_until` gives the temporal variant
(train strictly before 2022-04-22), reported as a robustness row.

---

## 3. Headroom diagnostics on KuaiRand (why this dataset, and where gains can come from)

All "diagnostic" (test labels used; never a method).

- **D1 item-level headroom (`diag_headroom.py`, all 27K users, click).** Ranking each user's random
  candidates by the leave-this-user-out uniform item click rate: nDCG@10 **0.5072**. The log-fitted
  additive prior (item part): **0.3884**; random ranking 0.3329. In the stored 10K setting (LOO over the
  10K users only, `diag_oracle_plus_personal.py`): oracle **0.4889**, against the best stored method, the QP
  family, **0.4613** (stored, `results/v6/tests.txt`), LightGCN 0.4569, pop prior 0.4204. So KuaiRand has
  **+28 to +46e-3 of item-level headroom** above the best stored method - ten times KuaiRec's (v5 H1:
  +1.8e-3). This is why KuaiRand, not KuaiRec, is the method dataset. The same holds for long view
  (oracle 0.4433 vs prior 0.3247).
- **D2 personal headroom.** (i) Uniform-data low rank (v5 H2 design; 50/50 split of each user's random
  candidates): item means 0.5850; + rank 4/16/64 SVD of residuals: 0.5851 at best (+0.1e-3). (ii) Item
  oracle + a log-trained personal EASE score (weight 0.5): **0.4962** vs oracle alone 0.4889
  (**+7.3e-3**, all 5 test splits, no selection). Personal signal exists but is small relative to item
  appeal.
- **D3 temporal context (`diag_context.py`).** Ranking a user's random impressions by the user's state in
  the standard log just before the impression (cumulative click rate; minutes since last impression):
  0.3353 / 0.3203 vs random 0.3317. Click rate rises from 0.061 (random video within 1 min of a standard
  impression) to 0.184 (> 1 day), but that ranks impressions, not items; within-user it carries no
  ranking signal. Sequential state is not a source of gains on this test.
- **D4 can log statistics recover the item headroom? (`diag_item_signals.py`, `diag_item_trend.py`,
  `diag_item_early.py`; items with >= 50 random impressions).** Split-half reliability of the uniform
  item click rate (random users split in two): Spearman **0.772**. Log statistics against the uniform
  rate: clicks 0.448, long views 0.441, impressions 0.410, distinct users 0.415, platform statistics
  (month aggregates, leak) 0.25-0.39, log CTR 0.294 (tab-1 only 0.201), early-life CTR of the new video
  (first 10-100 impressions) 0.15-0.23, exposure trend 0.009, duration -0.078, play ratio -0.454. **No
  log statistic exceeds raw click counts**, and click counts reach about 58% of the attainable rank
  correlation. The item headroom is real but not visible in log item statistics.

---

## 4. Method ideas, scores and probes

### 4.1 Ideas and scores

Scores: novelty against the closest named paper (arXiv ids checked through the arXiv API on 2026-11-04);
expected gain against the strongest stored KuaiRand method (QP family 0.4613; LightGCN 0.4569); risk;
cost on 4 CPU cores. "Probe" gives the outcome of Section 4.2.

| # | Idea | Closest prior (verified) | Novelty | Expected gain | Risk | Cost | Probe outcome |
|---|---|---|---|---|---|---|---|
| I1 | Multi-behaviour input (long view, like/follow/comment/forward/profile) in the CF model | ESMM (Ma et al., SIGIR 2018, 1804.07931); DCMT (Zhu et al., ICDE 2023, 2302.06141) | L | small | high | low | **fails**: validation sets both extra weights to 0 in 5/5 splits (+0.00) |
| I2 | Policy-revealed item appeal: one latent factor over item indicators (impressions, clicks, long views, likes, users, CTR) | Deconfounded Recommender (Wang et al., RecSys 2020, 1808.06581); TIDE (Zhao et al., 2109.07946) | M | item headroom +28e-3 | high | low | **fails**: 0.4065 vs popularity 0.4204 |
| I3 | Selection-corrected (control-function / Heckman) item effects with a personalised exposure index | Heckman-rank (Ovaisi et al., WWW 2020, 2001.11358); Hernández-Lobato et al. ICML 2014 | M | item headroom | high | low | **fails**: 0.377-0.381 vs 0.384 without correction |
| I4 | Exploration-phase / scenario-matched / trend appeal for new videos (all pool videos were uploaded 2022-04-09..11) | PDA (Zhang et al., SIGIR 2021, 2105.06067); TIDE | M | item headroom | high | low | **fails** (D4): early-life CTR Spearman 0.15-0.23, tab-1 CTR 0.20, trend 0.01 vs click counts 0.45 |
| I5 | Sequential user-state model at the time of the random impression | SIM / TWIN (below); KuaiRand paper | L | ? | high | medium | **fails** (D3): no within-user ranking signal |
| I6 | Impression-level response (CTR) model with item/user side information, scenario (tab) adjustment and multi-behaviour heads, predicted at tab 1 | DCMT; ESMM; AutoInt-type CTR models | L | ? | high | medium | plain **fails** (0.4088 < popularity); user features, tab adjustment and multi-behaviour heads make it worse; **item side information passes in combination** (+7.26 [1.82] over S3, rows C2, C2') |
| I7 | Pool all users' logs for item estimates (data scale) | - | none | small | low | low | **no gain**: popularity 0.4212 vs 0.4204; EASE+pop 0.4437 vs 0.4440 |
| I8 | Author-affinity channel built from in-pool clicks | side-information CF; add-EASE / cEASE (Jeunen et al., RecSys 2020, DOI 10.1145/3383313.3418480) | L | small | medium | low | **fails**: +0.25 [0.50] |
| I9 | **Policy channel**: impressions (shown, not clicked) as a fractional positive input of a closed-form item-item model | impression-aware recommenders survey (Pérez Maurera et al., ACM TORS 2025, 2308.07857); ExpoMF (1510.07025); ranking distillation (Tang and Wang, KDD 2018, 1809.07428) | L-M | +9 to +12e-3 vs own backbone; ~0 vs LightGCN | low | low | **passes vs own backbone**: +12.31 [2.89]; temporal +8.97 [2.19] |
| I10 | **Lifelong channel**: exposure-conditional affinities on shared content keys from the users' out-of-pool history | SIM (Pi et al., CIKM 2020, 2006.05639); TWIN (Chang et al., KDD 2023, 2302.02352) | M | +20 to +40e-3 | medium | medium (27K archive, 9.9 GB) | **passes on 1K**: +24.04 [9.18] temporal, +39.02 [10.56] concurrent |
| I11 | **LEAF** = I9 + I10 in one closed-form collective autoencoder with the click target | as I9, I10, cEASE | M | +25 to +35e-3 vs LightGCN / QP | medium | low-medium | components pass; joint fit is Stage 1 |

### 4.2 Probe results (KuaiRand, nDCG@10, validation-selected; paired x1e-3 [95% half-width])

Stored references (stored setting, 10K users, 8,321 paired users): LightGCN 0.4569, r-AdjNorm 0.4590,
QP family 0.4613, iALS 0.4528, LightGCN + additive prior 0.4565, popularity prior 0.4204
(`paper/v4/tables/T4_main.tex`, `T9_prior_anatomy.tex`, `results/v6/tests.txt`).

| Row | Model (setting) | Test | vs own backbone | vs LightGCN | vs QP family | Source |
|---|---|---|---|---|---|---|
| A0 | popularity, clicks (stored) | 0.4204 | - | -37.58 [4.13] | -41.51 [4.04] | `probe_appeal.txt` |
| A1 | appeal factor over 9 item indicators | 0.4065 | -13.9 vs A0 | -51.93 [4.21] | -55.86 [4.14] | `probe_appeal.txt` |
| A2 | additive item effect + control function | 0.377-0.381 | <0 | - | - | `probe_controlfn.txt` |
| S0 | EASE(click) + popularity offset (backbone) | 0.4433 | - | -14.16 [3.38] | -18.09 [3.33] | `probe_signals.txt` |
| S1 | S0 + long view / engagement inputs | 0.4433 | +0.00 (weights 0 in 5/5 splits) | -14.16 | -18.09 | idem |
| S2 | S0 + author-affinity channel | 0.4435 | +0.25 [0.50] p=0.34 | - | - | idem |
| S3 | S0 + EASE(impressions) channel | 0.4523 | **+9.22 [2.24]** p=9e-16 | -4.94 [3.24] | -8.87 [3.28] | idem |
| S4 | EASE(impressions only) + popularity | 0.4525 | +9.24 [3.01] | - | - | idem |
| S5 | S0 + EASE(shown-not-clicked) channel | 0.4470 | +3.35 [2.11] | - | - | idem |
| S6 | **EASE(click + 0.5 shown) + popularity** | **0.4553** | **+12.31 [2.89]** p=7e-17 | -1.85 [3.39] p=0.29 | -5.78 [3.44] p=0.001 | idem |
| T0/T3 | S0 / S3 under the temporal protocol | 0.4362 / 0.4450 | **+8.97 [2.19]** p=1e-15 | (test set differs) | - | `probe_signals_until0422.txt` |
| C0 | logistic MF on impressions (CTR model), + popularity | 0.4088 | -35 vs S0 | -49.62 [4.74] | -53.55 [4.53] | `probe_ctr.txt` |
| C1 | C0 + EASE(click) + EASE(impressions) | 0.4541 | ~0 vs S3 | -2.93 [3.27] | -6.85 [3.25] | idem |
| C2 | C0 + item side information (author, tag, duration embeddings) | 0.4227 | +2.56 [3.36] vs A0 (ns) | - | - | idem |
| C2' | C2 + EASE(click) + EASE(impressions) | **0.4593** | **+7.26 [1.82]** vs S3, p=6e-15; +5.25 [1.42] vs C1 | +2.32 [3.37] p=0.18 | -1.61 [3.34] p=0.35 | idem |
| C3 | C2 + user features (and + EASE channels) | 0.4098 (0.4548) | < C2 | - | - | idem |
| C4 | C3 + scenario (tab) adjustment, scored at tab 1 (+ EASE) | 0.4074 (0.4539) | < C3 | - | - | idem |
| C5 | C4 + long-view / like heads (+ EASE) | 0.3752 (0.4519) | < C4 | - | - | idem |

KuaiRand-1K (993 users, 842 evaluated; backbone EASE fitted on all Pure users' in-pool clicks; channels
from the out-of-pool log; `probe_1k*.txt`):

| Row | Model | Concurrent protocol | Temporal protocol (all training data < 2022-04-22) |
|---|---|---|---|
| L0 | popularity | 0.4264 | 0.4235 |
| L1 | EASE(click) + popularity | 0.4549 (+26.73 [12.72] vs L0) | 0.4542 (+30.77 [10.17]) |
| L2 | L1 + lifelong tag lift | 0.4752 (**+22.82 [8.74]** vs L1, p=4e-7) | 0.4645 (+11.22 [7.58], p=0.004) |
| L3 | L1 + tag lift + author familiarity | 0.4892 (**+39.02 [10.56]**, p=9e-13) | 0.4747 (**+24.04 [9.18]**, p=3e-7) |
| L4 | popularity + lifelong channels, no in-pool CF | 0.4756 (+54.41 [13.63] vs L0) | 0.4561 (+36.39 [12.73]) |

Diagnostic D3' (no selection, fixed weight 0.5, `diag_outpool_1k.txt`): with popularity, tag lift 0.4431,
tag impression count 0.4365, author count 0.4351, author lift 0.4192 (author coverage of candidates 7%),
music 0.418; at item level the author's out-of-pool click count correlates 0.373 with the uniform pool
rate (log in-pool clicks: 0.448). Lifelong history therefore carries mainly *personal* signal.

### 4.3 What the probes establish
1. On KuaiRand, the uniform-insertion target is item-dominated, but item appeal cannot be recovered from
   in-pool log statistics; every item-appeal idea (I2-I4) fails. This closes the "better prior" route.
2. Exposure is informative in one specific form: as *input* evidence of what the platform believes about the
   user (policy channel, I9), not as negatives (v5 P1; C0 here) and not as item appeal (I2).
3. The richest new signal is the users' behaviour on videos that never enter the pool (I10). It transfers
   through content keys only, so it cannot be a leak of the test items; the temporal protocol removes the
   one leak path that exists (policy feedback after the random impression), and the gain survives.

### 4.4 Scale-up probe on the full KuaiRand-27K histories

L27-PLACEHOLDER

---

## 5. Recommended method: LEAF (lifelong exposure-conditional affinity transfer)

### 5.1 Setting and notation

Users u (all KuaiRand users), random-pool videos i in I (|I| = 7,583). Training window t < T0 =
2022-04-22 00:00 UTC+8 (the first day of random insertion; temporal protocol). Test: every random
impression (99.3% tab 1) with its click, never ranked if (u, i) was impressed in the training window.

- In-pool evidence: X_ui = 1 if u clicked i at least once before T0; N_ui = 1 if i was shown to u before
  T0 and never clicked. (Exposure = X + N.)
- Lifelong evidence: for each content-key family k in K = {tag, author, music} with values V_k and map
  kappa_k: I -> V_k (empty if the pool video has no value), n^k_uv and c^k_uv are u's impressions and clicks
  before T0 on videos **outside the pool** whose key value is v. r_u = (C_u + 1) / (N_u + 2) is u's lifelong
  click rate (C_u, N_u summed over all out-of-pool impressions).
- Two lifelong statistics per (u, k, v):
  - exposure-conditional lift  l^k_uv = (c^k_uv + s r_u) / (n^k_uv + s) - r_u   (Beta-binomial shrinkage s);
  - familiarity                 f^k_uv = log(1 + n^k_uv).
  The lift conditions on being shown (impressions without a click are in the denominator), which is the
  quantity a uniform insertion measures; familiarity carries the "known creator / known topic" effect.
  Phi_u is the sparse vector of all (l, f) over (k, v), restricted to key values that occur on pool videos.

### 5.2 Model (closed-form collective linear autoencoder)

Input Z = X + w N (policy channel: shown-not-clicked as a fractional positive, w in [0, 1]); target X.

    min_{B, W}  || X - Z B - Phi W ||_F^2 + lam ||B||_F^2 + lam_Phi ||W||_F^2     s.t.  diag(B) = 0

with B (|I| x |I|) item-item weights and W (p x |I|) lifelong-key-to-item weights. With A = [Z, Phi],
Lambda = diag(lam 1_|I|, lam_Phi 1_p), P = (A'A + Lambda)^-1 and Theta_hat = P A' X, the solution is

    Theta = Theta_hat - P[:, :|I|] diag(mu),   mu_j = Theta_hat_jj / P_jj   (j <= |I|),

which reduces to EASE when w = 0 and Phi is empty (Steck, WWW 2019). Because the target (X) differs from
the input (Z), the zero-diagonal constraint is needed to stop the trivial copy of clicked items.

Score for a uniform insertion: s_ui = (Z_u B)_i + (Phi_u W)_i + beta * pi_i, where pi_i is the
standardised log click count of i before T0 (the popularity offset that every stored KuaiRand method
already uses; beta chosen on validation, beta = 0 allowed).

Two variants:
- **LEAF-tied** (the probed form): W is tied to the key match, (Phi_u W)_i = sum_k [gamma_k l^k_{u,kappa_k(i)} +
  eta_k f^k_{u,kappa_k(i)}]; 2|K| = 6 scalars fitted by least squares on the training target after B is
  fitted (or chosen on validation as in the probes). Cost: one |I| x |I| inverse.
- **LEAF-full**: free W, so liking tag A can raise items with tag B; p = 2 sum_k |V_k restricted to pool
  values and to values with >= 20 lifelong impressions among training users| (about 4-10K), one inverse of
  size |I| + p (<= 18K; ~10 min on 4 cores, float32 4 GB peak).

Optional item-content block (from probe C2'): append the pool items' own key indicators (author, tag,
duration decile) as extra input columns, as in add-EASE (Jeunen et al. 2020); it is an ablation row, not
part of the claim.

Hyperparameters (validation): lam, lam_Phi, w, s, beta.

### 5.3 Why it should work, and why it is not a prior
- The probes show that the uniform-insertion response is not recoverable from in-pool item statistics
  (Section 4.3), so any gain must come from *new information about the user*. LEAF adds two sources: the
  platform's choices for the user (policy channel) and the user's responses to the same creators and
  topics on millions of other videos (lifelong channel). Neither is an item-level prior and neither
  re-weights negatives; both are personal.
- The lifelong statistics are exposure-conditional, so they estimate P(click | shown) - the randomised
  estimand - and not how often the policy chose to show a key (the 1K diagnostic: tag lift 0.4431 vs tag
  impression count 0.4365, both with popularity).
- Transfer is between disjoint item sets through content keys, so it cannot copy test labels; the only
  leak path (policy feedback after a random impression) is removed by the temporal protocol, and the
  probe shows the gain survives it.

### 5.4 Novelty, stated exactly
- vs **SIM** (Pi et al., CIKM 2020, arXiv 2006.05639) and **TWIN** (Chang et al., KDD 2023, arXiv
  2302.02352): they retrieve category-matched *clicked* items from lifelong sequences inside a deep CTR
  model trained and evaluated on policy logs, with item ids shared between history and target. LEAF uses
  exposure-conditional statistics (impressions without a click in the denominator) and familiarity, transfers
  from out-of-pool videos to a disjoint randomised pool through content keys only, is closed-form, and is
  evaluated on randomised insertions under a temporal protocol that we show is necessary (the concurrent
  protocol inflates the lifelong gain from +24 to +39e-3 on 1K).
- vs **add-EASE / cEASE** (Jeunen, Van Balen, Goethals, RecSys 2020): they append *item* side information
  to the item-item model; LEAF appends *user-side* behaviour on other items plus an exposure channel, with a
  target different from the input (requires the generalised zero-diagonal closed form of 5.2).
- vs **ExpoMF** (arXiv 1510.07025) and the **Deconfounded Recommender** (arXiv 1808.06581): exposure is a
  latent variable or a substitute confounder there; here observed impressions are input evidence, and the
  claim is tested on randomised exposure.
- vs the **impression-aware recommender** literature (arXiv 2308.07857): impressions as negatives or as
  features are known; the policy channel alone is not claimed as new.
- Honest summary: the model is simple; the novelty is the identification of which exposure statistics
  transfer to a uniform-insertion target (conditional lift and familiarity from out-of-pool history, and
  in-pool impressions as input) and the controlled evidence. A reviewer may call LEAF "EASE with features";
  the paper must carry it with the randomised-exposure evidence, the leak analysis and the ablations.

---

## 6. Implementation spec (for a junior engineer)

New code only; existing functions are not changed except the two one-line dataset branches named in 3.

1. **Lifelong tables** - `experiments/v9_build_lifelong.py` (exists, tested on the 1K archive: 11.7M rows
   in 114 s). Streams each standard-log member of the tar.gz (no extraction), keeps out-of-pool videos,
   aggregates (user, key value) impressions / clicks / long views for the windows "pre" (< T0) and "all",
   writes `data/raw/lifelong_27k.npz`. Expected 27K run: 322M rows, about 1 h. Unit checks in
   `tests/test_lifelong.py`: no pool video id in any table; all "pre" rows have time < T0 (re-read a 1M-row
   sample); totals equal a direct pandas aggregation on the 1K archive.
2. **User linkage** - `experiments/v9_link_users.py`: join the 27K random log to the Pure random log on
   (time_ms, video_id); require a one-to-one user mapping on >= 99.9% of rows (on 1K vs Pure: 43,075 time
   matches, 99.9% same video). Save `data/raw/user_map_27k_pure.npy`. If the ids are identical, assert it.
3. **Data cache for baselines** - `experiments/v9_make_cache.py --train_until 1650556800000 --n_users 0
   --out data/raw/kuairand_t.pt` (dict O, Y, test in the `drup/data.py` format, from
   `drup.data2.load_kuairand_rich`, float32). Add a `kuairand_t` branch next to `kuairand` in
   `experiments/run_learned.py::main` and `experiments/learned_ctx.py::prepare` (same `ks`, `key`, `by`).
4. **`drup/leaf.py`**:
   - `lifelong_features(npz, users_raw, item_feat, window="pre", s=5.0, min_imp=20) -> (Phi: torch sparse
     (m, p), cols: list, tied: dict[name -> (m, n) sparse on candidate pairs])`;
   - `collective_ease(Z, X, Phi, lam, lam_phi) -> (B, W)` implementing 5.2 (float32, Cholesky solve, chunks of
     4,096 rows for Z'Z);
   - `fit_tied(X, ZB, tied, ridge) -> gamma` (least squares of X - ZB on the tied channels over training
     users' exposed and unexposed pairs, subsampled to 5M pairs);
   - `class LEAF(variant, lam, lam_phi, w, s, beta)` with `fit(O, Y, Phi)` and `scores(rows)`.
   Unit test: with w = 0 and Phi empty, `collective_ease` equals `experiments/v9_probes/common.ease` to
   1e-5; LEAF-tied with the probe's fixed gammas reproduces `probe_27k.py` row "EASE(click+0.5shown)+pop+
   lifelong" for split 0 to 1e-4.
5. **`experiments/run_leaf.py`** - CLI like `run_learned.py` (`--dataset kuairand_t|kuairand1k|kuairec`,
   `--window pre|all`, `--variant tied|full`, `--ablate ...`, `--seeds 5`, `--out`). Per-split validation
   selection over the whole grid with the vectorised `FastSelector` of `experiments/v9_probes/common.py`;
   output JSON in the `run_learned` format (`results[method] = {per_split, per_user, val_best, chosen}`) so
   that `experiments/v7_common.py` paired tests and Holm apply unchanged.
   Grids: lam {1000, 3000, 10000}; w {0, 0.25, 0.5, 1}; beta {0, 0.25, 0.5}; s {5, 20}; tied: gamma fitted
   (ridge {0, 10}) or validation grid {0, 0.5, 1} per channel; full: lam_Phi {1e2, 1e3, 1e4}, families
   {tag}, {tag, author}, {tag, author, music}.
   Ablations (`--ablate`): `no_lifelong`, `no_policy` (w = 0), `lift_only`, `familiarity_only`,
   `shuffle_users` (rows of Phi permuted across users: the gain must vanish), `concurrent` (window "all"
   and no `train_until`: measures the leak), `label_long_view`.
6. **Lifelong-aware baselines** (`experiments/run_leaf_baselines.py`), needed because a reviewer will ask
   whether LEAF's gain is only "more data":
   - `lightgcn_l`: LightGCN on the in-pool graph plus key-value nodes; user-key edges weighted by
     max(l, 0) + f (lifelong), pool item-key edges from `item_feat`; same grid as the stored KuaiRand
     LightGCN (lr {1e-2, 3e-3}, wd {1e-4}, dim 64, 2 layers, 40 epochs, patience 3).
   - `din_l`: target-attention CTR model over each user's top-64 lifelong (key value, l, f) tokens plus the
     in-pool click history, trained with BCE on in-pool impressions before T0, scored at tab 1.
   - `lgn_plus_leafchannels`: stored-type LightGCN score + LEAF lifelong channels (weights on validation).
7. **Analysis** - `experiments/v9_tests.py`: per-user paired t-tests, Holm within the primary family
   (Section 7), TOST for non-inferiority rows, MDE, and tables.

Queue (`experiments/v9_queue.sh`, style of `experiments/queue_v5.sh`; at most 2 jobs in parallel; peak
memory about 6 GB per LEAF job, 4 GB per LightGCN job):

```
cd "$(dirname "$0")/.."
T0=1650556800000
case "$1" in
  build_lifelong) python3 experiments/v9_build_lifelong.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz --version 27k ;;
  link_users)     python3 experiments/v9_link_users.py --archive data/raw/dl_kr27k/KuaiRand-27K.tar.gz ;;
  cache_t)        python3 experiments/v9_make_cache.py --train_until $T0 --n_users 0 --out data/raw/kuairand_t.pt ;;
  probe27k)       OMP_NUM_THREADS=3 python3 experiments/v9_probes/probe_27k.py results/v9_probes/probe_27k.txt ;;
  leaf_tied)      OMP_NUM_THREADS=3 python3 experiments/run_leaf.py --dataset kuairand_t --window pre --variant tied --seeds 5 --out results/v9/leaf_tied.json ;;
  leaf_full)      OMP_NUM_THREADS=4 python3 experiments/run_leaf.py --dataset kuairand_t --window pre --variant full --seeds 5 --out results/v9/leaf_full.json ;;
  leaf_ablate)    OMP_NUM_THREADS=3 python3 experiments/run_leaf.py --dataset kuairand_t --window pre --variant tied --seeds 5 --ablate no_lifelong no_policy lift_only familiarity_only shuffle_users label_long_view --out results/v9/leaf_ablate.json ;;
  leaf_concurrent) OMP_NUM_THREADS=3 python3 experiments/run_leaf.py --dataset kuairand_all --window all --variant tied --seeds 5 --out results/v9/leaf_concurrent.json ;;
  base_t)         OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairand_t --prop pop --prior add_noips --seeds 5 --threads 2 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-5 1e-4 --dims 64 --layers 2 --seed_check 0 --methods LightGCN LightGCN-p iALS iALS-p BPR-MF r-AdjNorm --out results/v9/learned_kuairand_t.json ;;
  qp_t)           OMP_NUM_THREADS=2 python3 experiments/run_learned.py --dataset kuairand_t --prop pop --seeds 5 --threads 2 --epochs 40 --patience 3 --batch 8192 --lrs 1e-2 3e-3 --wds 1e-4 --dims 64 --layers 2 --seed_check 0 --methods DR-BPR-LGN-1 DR-BPR-LGN-p2 DR-BPR-LGN-p --out results/v9/learned_kuairand_t_qp.json ;;
  lgnL_t)         OMP_NUM_THREADS=2 python3 experiments/run_leaf_baselines.py --model lightgcn_l --dataset kuairand_t --seeds 5 --out results/v9/lgnL_t.json ;;
  dinL_t)         OMP_NUM_THREADS=2 python3 experiments/run_leaf_baselines.py --model din_l --dataset kuairand_t --seeds 5 --out results/v9/dinL_t.json ;;
  leaf_1k)        OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairand1k --window pre --variant tied --seeds 5 --out results/v9/leaf_1k.json ;;
  leaf_kuairec)   OMP_NUM_THREADS=2 python3 experiments/run_leaf.py --dataset kuairec --variant policy_only --seeds 5 --out results/v9/leaf_kuairec.json ;;
  seeds_*)        OMP_NUM_THREADS=2 python3 experiments/run_seeds.py --dataset ${1#seeds_} --seeds 1 2 3 ;;
  tests)          python3 experiments/v9_tests.py > results/v9/tests.txt ;;
  *) echo "unknown label: $1" >&2; exit 2 ;;
esac
```
(Method names in `base_t` / `qp_t` are those registered in `run_learned.METHODS`; the QP family is the
validation-selected best of DR-BPR-LGN-1 / -p2 / -p, as in `experiments/v8_probes/harness.py`.)

---

## 7. Staged pilot plan with go / no-go thresholds

All comparisons: per-user paired t-tests on the test users of 5 user-level splits (frac_val 0.3), Holm
within the primary family {LEAF - LightGCN, LEAF - QP family, LEAF - iALS-p, LEAF - LEAF(no lifelong)};
seeds 1-3 for every trained baseline in the final table. Effect sizes x1e-3 nDCG@10.

**Stage 0 - data and scale-up probe (about 3 h; mostly the archive pass).** build_lifelong, link_users,
probe27k (Section 4.4).
- GO if (a) user linkage >= 99.9% one-to-one; (b) leak checks pass (no pool video in the tables; every
  "pre" row before T0); (c) lifelong channels add **>= +10e-3 with p < 0.001** over
  EASE(click + 0.5 shown) + popularity on all Pure users under the temporal protocol.
- NO-GO otherwise: the 1K result did not scale; stop method work and report the dataset survey and
  diagnostics (Sections 1-4) as part of the evaluation study.

**Stage 1 - LEAF against re-run baselines on the temporal all-user setting (about 8 h).** cache_t,
base_t, qp_t, leaf_tied, leaf_full, leaf_ablate.
- GO if LEAF (the variant better on validation) beats **each** of LightGCN, QP family and iALS-p by
  **>= +10e-3 with p_Holm < 0.05**, and LEAF - LEAF(no lifelong) >= +8e-3 with p_Holm < 0.05, and the
  shuffled-users control returns to LEAF(no lifelong) within +-2e-3.
- If LEAF beats its own ablations but not LightGCN/QP: try `lgn_plus_leafchannels`; if that passes, the
  method becomes "LightGCN + LEAF channels" (weaker novelty; proceed only if the gain over LightGCN is
  >= +10e-3).
- NO-GO otherwise.

**Stage 2 - robustness and the lifelong-aware baselines (about 6 h).** lgnL_t, dinL_t, leaf_concurrent,
leaf_1k, leaf_kuairec, seeds.
- Required for the paper: LEAF non-inferior to the better of LightGCN-L and DIN-L at margin 3e-3 (TOST).
  If DIN-L wins clearly (> +5e-3), report LEAF as the closed-form, leakage-safe version and the lifelong
  exposure-conditional statistics as the contribution (reframe, not abandon).
- Report: concurrent vs temporal gap (expected positive, the leak); long-view label; KuaiRec (no lifelong
  data: LEAF = policy channel; non-inferiority to DRUP at margin 3e-3); KuaiRand-1K replication.

**Stage 3 - write-up**, with Sections 1 and 3 of this document as the data / headroom part.

---

## 8. Probability of success

- Stage 0 passes: **PROB0** (1K: +24.04e-3 [9.18] under the temporal protocol, chosen in 5/5 splits; the
  signal is per user, so it should not depend on the number of users, but 842 users give a wide interval
  and the 27K archive pass is untested at full size).
- Stage 1 passes given Stage 0: about 60% (the re-run LightGCN / QP family are ~5-10e-3 above the EASE
  backbone used in the probes; the lifelong information is orthogonal to what they see, so the gain should
  mostly add, but the linear fusion may lose part of it).
- Stage 2 (lifelong-aware deep baselines do not beat LEAF by more than the margin): about 60%.
- Accepted as novel at a good venue given clear gains: about 60%.
- Overall: **about 25-30%** for a method paper with real gains; about 45% for a publishable paper with
  real gains if the reframing of Stage 2 is needed. The probability that any method that uses only the
  in-pool KuaiRand log beats the QP family by a paired margin >= 10e-3 is low (< 10%): every in-pool idea
  probed here stays below it.

---

## 9. Reproducibility

- Data: `data/raw/KuaiRand-Pure/` (existing), `data/raw/KuaiRand-1K/` (extracted from
  `data/raw/dl_kr1k/KuaiRand-1K.tar.gz`, `https://zenodo.org/api/records/10439422/files/KuaiRand-1K.tar.gz/content`),
  `data/raw/dl_obd/open_bandit_dataset.zip` (`https://research.zozo.com/data_release/open_bandit_dataset.zip`).
- Loader `drup/data2.py`; statistics `python3 experiments/v9_dataset_stats.py --version pure --check_v1`
  and `--version 1k --n_users 0`.
- Probes and diagnostics: `experiments/v9_probes/` (see its `README.txt`), outputs `results/v9_probes/`.
