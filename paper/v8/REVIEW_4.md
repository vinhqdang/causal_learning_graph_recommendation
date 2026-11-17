# Referee report 4 on paper/v8 (blind read of main.tex, sections/*.tex, tables/*.tex and main.pdf, 60 pp.; figures 1-3 inspected)

Scope: I re-read every source file and the PDF text (built 17 Nov, newer than the sources), compared text with every table, and checked each item of REVIEW_3.md. "s5.5" means Section 5.5 of the PDF; "App. D.6" means Appendix D, subsection 6. Main text is now 32 pages (41 in v8 at REVIEW_3), appendices 27 pages.

## 1. Contribution (4 sentences)

The paper studies ranking of videos that a platform inserts into a feed uniformly at random (KuaiRand), when training may use only impressions logged before the random period and the random log is kept for validation and testing. It proposes LEAF, per-user lift and familiarity statistics on tag/author/music keys, computed from each user's out-of-pool history and fed as six channels into LightGCN (LEAF-LGN), and DAX, a click model (click = examination(user, scenario) x attraction(item)) fitted by approximate EM with tag shrinkage and added to any backbone through a validation-tuned weight. On KuaiRand-27K, LEAF-LGN+DAX reaches nDCG@10 0.4854, against 0.4634 for the authors' unpublished propensity-weighted LightGCN, and is above 13 comparators at split level after Holm correction (the margins over that baseline and over DIN+DAX are significant at adjusted p 0.0456, and are replicated under extra training seeds and matched-capacity DIN/SIM). DAX adds about +10 to +13 (x1e-3) to six backbones, switching the examination factors off costs +8.80, permuting attractiveness across items removes the gain, shuffling users' lifelong tables removes the LEAF gain, and DAX alone is no better than popularity.

## 2. Recommendation: MAJOR REVISION (light; one round of additional controls and a rescoping of the title claim should be enough)

The paper has changed a great deal since REVIEW_3 and most of what I asked for in the text has been done: one stated significance rule (s4.4), an abstract that says "unpublished", "narrow" and "five of six", tables inside the text, a family list, matched-capacity DIN/SIM, training-seed replicates of the three comparators that matter (PW-LightGCN, PW-LightGCN+DAX, DIN+DAX), an examination-off control and stand-alone item scores (s5.6), a shortened and corrected headroom section (s5.1), removal of the unreported KuaiRec experiment, a forest plot with all 13 comparators, a baseline table, and a 32-page main text. The protocol (random log only for selection and test, cut-off, leakage checks) remains the strongest part and is convincingly described.

What stops me from recommending acceptance:

1. The title claim ("examination before attraction") is only partly isolated. Examination-off costs +8.80 (adjusted 0.025), but the two component ablations that belong to examination (scenario factor +1.51, lifelong prior +1.78) and the additive-logit alternative (+6.09) are not significant at split level (adjusted 0.228, 0.263, 0.088; Table `tab:ablate`), the examination factors are confounded with user activity by construction, and no examination-free post-hoc item score of equal simplicity (popularity, raw item CTR, user-activity-normalised CTR) has been run through the same combination step, although DAX alone equals popularity (0.4206 vs 0.4194, s5.6).
2. Baseline credibility. The four published debiasing estimators (0.3765-0.4579) sit below, or barely above, plain popularity (0.4194, Table `tab:exam`), which indicates mis-specification or under-tuning rather than evidence. The "matched-capacity" DIN/SIM runs are worse than the d=16 runs and tune only two configurations at a single learning rate. No random-order reference is reported anywhere, so the size of 0.4854 vs 0.4796 cannot be judged.
3. One dataset, one cut-off, one scenario, one metric (nDCG@10 on users with a click), and the headline reference is an unpublished baseline of the authors. For TORS this is not enough; for a journal that accepts a careful single-platform case study it is acceptable.
4. Inference is still resampling-based on one training run per split family (Nadeau-Bengio with 5 splits), with a test that cannot even detect the removal of the entire DAX gain (permuted-across-items minus full: +15.36, adjusted 0.088, while the full gain itself is 0.037).

## 3. Status of every REVIEW_3 item

R = resolved, P = partly resolved, O = open.

### 3a. REVIEW_3 headline reasons (section 2 of REVIEW_3)

| Item | Status | Evidence in the current version |
|---|---|---|
| Reason 1: "examination before attraction" not isolated; need theta=kappa=1 control, theta-only/kappa-only, activity-normalised CTR | P | Control (a) added: s5.6, Table `tab:exam` (+8.80, adj. 0.025; control still +6.30 over LEAF-LGN, adj. 0.045). Missing: theta-only, kappa-only (only "without kappa" and "without lifelong prior" exist, both n.s.), user-activity-normalised CTR, any independent attention signal. See M1, M2. |
| Reason 2: abstract/intro use unpublished baseline; closest comparators DIN+DAX / QP+DAX; matched DIN/SIM | P | Abstract now says "unpublished" and names DIN+DAX; matched-capacity runs and seeds added (s5.7, s5.10). Headline number is still 0.4634 against the unpublished baseline; "matched capacity" is not matched tuning (M3). Published estimators unchanged (M3). |
| Reason 3: fragile inference (5 splits, one training run, post-hoc families, no baseline seeds) | P | One stated rule (s4.4); families listed; seeds for PW-LightGCN, PW+DAX, DIN+DAX (two each). No user-cluster bootstrap; NB still approximate; pre-specification unverifiable (M4). |
| Reason 4: LEAF half validated only partially; headroom narrative unsupported | P/R | Headroom section rebuilt and the "item-level" conclusion dropped (R). LEAF controls still compare 197 vs 12 configurations, not on LEAF-LGN+DAX, no raw-count channel (P, see N4 below). |

### 3b. REVIEW_2 items as re-graded in REVIEW_3 (now re-graded again)

| # | Item | Was | Now | Evidence |
|---|---|---|---|---|
| 2a | Who needs ranking of random inserts | P | P | Intro l.4-8 unchanged: one sentence (explore, promote, change recommender). No stakeholder or cost argument. |
| 2b | "Random insertion = direct test" argued | P | P | s1 and s4.1 argue leakage, not why a click on a random insert in scenario 1 measures appeal. |
| 2c | Gap partly constructed by protocol | O | P | "We found no method" is now "We are not aware"; but task and gap are still self-defined, and no baseline that uses a small uniform sample (AutoDebias is discussed in s2 but not run) shows what the forbidden data would buy. |
| 3.1 | Scenarios, click field, "almost entirely" | P | P | Still no count or names of scenarios, no source field of "click", no share of random inserts in scenario 1 (s3.1). |
| 3.2, 3.3, 3.4, 3.6, 3.7 | Eq. 2/4 std, readout, Eq. 13, group variant, scale | R | R | Re-checked; the nu_g selected-vs-fixed inconsistency is fixed (s3.4: "fixed strength, not tuned"). |
| 3.5 | EM approximation, convergence, init | P | P | Disclosed (s3.4, App. B), still 40 fixed iterations, init 0.7/0.6, no convergence test, monotonicity only on simulated data for 25 iterations (App. D.6). New: the stated justification is contradicted by the data, see m1. |
| 3.8 | QP, DR-BPR, r-AdjNorm, DIN/SIM sizes not in main text | P | R | Renamed PW-LightGCN, described in App. A; exponent and DIN/SIM sizes now in s4.5. |
| 4.1 | "QP family" vocabulary | P | R | Renamed and defined. Remaining problem is substantive (unpublished headline baseline), tracked under 8.1. |
| 4.2 | "LEAF development experiment", "LEAF-tied", "independent tuning" | O | P | Gone from the main text. "development comparison" remains in s4.5 last sentence, App. D.7 (closed-form scorer) and the baseline table (iALS "development comparison only", never otherwise used). |
| 4.3 | Policy channel, "kept %", family names | R | R | Families listed in s4.4 (six bullets). |
| 4.4 | Code on publication; byline; footer | P | O | Still "code will be made available upon publication", "Anonymous", footer "Preprint submitted to Elsevier" with a date. |
| 5.1 | Abstract reference and caveats | P | P | Caveats added; reference still the unpublished baseline; "narrow" is ambiguous (m2). |
| 5.2 | Intro findings overloaded | R | R | "Main findings" is short, but now repeats the abstract nearly verbatim (m3). |
| 5.3 | Headroom listed as contribution | O | R | Contribution 1 is now the task formulation; Conclusion no longer opens with headroom. Contribution 1 is a definition rather than a contribution, see m4. |
| 6.1 | Main table far from text | R | R | |
| 6.2 | Figure issues | P | P | `fig_forest` now has all 13 rows and a common axis (R). `fig_seeds` left panel still has a truncated y axis (0.4835-0.4870) that magnifies 1e-3 noise. |
| 6.3 | User counts | R | R | 22,661 vs 22,612 reconciled in App. E (49 users, consistent with 0.3^5 x 22,661); 19,474 count removed with the old headroom text. |
| 6.4 | LightGCN 0.4576 vs 0.4597 | P | R | Both bases and both p values (0.005/0.182) now in text and Table `tab:backbone`. |
| 6.5 | Several LEAF-LGN baselines | P | P | Four values remain in the main text: 0.4736, 0.4728, 0.4706 (ablation base), 0.4735 (headroom). Explained, but still too many versions of one model (m5). |
| 6.6 | Mean vs paired difference | P | R | Explained once in s4.4 ("Two quantities") and in every table note. |
| 6.7 | Reduced grid reproduces main | R | R | |
| 6.8 | Seed table mixes full and reduced | P | P | Table `tab:seeds` "Spread" row still averages seed 0 (full search) with seeds 1-3 (reduced search): 0.4853 +/- 0.0006, +13.65 +/- 1.48. |
| 6.9 | "NB p 0.000" | O | R | Now "<0.001". |
| 6.11 | Headline rests on two 0.0456 | P | R | Replicated under seeds and matched capacity (s5.10); residual caveat is M4. |
| 6.12 | Per-user p 1e-125 etc. | P | P | Out of Table `tab:main`; still in Table `tab:sens`, Table `tab:leafctl-lgn` (2.8e-39, 4.0e-32) and App. D.8. |
| 6.13 | Unequal budgets | O | P | Matched-capacity DIN/SIM added (2 configs plain, 640 with DAX); budgets still unequal (3,840 / 3,200 / 1,280 / 20) and disclosed (s4.3, s7). |
| 6.14 | LEAF controls only on LEAF-tied | P | P | Holm added (3 controls); reference (0.4728, 197 configs) still not the main model (0.4736), variants 12/12/4 configs, not run on LEAF-LGN+DAX (N4). |
| 6.15 | DAX never a stand-alone ranker | O | R | DAX alone 0.4206, tag CTR 0.3720, popularity 0.4194 (s5.6). The result (DAX = popularity) raises new questions, see M1. |
| 6.16 | Oracle uses a non-method blend | O | R | s5.1 uses the method (0.4854, about 36% of the gap). |
| 6.17 | "item-level" conclusion | P | R | Conclusion and s6 now hedge ("an interpretation"). |
| 7.1 | Novelty = combination, one dataset | P | P | s2 concedes "combination"; still one platform (M5). |
| 7.2 | Related work gaps (off-policy, MNAR/DR, cold start, TWIN) | P | P | DR estimators and TWIN discussed; still nothing on counterfactual/off-policy evaluation, learning from small randomised samples as baselines (CausE-type), user-dependent click models, cold-start content models. `TODO_citations.md` still lists about 17 sources "read from titles" (m7). |
| 7.3 | Ecological analogy; grid edges | P | P | Extended grid done (Table `tab:extgrid`, +0.55, p 0.367; R for edges). Analogy still in title, Section 1, s3.4. |
| 7.4 | Title | O | O | Unchanged, 21 words, and now over-claims relative to M1. |
| 8.1 | Replace QP as headline baseline | P | P | Abstract/intro/conclusion still headline PW-LightGCN. |
| 8.2-8.7 | Rewrite abstract/intro, move tables, timeline figure, DAX variants in s3.4, Eq. fixes, one baseline paragraph | R | R | |
| 8.8 | Drop tiny p, baseline seeds | P | R/P | Seeds added for three comparators; DIN, SIM, SIM+DAX, r-AdjNorm, LightGCN, EASE and the four estimators single-seed (disclosed). |
| 8.9 | LEAF shuffle on LEAF-LGN; LEAF vs LightGCN with plain features; DAX stand-alone and alpha table | P | P | Shuffle and stand-alone done; no LightGCN-with-six-channels-as-plain-features run; no raw-count vs lift comparison at graph level. |
| 8.10 | Rework old s5.1 | R/P | R | |

REVIEW_1 residuals named in REVIEW_3: abstract silence on borderline results R; ablation base/NB p R; LEAF-LGN vs LightGCN/r-AdjNorm borderline values (0.049/0.047) still not reported (Table `tab:main` only reports LEAF-LGN+DAX vs them) O; length P (main 32 pp; 18 per-split grid tables, about 10 pages, remain); "the paper does not claim" repetition R; provenance pointers O.

### 3c. REVIEW_3 new issues N1-N8

| # | Issue | Status | Evidence |
|---|---|---|---|
| N1 | Inconsistent significance language; abstract hides borderline cases | R | s4.4 states one rule (0.04-0.05 = "significant but narrow"); abstract says "narrow", "five of six backbones", "unpublished". Residual wording: m2. |
| N1 | DAX on QP unsupported at split level; "improves every backbone" | R | Abstract: "per-user on all six, split level on five". |
| N1 | Strongest-comparator statement | P | DIN+DAX +5.73 [1.64] is in s5.2, not in the abstract. |
| N2 | Mechanism not isolated (controls a, b, c; "not simply popularity" non sequitur; user groups on in all splits) | P | (a) done; (b), (c) open; non sequitur replaced by a proper stand-alone comparison (R); user-group switch still not described as part of DAX (s3.4 says "two switches ... selected in some splits", yet groups are on in all five splits, App. C), and there is no "no user groups" ablation (M2). |
| N3 | theta_u = examination unvalidated; non-identifiability; no initialisation check | P | s3.6 now says theta/kappa "should not be read as user traits"; s7 admits confounding with user activity. No independent attention signal, no init/iteration robustness. |
| N4 | LEAF not isolated: 197 vs 12 configs, reference not main model, not on +DAX; familiarity vs lift; unused policy channel | P | Holm added; policy channel now honestly stated as part of the headline backbone (w = 0.25/0.5 in every split of LEAF-LGN+DAX, s3.3) while its control (-0.55, s5.8) shows no benefit and was never run on LEAF-LGN+DAX; the headline DAX increment +11.96 therefore also contains a change of w (acknowledged, s4.5). Raw click-count vs exposure-conditional lift still never compared at graph level (App. D.8 closed-form: lift-only 0.4634 vs familiarity-only 0.4585 is suggestive only). |
| N4 | Contribution 2 vs split-level evidence (LEAF-LGN vs PW +9.63, p 0.063) | R | Abstract and contributions no longer claim a LEAF-LGN-alone gain; s5.3 states it. |
| N5 | No trivial references; weak published estimators; DIN/SIM small; baselines tuned by authors | P | Popularity and tag CTR appear only inside Table `tab:exam`; no random order; estimators unchanged and below popularity; matched DIN/SIM added but poorly tuned (M3). |
| N6 | Inference: shared training run; seeds; families after the fact; ablation Holm | P | Ablation and control tables now report Holm values (R); the "nine-member family" sentence is gone (R); seeds partly (P); cluster bootstrap and verifiable pre-specification O (M4). |
| N7 | Headroom diagnosis | R | s5.1 is now a short diagnostic, uses the method, says the remaining headroom is also at user level. Minor: it still sits first in Results although it uses test labels (m6). |
| N8 | Generality; unreported KuaiRec experiment; test-population conditioning | P | KuaiRec number removed (R); conditioning on users with a click is stated (s3.1) but its effect is not discussed; one platform remains (M5). |

### 3d. REVIEW_3 minor items

| Item | Status |
|---|---|
| Abstract jargon ("content keys", "pool", "backbone", unexpanded LEAF/DAX/DIN) | P (m2) |
| "Debiased (tuned and published)" wording | R |
| "As far as our searches went" / titles-only sources | P (wording softened to "not aware"; sources unverified, m7) |
| IPS-MF is not a "debiased graph recommender" | R |
| Contribution 1 diagnostic; contribution 4 methodology | R / P |
| Multiple-testing sentences in Introduction | P (shorter, still in "Evaluation design") |
| Symbol overload (r, c, w, C_u, g, strata/groups) | O (m8) |
| lambda=0 pseudo-count in Eq. `m-theta` | P (only in App. B) |
| nu_g selected vs fixed | R |
| "main experiment" referenced before it exists | R |
| Policy channel sentence in method | R (re-written, though see N4) |
| Simulation thresholds read as unit tests (App. D.6) | O |
| Self-containedness jargon list (QP family, LEAF-tied, development experiment, independent tuning) | R except "development comparison" (P) |
| App. D.5 as a software log | P (now D.6, condensed; still "changing the labels ... changes nothing else", "99.99% of joined events", "two CPU threads") |
| `app:fh` / Fay-Herriot | R |
| Data/code statement, byline, footer | O |
| Meta-sentence "never cited or called our previous work" | R |
| Number-heavy prose in 4.5, 5.x, 7 | P (new number-heavy text in s5.7 and s5.10, m3) |
| Mean vs paired differences unreconciled | R |
| Main/sensitivity Holm differences | R |
| "Holm 0.023 in family of five" with no table | O |
| Forest plot shows 9 of 13 | R |
| "3840", "1.60e-1", "costs +1.51" sign | O (Table `tab:1k`; s5.5) |
| Policy-row p in unfavourable direction | R |
| "Is the result specific to one seed? No" over-strong | R (now "limited") |
| "DAX or budget? No" over-strong | R (now "accounts for part") |
| "Complementarity" untested | R in effect (backbone gains + stand-alone result) |
| Speculation stated as fact in s6/s8 | R |
| Length | P |

### 3e. REVIEW_3 "three changes"

1. Isolate the mechanism: P (examination-off and stand-alone rows added; popularity/CTR combination, theta-only/kappa-only, activity-normalised CTR, attention signal not done).
2. Robust, honest headline comparison: P (seeds, matched capacity, one rule: done; cluster bootstrap, abstract against DIN+DAX, properly tuned matched DIN/SIM: not done).
3. Isolate LEAF and prune: P (pruning and renaming done, main text 32 pp; LEAF controls not re-run on LEAF-LGN+DAX with matched budgets).

## 4. Remaining major issues

**M1. The examination story is still undetermined by the evidence, and the new stand-alone result sharpens the question.**
- s5.6: DAX alone 0.4206, popularity (pre-cut click count) 0.4194, tag-shrunk CTR 0.3720. DAX is therefore not a better stand-alone item ranker than a count, and its stand-alone advantage over the CTR it is meant to correct is +48.56, five times its +8.80 advantage in combination. Most of what examination correction buys is evidently redundant with what a backbone already has (q_i in Eq. 6 is a popularity term). The right control is not "theta=kappa=1" alone but the same post-hoc combination, with the same gamma grid and selection, for (i) pre-cut popularity, (ii) raw item CTR, (iii) a user-activity-normalised CTR (impressions weighted by 1/r_u, which is the examination-free way to use the same lifelong rate), (iv) theta-only and kappa-only DAX. Without them "+8.80 from examination" may be "+8.80 from any decent item prior that is not the backbone's learned popularity". The additive-logit model, which also adjusts for user and scenario, already captures 59.7% of the gain (Table `tab:ablate`), and the tag-shrunk CTR with no examination captures 6.30 of 15.10, so the examination-specific part is the remaining roughly 6-9, with a split-level p of 0.088 against the additive form.
- s3.4 describes user groups as one of "two switches ... selected in some splits", but App. C reports them on in all five splits, and the ablation list (s4.6, s5.5) has no "without user groups" row. The model as run estimates item attractiveness separately in six strata of the user's lifelong click rate, i.e. a user-activity-by-item table. This is a user-level collaborative signal, it uses the same quantity (r_u) as the "examination" prior, and it is kept in the examination-off control (80 = 2 x 8 x 5 configurations). Say so in the abstract and ablate it.
- Because theta_u is centred on the user's lifelong click rate (Eq. 14), examination and taste are conflated by construction (acknowledged in s7). The paper offers no signal (watch time, skips) that theta or kappa track attention. Consequence: the title and the abstract sentence "Correcting click rates for examination makes the item score useful" should be reworded as "adjusting item click rates for user activity and scenario", unless M1's controls support the stronger reading.

**M2. The abstract and several summary sentences claim more than the split-level tables support.**
- "Removing the examination model significantly lowers the gain" rests on one comparison (adjusted 0.025, search 80 vs 320 configurations) while the two components of examination that can be removed separately are not significant (adjusted 0.228, 0.263). State both in the abstract or drop the word "significantly".
- "Permuting attractiveness across items removes the gain": the gain is not distinguishable from zero (adjusted 0.754), but full minus permuted (+15.36 [1.40]) has adjusted split-level p 0.088 (Table `tab:ablate`), the same as the additive model. Absence of significance is being read as removal; the split-level test cannot detect the loss of the entire effect, which means its power is too low for the ablation conclusions, and the per-user tests (treating users as independent, s4.4) carry the narrative.
- "Shuffling lifelong statistics ... removes the gain" refers to the lifelong gain of LEAF-LGN (not of the method) measured on the separately tuned model (Table `tab:leafctl-lgn`: 197-configuration reference vs 12-configuration variant, adjusted 0.036 and 0.005).
- "The DIN margin is larger under matched capacity and those seeds": the point estimates are 5.73 (seed 0), 9.67 (d=64), 9.99 and 6.76 (seeds 1, 2). "Larger" is only a statement about the point estimates; what the seeds show is that DIN+DAX varies by about 4e-3 across seeds (0.4756 to 0.4796), comparable to the seed-0 margin, while the seed-0 value was the best of the three. That is an argument for averaging comparator seeds and reporting between-seed SD, not for "larger".

**M3. Baseline quality still limits what the comparison means.**
- IPS-MF 0.3765, DR-JL 0.3823, MRDR 0.3826 are 4-4.5 points below popularity (0.4194); NAVIP 0.4579 equals LightGCN (0.4576). A method below a count-based ranker is not a competitive debiasing baseline; it indicates a propensity from item popularity with 64 dimensions and four grid points that does not fit this task. Either tune them (propensity model, regularisation, clipping) until they at least exceed popularity, or drop the claim "four published debiased estimators" from Intro, s2 and Conclusion and describe them as sanity references.
- Matched-capacity DIN/SIM (s5.7): d=64 gives DIN+DAX 0.4759 against 0.4796 at d=16, i.e. the "larger" model is worse, with lr fixed at 1e-2 (the default grid also had 3e-3), two weight decays, 20 epochs, patience 2 (vs 40 and 3 for LEAF-LGN). "Larger capacity therefore does not close the gap" does not follow; the d=64 models are less tuned. Run d=64 with the full lr x weight-decay grid and the same epochs, or say only "we did not tune the larger models".
- No random-order reference is reported. With about 44 candidates (median 22) and nDCG@10 only for users with a click, the range 0.37 (tag CTR) to 0.51 (oracle) is probably compressed near the random-order value, so +5.7 or +22 may be a small or large fraction of what any model can gain. Report random order, popularity, DAX alone and a LightGCN-with-six-LEAF-channels-as-features row in Table `tab:main` or `tab:methods`, together with the share of users whose candidate lists are all clicked (they contribute a constant 1).
- The headline reference remains the authors' unpublished PW-LightGCN (searched over 20 configurations). The two comparators that decide the claim are authors' constructions as well (PW-LightGCN+DAX, DIN+DAX), both using the authors' DAX. A published strong baseline that does not involve DAX (SIM-soft, TWIN, a tuned DIN/SIM, or a graph model with tag features) is the honest reference; TWIN is not run although s2 says it is the Kuaishou lifelong model.

**M4. Inference.**
- Split-level inference is Nadeau-Bengio over five splits that share one training run (s4.2, s4.4); the paper says the test needs a mean/SD ratio of 4.42 and is "approximate". The seeds help for three comparators but are each tested separately, unadjusted for LEAF seeds, and use a different (reduced) search for LEAF-LGN+DAX than for seed 0. A user-cluster bootstrap that re-runs selection, or a paired analysis pooling seeds 0-2 for DIN+DAX and PW-LightGCN and seeds 0-3 for the method, would give one interval per comparison and make the 0.0456 values moot.
- "The criterion was written down before the main experiment" (s4.4) has no verifiable record; the narrative about the family being extended after the fact is honest but raises the question why a 13-comparator family is "the" test. Cite a dated artefact (commit, registered note) or state it as an analyst choice.
- The "SD over five splits" column in Table `tab:methods` is split variation only and should not be read as uncertainty of a method (LEAF-LGN SD 0.0002).
- Table `tab:seeds` "Spread over seeds" mixes the full search (seed 0) and the reduced search (seeds 1-3).

**M5. Generality and venue-level novelty.**
One platform release (the 1K data are a subset of 27K, so s5.9 is not a replication), one cut-off, one scenario (random inserts "almost entirely in scenario 1", never quantified), click label only, nDCG@10 only. The paper argues convincingly that other public randomised datasets lack scenarios or history (s4.2), but then the result is a KuaiRand case study. At least add: AUC or MRR and nDCG@5/20; results by user-history stratum (little vs much lifelong history) and by item-exposure stratum (few vs many pre-cut impressions), which is where LEAF and DAX should help if the mechanism is right; and a rescoped title and abstract ("on KuaiRand"). The method is a combination of known ingredients (s2, last paragraph), which the authors concede.

**M6. EM estimator.**
App. B states that the theta and kappa updates are maximisers under a Poisson approximation and "not the MAP"; s3.4 says the approximation "is accurate when examination probabilities are small". Pre-cut CTR in Table `tab:data` is 523,176/1,141,300 = 45.8%, so tau = theta x kappa must on average exceed 0.46 (alpha <= 1), and the initial value is 0.7; the approximation is outside its stated regime. With 40 iterations, no convergence test, the update denominators counting all rows, and monotonicity of the likelihood checked only on simulated data over 25 iterations, the fitted object is a heuristic, not the EM of Eq. 7. One check is needed: exact M-step (one-dimensional root find per user) or sensitivity to initial values, iteration count (20/40/80) and exact vs approximate update on the headline metric. `TODO_citations.md` item 7 records that the authors know the update differs from the stationary point and left the decision open.

## 5. Minor issues

**m1. Poisson justification** (see M6) and "DAX ... fitted by expectation-maximisation" in the abstract-level descriptions should say "approximate".

**m2. Abstract (198 words by PDF count, within 200 but no room to add the caveats above).**
- Undecodable for a reader who has not read the paper: "log-only signals", "test pool", "backbone", "split level", "DIN with DAX" (DIN is never expanded in the abstract, intro or s4, only once in App. D.7), "attraction", "per-user nDCG@10"; LEAF and DAX are not expanded.
- "Significant but narrow" reads as small effect size. The PW-LightGCN margin is +22.3e-3 (4.8% relative) and its "narrowness" is an adjusted p of 0.0456. Say "adjusted p just under 0.05".
- "makes the item score useful ... alone DAX does no better than popularity" is internally awkward; say what is shown ("improves a personalised backbone; no better than popularity as a stand-alone score").
- The sentence on permutation/shuffling controls is an over-statement (M2).
- The abstract gives no effect size or interval for the closest comparator (+5.73 [1.64] over DIN+DAX).

**m3. Number-heavy prose and repetition.** s5.10 first paragraph has 13 numbers in five sentences; s5.7 matched-capacity paragraph has 9; s4.5 last paragraph has 7 values of LightGCN/LEAF-LGN; s7 "Unequal budgets" lists five configuration counts. Move seeds and matched-capacity values to the tables (they are there) and keep one sentence each. The Intro "Main findings" paragraph repeats the abstract nearly verbatim; cut it to one sentence and a pointer.

**m4. Contributions.** Bullet 1 (task formulation) is a definition of the protocol, and bullet 4 (evaluation design) is method for evaluation; neither is a new result. Fold them into one "evaluation protocol" bullet, and state the empirical findings (M1) as results.

**m5. Number of LEAF-LGN versions.** 0.4736 (main), 0.4728 (separate search, 197 configs), 0.4706 (ablation base), 0.4735 (headroom, first-split configuration). A reader needs one table of "which LEAF-LGN" with the experiment that uses it (the Table `tab:methods` in App. E has the first two only).

**m6. Structure.** s5.1 (headroom, uses test labels and an oracle) opens Results before the main result; move it to s6. s5.5 and s5.6 should be one "mechanism" subsection with the added controls. Sections 6 and 7 repeat s5 findings; cut.

**m7. References and claims to verify before submission.** `TODO_citations.md` still lists about 17 works whose description was taken from titles or outlines (richardson2007, craswell2008, chapelle2009, wang2016/2018, kim2022navip "title only", chen2021autodebias, others), a 2026 arXiv source (`zhang2026categorymeans`, "accepted" only in the arXiv comment), and an unverified IEEE record (`hu2025uncertainty`); Table `tab:positioning` depends on them. The statement "Debiasing methods ... A shown but unclicked item still counts against the item, whatever the attention of the viewer" (s1, "The gap") is not true of ExpoMF, which the same paragraph cites, and click models with user- or context-dependent examination exist (contextual and user-browsing variants); soften and cite.

**m8. Notation.** r is the impression index, the base rate r_u, \bar r and the r-AdjNorm exponent; c is a click count (c^k_uv), the tag centre c_g and the user centre c~_u; w is the policy weight, the ridge weights w_i (Eq. 12) and w_uj (App. A); C_u is the candidate list and C_u^life the lifelong clicks; g is the channel weight vector and g(i) the tag group; "user groups" (s3.4) vs "strata"/"scenario groups" in other places. Eq. `m-theta` has no case for lambda = 0, which is a grid value (App. B has it).

**m9. Text vs table.**
- s5.5/s5.6: "costs +1.51", "lowers the score by +8.80" mix sign conventions; write "lowers by 8.80".
- s5.10: "The DAX variant gives +13.76 [1.62] under seed 1 (0.4721 against 0.4858)" is the margin of LEAF-LGN+DAX over PW-LightGCN+DAX, not a DAX increment; reword.
- Table `tab:policy` footnote refers to "Table `tab:policy`" itself and says "(12, 12, 4 configurations)" although this table has the 197/4 pair; it is a copy of the `tab:leafctl-lgn` note.
- Table `tab:1k`: "3840", "6400", "$1.60\times10^{-1}$" vs "3,840" elsewhere; the 1K comparison with LEAF-LGN (TODO item 6) is still not reported.
- Table `tab:fair` caption "against the PW-LightGCN" (article); "Fair-baseline reading" is unclear.
- s5.2 states the family-of-five value 0.023 but no table shows the family of five.
- Table `tab:exam` Holm family mixes "examination on/off", "off vs LEAF-LGN", "DAX vs CTR" and "DAX vs popularity"; say why these four form one family.
- `fig_seeds` truncated axis; Table `tab:seeds` spread row (see M4).
- s7 "Few splits" says the PW-LightGCN and DIN+DAX seeds "reuse the same five splits"; add that LEAF-LGN+DAX seeds 1-3 use a reduced search.
- Checked and consistent: user counts, 0.3^5 x 22,661 (about 55 vs 49), 4.42 = 2.776 x sqrt(0.2 + 7/3), configuration counts (12x32x10, 10x32x10, 4x32x10, 2x32x10, 80/320), all Holm values of Tables `tab:main`, `tab:backbone`, `tab:seeds-din`, `tab:seeds-pw`, `tab:deep` against the raw p values, 36% of the headroom gap, 0.4854 - 0.4706 vs +15.10 (explained), margins vs mean differences in Tables `tab:seeds-din`, `tab:seeds-pw`.

**m10. Self-containedness.**
- "development comparison" (s4.5, App. D.7, App. D.8, iALS row): undefined; iALS appears in the baseline table with "development comparison only" and nowhere else. Delete or explain.
- "separate search", "same-run base", "reduced backbone search", "epoch window of three", "primary family of five", "closed-form LEAF scorer" appear in the main text without a one-line definition at first use.
- "DR-BPR-LGN-1/p2/p" and "r-AdjNorm" are explained only in App. A and s4.5 respectively; fine, but give DIN/SIM the one-sentence expansion in the Introduction where they are first named.
- App. D.2 and D.4 contain run-time and thread details ("two CPU threads", seconds per fit); App. D.6 describes tests of the pipeline ("changing the labels ... changes nothing else", "99.99% of the joined events", "less than 0.02%"). These are fine in a supplement but should be one paragraph of the form "we verified X by Y".
- The "closest works" table is in App. E but cited from s2; either move it to s2 or cut s2's "Difference" sentences, which duplicate it.
- Data/code: "will be made available upon publication", Anonymous byline and Elsevier footer. For review, an anonymised repository link is expected; the PW-LightGCN baseline is unpublished, so the code is the only way to verify it.

**m11. Unsupported or over-stated sentences.** "The signal that transfers to randomly exposed items is therefore the users' own lifelong affinities" (s6): shown only with the separately tuned LEAF-LGN and the closed-form scorer, not with LEAF-LGN+DAX. "Larger capacity therefore does not close the gap" (s5.7, see M3). "The default grid therefore does not limit the test score" (s7): a +0.55 [0.52] difference with p 0.367 on five splits cannot exclude a limit of the order of 1e-3. "DAX ... estimates how attractive a video is once it has been looked at" (s3 intro) is the estimand the paper says is not identified (s3.6).

## 6. The three changes that would most raise acceptance odds

1. **Close the mechanism question with five cheap runs, and let the result set the title.** Through the existing selection protocol and Holm family, add post-hoc combinations with LEAF-LGN of (a) pre-cut popularity, (b) raw item CTR, (c) user-activity-normalised CTR, (d) theta-only and kappa-only DAX, (e) DAX without user groups; add a random-order row and a LightGCN-with-six-channels-as-features row; and report the user-group switch in the abstract. If (c) or (d) reproduces most of the +8.80, retitle ("adjusting item click rates for user activity and scenario") and drop "examination" from the abstract; if not, the multiplicative claim is established and the title stands. Either outcome removes the paper's largest vulnerability.
2. **Make the headline comparison one that a sceptic cannot dismiss.** Report the abstract against DIN+DAX (+5.7 [1.6]) and PW-LightGCN+DAX (+12.0), not against the unpublished baseline; tune the four published estimators until they exceed popularity (or demote them); give d=64 DIN/SIM the same learning-rate x weight-decay grid and epochs as the d=16 models; pool comparator seeds (0-2) and method seeds (0-3) into one paired interval per comparison and add a user-cluster bootstrap; add one published lifelong baseline that does not involve DAX (SIM-soft or TWIN); release an anonymised repository.
3. **Show where the gain comes from and widen the evidence.** Report nDCG@5/20, MRR and AUC; results by lifelong-history stratum and by item pre-cut-exposure stratum; the effect of conditioning on users with a click; and, if any second randomised-exposure dataset with persistent users and history can be built, one replication. If none exists, rescope the title and abstract to KuaiRand and state it. At the same time cut the main text to about 25 pages (move s5.1 to the discussion, merge s5.5-5.6, move the closest-works table into s2, drop repeated numbers, move App. D.6 to a short supplement).

## 7. Venue fit

- **Information Processing & Management (best fit).** The paper is an evaluation-protocol-plus-model study built on click-model and exposure-bias ideas; IP&M accepts careful single-dataset empirical work with thorough controls and a combination novelty, and the manuscript is already in the elsarticle template with Elsevier footer. Realistic outcome after the three changes: minor revision. Expect a request for the second metric set, the code, and clearer positioning against unbiased learning-to-rank.
- **ACM TORS (reach, not recommended now).** TORS expects a broader contribution (new problem formulation with theory or a significant, general empirical finding, usually two or more datasets) and would read the single-platform design, the combination novelty and the unpublished headline baseline harshly. It becomes plausible only with a second randomised-exposure dataset or substantial theory for the non-identifiability, and with a re-typeset in ACM format.
- **Knowledge-Based Systems (reasonable second choice).** Accepts new models with strong empirical validation; the paper's rigor exceeds the typical KBS submission, but KBS reviewers tend to weigh the number of datasets and baselines, so M5 matters there too. Page count is not an obstacle.
- **Expert Systems with Applications (fallback).** Lowest bar on novelty and generality; the paper would likely be accepted after minor changes, but the venue under-uses the protocol and statistics that are the manuscript's real strength.

Order of preference: IP&M, then KBS, then TORS only after the extra dataset, with ESWA as the safe fallback.
