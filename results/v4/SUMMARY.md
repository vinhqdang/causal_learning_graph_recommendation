# Round 6 (after the IP&M rejection): what was tried and what was found

All numbers: same splits, per-split selection on validation data, uniform data only for validation and testing.

## C. Walk correction (DRUP) on real data, matched configurations
DRUP - DR at the same configuration (Coat, nDCG@5): -0.0013 (beta 0.1) to -0.0074 (beta 3); negative in 10/10 splits for beta >= 0.3. Yahoo R3: about -0.0001 to -0.0003. Semi-synthetic KuaiRec with fixed degree weights: +0.005 to +0.020 (see rank_dose.py); with imputation-based degrees: +0.002 to +0.006, reversing at large beta.

## D1. Entropy-balanced edge weights and popularity deflation (Coat, training-free)
EB 0.539, EB-DR 0.537, DR 0.546, DRUP 0.547, Obs 0.557 (nDCG@5). Deflation weight 0 is chosen in most splits. No gain.

## D2. Exposure-aware negative weighting (DR-BPR-LightGCN)
BPR-LightGCN in which a sampled negative j of user u gets weight 1 - Q_uj, Q_uj = Yhat (1 - p) / (1 - p Yhat), the probability that a pair that is not a logged positive is a positive (additive IPS-fitted imputation Yhat, propensity p, no uniform data).
- Coat (nDCG@5): LightGCN 0.5683, DR-BPR 0.5912-0.5932; +0.0134 at matched configurations (100% positive); versus iALS (0.5812) +0.012, p = 0.18.
- Yahoo R3 (nDCG@5): 0.6837 vs r-AdjNorm 0.6742 (+0.0096, Holm p = 0.0006), LightGCN 0.6708, iALS 0.6625.
- KuaiRand (nDCG@10): 0.4623 vs r-AdjNorm 0.4590 (+0.0030, Holm p = 0.0001), LightGCN 0.4569, iALS 0.4528.
- KuaiRec (nDCG@20): 0.6095, +0.0014 over LightGCN (Holm p = 0.097); below NAVIP, DR-LightGCN, iALS, DR-JL/MRDR (0.633), and the training-free DRUP (0.638).
- Pointwise variant on KuaiRec: LightGCN-pn 0.5999, DR-LightGCN-pn 0.6027; both far below DR-JL/MRDR.
- Ablation (Coat / Yahoo, lr 3e-3): imputation-only weight 0.5937 / not run (same as full); popularity-only 0.5770 / 0.6550; shuffled Q (control) 0.5689 / 0.6690. The gain needs informative Q.
- iALS with the same weights (DR-iALS, Coat only): 0.5871 vs 0.5812; not run on larger datasets (too slow).

## Caveats
Small effects (0.003-0.013 nDCG); one imputation model; no AdvInfoNCE reproduction (plain InfoNCE-LightGCN: 0.529 on Coat, DR-InfoNCE 0.553); tuning budgets differ (DR variants use fn = 1.0 only, 36-54 configurations); KuaiRec not improved.

## D3. Imputation offset (round 6, final)
Score = f_theta(u, i) + beta * z(logit Yhat_ui), where Yhat is the IPS-fitted additive imputation (fixed, standardised), f_theta a LightGCN trained with BPR whose negatives are weighted 1 - Q_ui. beta in {0, 0.5, 1, 2} is selected on validation data (per split, across jobs). Uniform data is used only for validation and testing. Full table: results/v4/final_summary.txt (experiments/final_analysis.py).
- Coat nDCG@5 0.6015 (iALS 0.5812, Holm p = 0.047; LightGCN 0.5683).
- Yahoo R3 nDCG@5 0.6866 (r-AdjNorm 0.6742, Holm p < 1e-4).
- KuaiRand nDCG@10 0.4613 (r-AdjNorm 0.4590, Holm p = 0.019); beta = 0 chosen in 4/5 splits, the offset hurts there when forced.
- KuaiRec nDCG@20 0.6285 (LightGCN 0.6085); still below DR-JL / MRDR (0.633, Holm p = 2e-4) and the training-free DRUP (0.638).
- Ablations: BPR + offset only (no negative weights) Coat 0.6020; low-rank imputation for Q and offset is worse than additive (0.5887-0.5897).
- Caveats: grids are larger than some baselines' (union of several jobs); each variant is one imputation model; no AdvInfoNCE reproduction.
