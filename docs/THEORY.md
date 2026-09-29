# DRUP: Doubly Robust, Walk-Unbiased Graph Propagation

This note gives the full method and its theory: exact unbiasedness, bias lower
bounds for existing propagation debiasers, variance and concentration bounds,
causal (exposure-invariant) item fairness, exact explanations, certified
robustness, and a differentially private release. Every statement has a
numerical check in `experiments/`.

---

## 1. Setting

- Users $u\in[m]$, items $i\in[n]$, pairs (edges) $e=(u,i)$.
- **Potential outcome** $Y_e\in\{0,1\}$: would $u$ like $i$ if it were shown.
- **Exposure** $O_e\in\{0,1\}$ with propensity $p_e=\Pr(O_e=1\mid Y)$, which may
  depend on the pair's own outcome.
- Observed log: $\{(O_e, O_eY_e)\}$, which is missing not at random (MNAR).

**Assumption A1 (conditionally independent exposure).** Conditionally on $Y$,
the $O_e$ are independent Bernoulli($p_e$) with $p_e>0$. All expectations are
over $O$ given $Y$.

**Assumption A2 (fixed nuisances).** The propensity model $\hat p$, the
imputation $\hat Y\in[0,1]^{m\times n}$ and the edge weights $C$ are fixed given
$Y$, i.e. they do not depend on $O$ (for instance, fitted on an independent
log). We take $C_{ui}=d_u^{-\alpha}d_i^{-(1-\alpha)}$ with degrees from $\hat Y$.

*Cross-fitting.* In the experiments $\hat p$ and $\hat Y$ are cross-fitted over
ten random folds of pairs: the nuisance used at a pair of fold $k$ is fitted on
the other nine folds (`drup/pipeline.py`, class `Nuisance`). The degree weights
come either from $\hat Y$ or from cross-fitted edge estimates (row and column
sums of $W$ over the other folds); the degree source is a searched and reported
hyper-parameter. This removes the dependence of every nuisance on its own pair,
which is enough for one-hop terms, but not the dependence on the other edges of
a walk: a product of several cross-fitted edge estimates can be biased although
each of them is unbiased. Cross-fitted $W$-degrees make $C$ depend on the log,
and the scale constants $c_1,c_3$ of the score are computed from the scores.
The theorems below therefore hold for nuisances from an independent log and
are an approximation for the experimental protocol;
`experiments/mc_protocol.py` measures the gap by Monte Carlo.

**Target.** A graph recommender propagates preference along walks of the
*counterfactual full-exposure* graph $Y$, not along walks of the logged graph
$O\odot Y$. Write $\tilde Y = C\odot Y$. For one hop and three hops the targets
are
$$
F^*_{ui}= a\,\tilde Y_{ui} + b\,T^*_{ui},\qquad
T^*_{ui}=\sum_{j,v}\tilde Y_{uj}\tilde Y_{vj}\tilde Y_{vi}=(\tilde Y\tilde Y^\top\tilde Y)_{ui}.
$$
$F^*$ is exactly the linear LightGCN / graph-filter score computed on the
oracle graph.

## 2. Estimator

Clip the propensities: $\bar p_e=\max(\hat p_e,\tau)$. The edge estimates are

$$
\text{IPS: } W_e=\frac{O_eY_e}{\bar p_e},\qquad
\text{DR: } W_e=\hat Y_e+\frac{O_e(Y_e-\hat Y_e)}{\bar p_e}.
$$

**The problem.** Let $\tilde W=C\odot W$. The plug-in operator
$\tilde W\tilde W^\top\tilde W$ is what you get from NAVIP-style
inverse-propensity adjacencies (Kim et al., CIKM 2022) or from a DR-imputed
adjacency. It contains walks that traverse the same edge more than once, and
$\mathbb E[W_e^k]\neq Y_e^k$ for $k\ge2$.

**The fix (idempotent walk correction).** $Y_e$ is binary, so
$Y_e^k=Y_e$. An unbiased replacement for $\tilde W_e^k$ is therefore
$C_e^kW_e$. Define $\Delta=\tilde W\odot\tilde W-C\odot C\odot W$, together
with its row sums $r_u=\sum_j\Delta_{uj}$ and column sums $\kappa_i=\sum_v\Delta_{vi}$.
The walk-corrected 3-hop estimator is
$$
\hat T=\tilde W\tilde W^\top\tilde W
-\tilde W\odot(r\mathbf 1^\top-\Delta)
-\tilde W\odot(\mathbf 1\kappa^\top-\Delta)
-(\tilde W^{\odot3}-C^{\odot3}\odot W).
$$
The three corrections cover the three ways a walk $u\!-\!j\!-\!v\!-\!i$ can repeat
an edge: $v=u$; $j=i$; and both.

**DRUP score.** $s=a\,\tilde W+b\,\hat T$, where $a,b$ are global constants,
so $b/a$ is the only mixing hyper-parameter.

**Public-operator form.** Let $G=\tilde W^\top\tilde W-\operatorname{diag}(\tilde W^\top\tilde W)+\operatorname{diag}(\mathbf 1^\top(C\odot C\odot W))$
be the item Gram matrix with its diagonal corrected. Then, for every user,
$$
\hat T_{u\cdot}=\tilde w_uG-\tilde w_u\odot(r_u\mathbf 1^\top-2\delta_u)-(\tilde w_u^{\odot3}-c_u^{\odot3}\odot w_u),
$$
with $\delta_u=\tilde w_u^{\odot 2}-c_u^{\odot2}\odot w_u$. The only shared object
is $G$; every other term uses the user's own row. See
`propagation.local_three_hop`; the relative error against the direct formula
is $4\times10^{-15}$.

**Cost.** At three hops, one $n\times n$ Gram product plus $O(mn)$ reductions,
the same cost as the uncorrected operator. There is no training. The $K$-hop
correction evaluates one contraction per pair of equality patterns (99 at
$K=5$); `experiments/bench_khop.py` measures its cost.

---

## 3. Unbiasedness and double robustness

**Theorem 1 (edge-wise double robustness).** Under A1–A2, let
$\delta_e=(p_e/\bar p_e-1)(Y_e-\hat Y_e)$. Then
$$
\mathbb E[\hat T_{ui}]=\sum_{j,v}\;\prod_{e\in\text{distinct}(u j v i)}C_e^{k_e}(Y_e+\delta_e),
$$
where $k_e$ is the number of times the walk uses $e$. In particular, if every
edge has either a correct propensity ($\bar p_e=p_e$) or a correct imputation
($\hat Y_e=Y_e$), then $\mathbb E[s]=F^*$ exactly.

*Proof.* After the correction, each walk contributes a product over *distinct*
edges of $C_e^{k_e}W_e$. By A1 the $W_e$ are independent, so the expectation
factorises. Moreover $\mathbb E[W_e]=\hat Y_e+p_e(Y_e-\hat Y_e)/\bar p_e=Y_e+\delta_e$.
The target monomial is $\prod C_e^{k_e}Y_e^{k_e}=\prod C_e^{k_e}Y_e$ because $Y_e$ is binary. ∎

Monte-Carlo check (`experiments/mc_unbiasedness.py`, 20 000 replications):

| estimator | rel. \|bias\| | frac. entries with \|z\|>3 | rel. RMSE |
|---|---|---|---|
| propagation on logged graph | 0.992 | 1.000 | 1.02 |
| IPS adjacency (NAVIP-style) | 2.403 | 0.404 | 21.0 |
| IPS + walk correction | **0.028** | **0.0025** | 4.93 |
| DR adjacency | 1.903 | 0.629 | 20.9 |
| **DRUP** (DR + walk correction) | **0.020** | **0.0008** | **3.98** |

With unbiased estimators, $|z|>3$ occurs 0.27% of the time by chance, which is
what the corrected estimators show. The correction also removes the
$1/p^2$ terms that dominate the variance, so RMSE drops about 5×.

**Remark (control-variate family).** For any $\lambda\in[0,1]$, the edge
estimate $W^\lambda_e=O_eY_e/\bar p_e+\lambda(\hat Y_e-O_e\hat Y_e/\bar p_e)$ has
$\mathbb E[W^\lambda_e]=Y_e+(p_e/\bar p_e-1)(Y_e-\lambda\hat Y_e)$. It is therefore unbiased
whenever the propensity is correct. Equivalently, $W^\lambda$ is exactly the DR
estimate built with the shrunk imputation $\lambda\hat Y$, so every DR statement
(including double robustness with respect to $\lambda\hat Y$) carries over. $\lambda=0$ gives IPS and $\lambda=1$ gives DR,
which is the only member that is also robust to propensity errors. Theorems 1–9
hold for every member, because they only use $\mathbb E W_e=Y_e$, independence
and boundedness. Its variance is
$\frac{1-p_e}{p_e}(Y_e-\lambda\hat Y_e)^2$. The best $\lambda$ therefore shrinks towards 0
when $\hat Y$ is poor, e.g. on very sparse logs where almost every
edge of the propagated graph is imputed. We select $\lambda$ on validation data.

**Corollary 1a (unexposed candidates).** Recommendations rank unexposed
candidates, where $W_{ui}=\hat Y_{ui}$ (DR) or $0$ (IPS) is deterministic. With
correct propensities,
$$
\mathbb E[s_{ui}\mid O_{ui}=0]=F^*_{ui}-(Y_{ui}-\hat Y_{ui})\,g_{ui},\qquad
g_{ui}=\partial F^*_{ui}/\partial Y_{ui}
=C_{ui}\big(a+b(\textstyle\sum_{v\ne u}C_{vi}^2Y_{vi}+\sum_{j\ne i}C_{uj}^2Y_{uj}+C_{ui}^2)\big).
$$
The residual is the imputation error at the candidate itself, which no estimator
built from the log can remove. With IPS edges the walks through $(u,i)$ vanish
on unexposed candidates, so at three hops the uncorrected IPS operator has no
repeated-walk bias there; it reappears from five hops on.

**Proposition 1b (exposure dependence within users).** If exposures are
independent across users but $|\mathrm{Cov}(O_e,O_f\mid Y)|\le\rho\,p_ep_f$ for distinct
pairs of the same user (fixed-size slates give $\rho\le1/k$), then with correct
propensities
$$
|\mathbb E\hat T_{ui}-T^*_{ui}|\le\rho\sum_{j\ne i}C_{uj}\Big(\sum_{v\ne u}C_{vj}C_{vi}\varepsilon_{vj}\varepsilon_{vi}+C_{uj}C_{ui}\varepsilon_{uj}\varepsilon_{ui}\Big).
$$
For DR edges the bias is of second order in the imputation error; for IPS
($\varepsilon=Y$) it is of first order. `experiments/mc_stress.py` checks this.

### 3.1 Any number of hops

**Theorem 1′ (exact K-hop correction).** Fix an odd $K$. A $K$-hop walk
$u=U_0\!-\!J_1\!-\!U_1\!-\cdots-\!U_r\!-\!J_{r+1}=i$ with $r=(K-1)/2$ repeats an
edge only through coincidences among its index variables. Encode the
coincidences as an equality pattern $\pi=(\pi_U,\pi_J)$, a pair of set
partitions of $\{U_0,\dots,U_r\}$ and $\{J_1,\dots,J_{r+1}\}$. The pattern fixes
the multiplicity $k_e$ of every distinct edge. Let
$F^{\text{corr}}_\pi=\prod_eC_e^{k_e}W_e$ and $F^{\text{plug}}_\pi=\prod_e\tilde W_e^{k_e}$. The estimator
$$
\hat T^{(K)}=\tilde W(\tilde W^\top\tilde W)^r+\sum_{\pi:\ \exists k_e>1}\ \sum_{\sigma\ge\pi}\mu(\pi,\sigma)\big(g^{\text{corr}}_\pi(\sigma)-g^{\text{plug}}_\pi(\sigma)\big)
$$
is exactly unbiased for $(\tilde Y\tilde Y^\top)^r\tilde Y$ under A1–A2. Here
$g_\pi(\sigma)$ is the unconstrained sum of $F_\pi$ over assignments that are
constant on the blocks of $\sigma$, computed as one tensor contraction, and
$\mu(\pi,\sigma)=\prod_{B\in\sigma}(-1)^{b_B-1}(b_B-1)!$ is the Möbius function of
the partition lattice, with $b_B$ the number of $\pi$-blocks inside $B$.

*Proof.* Möbius inversion turns the "at least $\sigma$" sums into "exactly
$\pi$" sums, so each walk is counted once, under its own pattern. For that
walk the corrected term uses $\prod C_e^{k_e}W_e$, which has mean
$\prod C_e^{k_e}Y_e=\prod\tilde Y_e^{k_e}$ by independence and $Y^k=Y$. ∎

The number of contraction terms is 5, 99 and 2,790 for $K=3,5,7$. Checks
(`drup/khop.py`): the $K=3$ case reproduces the closed form of Section 2 to
$7\times10^{-15}$; $K=5$ matches brute-force enumeration of all walks to
$10^{-13}$; the Monte-Carlo check is in `results/mc_unbiasedness_K5.json`.

## 4. Lower bound on the bias of uncorrected propagation

**Theorem 2.** Take correct propensities, $\tau\le\min p$, and a candidate
$(u,i)$ with $O_{ui}=0$. The uncorrected DR-adjacency propagation (and IPS
when $\hat Y\equiv 0$) satisfies
$$
\mathbb E[(\tilde W\tilde W^\top\tilde W)_{ui}\mid O_{ui}=0]-\mathbb E[\hat T_{ui}\mid O_{ui}=0]
= C_{ui}\hat Y_{ui}\big(R_u+K_i\big)-C_{ui}^3\hat Y_{ui}(1-\hat Y_{ui}^2),
$$
$$
R_u=\sum_{j\ne i}C_{uj}^2\,\tfrac{1-p_{uj}}{p_{uj}}(Y_{uj}-\hat Y_{uj})^2,\qquad
K_i=\sum_{v\ne u}C_{vi}^2\,\tfrac{1-p_{vi}}{p_{vi}}(Y_{vi}-\hat Y_{vi})^2 .
$$
Hence the within-user ranking bias is carried by the item term
$C_{ui}\hat Y_{ui}K_i\ge0$. It satisfies
$$
K_i\;\ge\;\Big(\tfrac1{\max_v p_{vi}}-1\Big)\sum_{v\ne u}C_{vi}^2(Y_{vi}-\hat Y_{vi})^2,
$$
which is unbounded as the item's propensities go to 0. With clipped propensities
$\bar p=\max(p,\tau)$, $\tau\le1/2$, the excess of a doubly traversed edge,
$C_e^2(\mathbb E W_e^2-\mathbb E W_e)$, is at most $C_e^2(1/\tau-1)\varepsilon_e^2$, with
equality at $p_e=\tau$ (for $p_e<\tau$ it decreases and the clipping bias of
Theorem 3 takes over), so the worst-case repeated-walk bias is $\Theta(1/\tau)$.
DRUP's bias is identically 0.

*Proof.* In the three repeated-edge cases the naive monomial is
$\tilde W_e^2\tilde W_f$ (or $\tilde W_e^3$), and
$\mathbb E[W_e^2]=Y_e+\operatorname{Var}(W_e)$ with
$\operatorname{Var}(W_e)=\frac{1-p_e}{p_e}(Y_e-\hat Y_e)^2$. The corrected estimator
replaces the square by $W_e$, whose mean is $Y_e$. Conditioning on
$O_{ui}=0$ fixes $W_{ui}=\hat Y_{ui}$. ∎

*Interpretation.* The bias inflates exactly the items whose past exposures were
rare and poorly imputed. Debiasing therefore turns into *over-correction*: a
data-dependent item bonus that grows as $1/p$. This explains the negative
exposure elasticity in Theorem 4. For IPS ($\hat Y\equiv0$) the
factor $\hat Y_{ui}=0$ hides the 3-hop bias on unexposed candidates, but the bias
reappears at 5 hops and in the diagonal of $G$ used by $G^2$-type filters.

## 5. Variance, concentration and the clipping trade-off

Let $\varepsilon_e=|Y_e-\hat Y_e|$ and $B=\max(1,1/\tau)$. For each edge $e$, let
$\Lambda_e(u,i)$ be the $C$-weighted mass of the walks from $u$ to $i$ that use $e$.

**Theorem 3.** Under A1–A2:

1. (per-edge) $\operatorname{Var}(W_e)=\frac{p_e(1-p_e)}{\bar p_e^2}\varepsilon_e^2\le\varepsilon_e^2/\tau$.
2. (Efron–Stein)
$\operatorname{Var}(\hat T_{ui})\le \dfrac{(1+1/\tau)^2}{\tau}\sum_e\varepsilon_e^2\Lambda_e(u,i)^2 .$
3. (clipping bias) $|\mathbb E\hat T_{ui}-T^*_{ui}|\le 3\bar\delta(1+\bar\delta)^2M_{ui}$, where
$\bar\delta=\max_e(1-p_e/\tau)_+\varepsilon_e$ and $M_{ui}=\sum_{j,v}C_{uj}C_{vj}C_{vi}$.
4. (McDiarmid) $\Pr(|\hat T_{ui}-\mathbb E\hat T_{ui}|\ge t)\le2\exp\!\big(-2t^2/\sum_e c_e^2\big)$,
where $c_e=\varepsilon_eB^2\Lambda_e(u,i)/\tau$. The same bound with $c_e$ for the
difference $\hat T_{ui}-\hat T_{uk}$ bounds the probability of mis-ranking two
candidates whose true gap is $t$.

*Proof sketch.* (1) is direct. (2): $\hat T$ is multilinear in independent
variables, so Efron–Stein gives
$\operatorname{Var}\le\sum_e\mathbb E[(\partial_e\hat T)^2]\operatorname{Var}(W_e)$.
Minkowski's inequality bounds $\|\partial_e\hat T\|_2$ by the walk mass
through $e$, weighted by $\prod_f\|W_f\|_2\le(1+1/\tau)$ over the other two
edges. (3) follows from Theorem 1 and
$\prod_f(Y_f+|\delta_f|)-\prod_fY_f\le(1+\bar\delta)^3-1$. (4): changing
$O_e$ moves $W_e$ by at most $\varepsilon_e/\tau$ and
$|\partial_e\hat T|\le B^2\Lambda_e$. ∎

*Consequences.* (i) The variance scales with the **imputation error**
$\varepsilon^2$, not with $Y^2$ as for IPS. This is where DR beats IPS on graphs.
(ii) The clip trades an $O(\tau^{-3})$ variance for a clipping bias that vanishes
when $\hat Y$ is accurate. (iii) Part 4 gives a finite-sample ranking guarantee.

## 6. Causal item fairness: exposure invariance

Intervene on an item's exposure mechanism with $do(p_{\cdot i}\leftarrow\pi\,p_{\cdot i})$
and define the **exposure elasticity**
$\eta_i=\partial\log\mathbb E[s_{ui}]/\partial\log\pi$.

**Theorem 4.** Under A1–A2 with correct propensities, and for $\pi p\ge\tau$:

- DRUP: $\mathbb E_\pi[s_{ui}]=F^*_{ui}$ for every $\pi$, so $\eta_i=0$.
  Two items with the same potential-outcome column get the same expected score,
  whatever their exposure history (counterfactual exposure fairness).
- Propagation on the logged graph (linear LightGCN): $\eta_i\ge1$ for the 3-hop
  term. Every walk to $i$ uses at least one edge incident to $i$, and each such
  edge contributes a factor $\pi p$ (popularity amplification).
- Uncorrected DR adjacency: $\eta_i<0$ whenever $\hat Y_{ui}K_i>0$, because
  $K_i$ is decreasing in $\pi$ (Theorem 2). The less an item was shown, the
  more it is boosted (over-correction).

*Proof.* Combine Theorems 1 and 2. For logged-graph propagation,
$\mathbb E[O_eY_e]=p_eY_e$, so each monomial is multiplied by $\pi^{\#\{\text{edges at }i\}}$. ∎

Checks:

- `experiments/mc_elasticity.py` (known propensities): logged graph $\eta=+1.78$,
  DR adjacency $\eta=-0.19$, **DRUP $\eta=+0.005$**, IPS $\eta=+0.007$.
- Real-data intervention (`run_fat.py --sections intervene`). A random half of
  the items has its logged exposures thinned with probability 1/2, a known
  $do(p\leftarrow p/2)$, and we measure the shift in the treated items' mean
  within-user rank:

  | | logged graph | IPS | DR | DRUP |
  |---|---|---|---|---|
  | KuaiRec | −0.134 | −0.029 | +0.0006 | **+0.0005 ± 0.0008** |
  | Coat | −0.058 | −0.059 | −0.003 | **+0.0006 ± 0.011** |

User side. Theorem 1 holds for every user separately. Less active users are
therefore not systematically under-scored in expectation; the remaining
accuracy gap between user groups comes from variance (Theorem 3), which is
reported as a group gap.

## 7. Transparency: exact attributions and optimal counterfactual explanations

Fix user $u$ and a candidate $i$ with $O_{ui}=0$. Let $G^{(-u)}$ be the public
operator with $u$'s own contribution
$\tilde w_u^\top\tilde w_u-\operatorname{diag}(\tilde w_u^{2})+\operatorname{diag}(c_u^2w_u)$
removed. Hold the nuisances fixed.

**Proposition 5 (the score is affine in the user's own data).**
$$
s_{ui}=\text{const}+\sum_{j\ne i}\big(c_{uj}G^{(-u)}_{ji}+c_{ui}w_{ui}c_{uj}^2\big)\,w_{uj}.
$$
So "un-logging" interaction $j$ ($w_{uj}\to\hat Y_{uj}$) changes the score by
exactly $\phi_j=\omega_j(w_{uj}-\hat Y_{uj})$ with $\omega_j=b\,a_j$ ($b$ the weight of the three-hop term), and removal effects are **additive over any
subset**. The $\phi_j$ are also the exact Shapley values of the interactions,
since the game is additive. Completeness holds exactly:
$\sum_j\phi_j=s_{ui}-s_{ui}(\text{nothing logged})$.

*Proof.* Expand $\tilde w_uG$ with $G=G^{(-u)}+\text{contrib}(w_u)$. The
self-walk terms $\tilde w_{ui}\sum_j\tilde w_{uj}^2$ cancel against the local
correction, leaving $\tilde w_{ui}\sum_jc_{uj}^2w_{uj}$, which is linear in $w_{uj}$. ∎

**Theorem 6 (optimal minimal counterfactual explanation).** For candidates $i,k$
with $s_{ui}>s_{uk}$, the smallest set of the user's logged interactions whose
removal makes $k$ outrank $i$ is found by sorting $g_j=\phi^{(i)}_j-\phi^{(k)}_j$
in decreasing order and removing the largest until the cumulative sum exceeds
the margin. This takes $O(n+d_u\log d_u)$. If the sum of the positive $g_j$ does
not exceed the margin, no counterfactual explanation exists.

*Proof.* By Proposition 5 the margin after removing $S$ is
$s_{ui}-s_{uk}-\sum_{j\in S}g_j$. Minimising $|S|$ subject to
$\sum_Sg_j>$ margin is solved by taking the largest $g_j$ first. ∎

GNNExplainer, PGExplainer and CF-GNNExplainer approximate this by optimisation
on a surrogate. DRUP explanations are exact and certified. Verified in
`run_fat.py --sections explain` (maximum relative error of
completeness/additivity is reported).

## 8. Accountability: certified robustness to injected profiles

Assume the nuisances are frozen (estimated on vetted data) and consider an
attacker who injects $F$ fake users. Each fake user logs at most $L$
interactions with arbitrary items, labels and propensities $\ge\tau$.

**Theorem 7.** One injected row $v$ changes user $u$'s 3-hop score of item $k$ by
$$
\Delta_k=x_k(D+a_kc_{vk})-a_kx_k^2\quad(\text{DRUP}),\qquad \Delta_k=x_kD\quad(\text{uncorrected}),
$$
where $a=\tilde w_u$, $x=c_v\odot w_v$ and $D=\langle a,x\rangle$. Logged entries satisfy
$w_{vj}\in[\hat Y_{vj}-\hat Y_{vj}/\tau,\ \hat Y_{vj}+(1-\hat Y_{vj})/\tau]$
(DR), and unlogged ones equal $\hat Y_{vj}$. Therefore $D$ lies in an interval computed from the $L$
most favourable slots, and $\Delta_k$ is bounded in closed form by its
extremes over the box of $(D,x_k)$: it is linear in $D$ and quadratic in $x_k$.
The box contains every feasible $(D,x_k)$ but ignores that both depend on the
same $L$ entries, so the bounds are conservative, not exact.
Effects add up over fake users. Consequently, item $t$ is **certified** to stay
out of $u$'s top-$K$ against any $F$ fake profiles if
$$
s_{ut}+F\,\overline\Delta_t<\big(K{+}1\big)\text{-th largest of }\{s_{uk}+F\,\underline\Delta_k\}.
$$
The bound is attack-agnostic. It is valid by construction and is checked against
brute force (float32 and float64, logged and unlogged items), where it
reaches 62–67% of the worst case. It is also checked against the realised
attack in `run_fat.py`, with no violation.

A second valid certificate uses the fact that $F$ profiles can log at most
$F\cdot L$ items. Every other competitor keeps its unlogged bound, so the
threshold becomes the $(K{+}1{+}FL)$-th largest of those bounds. We certify if
either certificate holds.

On Coat (300 items), DRUP certifies 100% of users against one 21-interaction
profile, 99.6% against two and 21% against five. On KuaiRec (10,728 items,
51-interaction profiles), the worst case over victim-tailored profiles is
vacuous even for $F=1$, although no realised attack with up to 100 fake users
reached any real user's top-20.

*Lower bound for unclipped IPS.* With $\tau\to0$ a single fake interaction on
an item with propensity $p$ moves $W$ by $1/p$, so its influence is unbounded.
The clip is what makes influence bounded, and $\tau$ trades bias (Theorem 3.3)
for a certified radius. With **degrees taken from the imputation**, the
attacker cannot shrink its own degree to inflate its edge weights ($c_v$ is
known exactly), which tightens the certificate. Degrees taken from $W$ are
attacker-controlled.

## 9. Privacy: a joint-DP recommender from one public operator

User $v$ contributes
$\text{contrib}_v=\tilde w_v^\top\tilde w_v-\operatorname{diag}(\tilde w_v^2)+\operatorname{diag}(c_v^2w_v)$
to $G$. Scale each row so that $\|\tilde w_v\|_2\le R$ and $\|c_v^2w_v\|_2\le R^2$.

**Theorem 8.** Let the population-level nuisances (propensity model, item
parameters of $\hat Y$, item degrees, $R$) be computed from public data and the
user-level parts from each user's own row. Adding symmetric Gaussian noise with
the analytic calibration $\sigma=\sigma_{\mathrm{aG}}(\epsilon,\delta;2R^2)$ of Balle and
Wang (2018), which is valid for every $\epsilon>0$ (the classical
$\sigma=2R^2\sqrt{2\ln(1.25/\delta)}/\epsilon$ is only valid for $\epsilon<1$), to the
upper triangle of $G$ is $(\epsilon,\delta)$-DP under adding or removing one user, because
$\|\text{contrib}_v\|_F\le\|\tilde w_v\|_2^2+\|c_v^2w_v\|_2\le2R^2$. Each user's
recommendations are then computed locally from the released $G$ and the user's
own row (Section 2). By the billboard lemma the whole system is therefore
$(\epsilon,\delta)$-**jointly** differentially private. Any post-processing of the
released $G$, such as low-rank denoising, is free.

*Scope.* The guarantee is conditional on public nuisances. The experiments
(`run_fat.py --sections privacy`) fit the population-level nuisances on a public
10% of the non-test users and compute the user-level parts locally; the
denoising ranks are fixed in advance and all are reported. Row scaling shrinks
the estimate for rows with norm above $R$.

## 9b. Exposure-constrained allocation with a causal utility guarantee

The causal guarantee of Theorem 4 does not bound how concentrated the top-$K$
lists are. We add a re-ranker (`drup/rerank.py`). Among feasible allocations
$\mathcal X=\{x\in\{0,1\}^{R\times N}:\sum_ix_{ui}=K,\ \sum_ux_{ui}\le\mathrm{cap}_i\}$
(candidates only), it maximises $\sum x_{ui}s_{ui}$. This is a bipartite
$b$-matching: its constraint matrix is totally unimodular, so the LP has an
integral optimum. We solve it exactly by min-cost flow (OR-Tools) on the
scores rounded to $L=10^6$ levels, which is optimal up to
$\Gamma\le RK(\max s-\min s)/L$.

**Theorem 9.** Let $\hat x$ be the returned allocation, $\Gamma$ the rounding gap, and $U(x)=\sum x_{ui}F^*_{ui}$ the full-exposure utility.
(a) Every cap holds, so each item's share is at most $\max\mathrm{cap}/R$
and at least $RK/\max\mathrm{cap}$ items are recommended.
(b) Deterministically,
$U(\hat x)\ge\max_{x\in\mathcal X}U(x)-\Gamma-2\max_{x\in\mathcal X}|\langle x,s-F^*\rangle|$.
(c) For DRUP, $\mathbb E s=F^*$ and Theorem 3(d) apply. With probability at least
$1-\delta$, $|s_{ui}-F^*_{ui}|\le t_\delta=\sqrt{\tfrac12\max_{ui}\sum_e\bar c_e^2\log(2mn/\delta)}$
for all entries, where $\bar c_e(u,i)=b\,c_e(u,i)+a\,C_{ui}\varepsilon_{ui}\mathbb 1[e=(u,i)]/\tau$
adds the sensitivity of the one-hop term to the three-hop constant $c_e$ and
the union runs over all $mn$ pairs because the candidate set depends on the log (up to the clipping bias of Theorem 3(c)), hence
$U(\hat x)\ge\mathrm{OPT}-\Gamma-2RKt_\delta$. For logged-graph propagation,
$s$ concentrates around the exposure-weighted $F^*(P\odot Y)$ instead
(Theorem 4). The extra term $2\max_x|\langle x,\mathbb Es-F^*\rangle|$ does
not vanish, so the constrained re-ranker then optimises popularity-weighted
utility.

*Proof.* (a) is feasibility. For (b), write
$U(\hat x)=\langle\hat x,s\rangle-\langle\hat x,s-F^*\rangle\ge\langle x^*,s\rangle-\Gamma-\langle\hat x,s-F^*\rangle$
and $\langle x^*,s\rangle=U(x^*)+\langle x^*,s-F^*\rangle$. (c) follows from a union
bound over the $RN$ entries and $\|x\|_1=RK$. ∎

---

## 10. What is new relative to the survey's taxonomy

The survey (Yu et al., ACM CSUR 2026) lists propensity adjustment, doubly robust
estimation and counterfactual explanations as separate tools. It names
graph-specific bias propagation, missing theoretical guarantees, scalability and
privacy as open problems. DRUP contributes the following:

1. It identifies a bias that existing propagation debiasers (inverse-propensity
   adjacency, DR-imputed graphs) introduce, namely repeated-edge walks, and gives
   an exact bias formula and lower bound (Theorem 2).
2. It removes that bias at no extra cost, giving the first graph propagation
   operator that is exactly unbiased and edge-wise doubly robust for the
   full-exposure graph (Theorem 1), with variance, clipping and concentration
   bounds (Theorem 3).
3. Exposure invariance in expectation (Thm 4) is specific to the corrected
   operator. The remaining properties (exact attributions and counterfactual
   explanations, Prop 5 and Thm 6; certificates, Thm 7; joint DP of the item
   operator, Thm 8; capped allocation, Thm 9) follow from linearity with fixed
   nuisances and hold for any linear propagation operator, including the
   uncorrected ones. They describe the recommender with its nuisances frozen
   (certificates, explanations) or public (joint DP).
4. The walk correction does not change top-$K$ accuracy on the three
   benchmarks. Its value is unbiased score estimates and verifiable exposure
   invariance; see `README.md` for the evaluation.
