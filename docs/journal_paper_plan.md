# Journal Paper Plan — Adversarial Attacks on Geospatial Feature Attribution

Scope: the fooling-LIME/SHAP/GeoShapley geospatial thread only. Excludes pEx-SBM / SBM-as-explainer.
Drafted for the Peng meeting. References: `project_overview.md`, `experiments_index.md`, `experiments_narrative.md`, `comparison_honest_vs_adv_off_baseline.md`.

## 1. Thesis & contributions
Two claims:
- **C1 — transfer.** Geospatial ML models are susceptible to the same scaffold / OOD-routing attack on explanations as tabular models (Slack et al.).
- **C2 — different failure modes.** When coordinates (x, y) are model features, post-hoc attributions fail *differently* than in the tabular case — the LIME/SHAP failure ordering inverts.

Contributions:
1. First scaffold attack in a spatial setting (digital-redlining threat model).
2. The reversal: LIME partially resists, SHAP **and** GeoShapley are fooled — opposite of Slack's tabular result.
3. GeoShapley is not a defense — it is the *most* compromised on coordinates (background is on-manifold by construction).
4. (Secondary) On-manifold worst-case x\* + robustness ordering Random > Medoid ≈ Spatial.

## 2. Current evidence state (be honest with Peng)
Every headline number (SHAP UTM_Y −57%, GeoShapley −93%, LIME −3%) is a **single-seed point estimate** (`random_state=42` everywhere). The only existing p-values are incidental (spatial-correlation r at n=600; Moran's I permutation tests) — **neither C1 nor C2 has a significance test yet.** Required upgrade: multi-seed replication + paired/interaction tests + effect sizes with CIs.

## 3. Statistical plan

### 3.0 Design principles
- **Unit of replication = seed**, N ≥ 10. Re-fit the full pipeline per seed, varying all five randomness sources: train/test split, XGBoost fit, OOD detector, SHAP background draw, LIME perturbations.
- **Avoid pseudo-replication.** 600 test-point attributions are correlated, not 600 independent samples — at that n almost any difference is "significant." Per-point paired tests give *within-pipeline* significance; **seed-level estimates give the CI that actually carries the claim.**
- Report **effect sizes + 95% CIs**, not just p-values.
- Multiple-comparison correction (Holm or BH) across explainer pairs and datasets.

### 3.1 Tests per claim
**Test 1 — Attack suppresses coordinate attribution (C1).**
- Quantity: per-point |UTM_Y attribution|, honest vs scaffold.
- Within-seed: Wilcoxon signed-rank (paired by point). Effect size: Cliff's δ / rank-biserial.
- Across seeds: mean % suppression ± 95% CI (t or bootstrap over seeds).

**Test 2 — The reversal (C2, the core claim). This is an *interaction*, not three separate tests.**
- Per seed, compute the attribution drop d = honest − adversarial for each explainer.
- Contrasts (d_SHAP − d_LIME), (d_GeoSHAP − d_LIME) tested across seeds (paired by seed), Holm-corrected.
- Unified alternative: linear mixed model `attribution ~ condition * explainer + (1|seed) + (1|point)` — the condition×explainer interaction term **is** the reversal. → **Decision for Peng: per-seed nonparametric contrasts vs LMM.**

**Test 3 — "Opposite of Slack" (domain contrast).**
- Run the same attack + explainers on COMPAS / German / Communities-and-Crime (already in repo).
- Domain × explainer interaction: tabular → both fooled; geospatial → LIME resists.

**Test 4 — Robustness ordering Random > Medoid ≈ Spatial (secondary, C4).**
- Per-point Spearman ρ across the 3 neighborhood levels (paired). Friedman omnibus + Holm post-hoc Wilcoxon for Random>Medoid and Random>Spatial.
- "Medoid ≈ Spatial" is an **equivalence** claim → TOST with a pre-specified margin. → **Decision for Peng: margin (e.g., |Δρ| < 0.05).**

### 3.2 Compute note
GeoShapley is the bottleneck (robustness matrix ran only n=20 points). 10 seeds × GeoShapley over full test set may need test-point subsampling or a cluster run.

## 4. Caveat revisions (the six items flagged in repo audit)
| # | Issue | Action | Cost |
|---|-------|--------|------|
| 1 | "Demographics hidden" is wrong — the attack launders **UTM_Y**; `pct_black_latino` actually *rises* under SHAP | Rewrite the claim to "coordinate-based bias is laundered" | free |
| 2 | No-location ablation (Exp 4b) shows the attack works *without* coordinates | Fold in as a finding: mechanism is OOD-routing, so "strip coordinates" is **not** a defense (strengthens the paper) | free |
| 3 | Level-3 neighborhood: code (coordinate-only Gaussian) ≠ design (Gower-weighted k-medoids) | Decide which; re-run robustness | absorbed by harness |
| 4 | Worst-case x\*: code maximizes attribution distance, design says model-output distance | Pick one; re-run | absorbed by harness |
| 5 | GeoShapley "every coalition lands on-manifold" is **inferred from the outcome**, not measured | New experiment: coalition-routing probe measuring routing fraction | new experiment |
| 6 | "Adv SHAP" = KernelSHAP only; `geoshapley_reversal` JSON contradicts the docs' "not confirmed" | Resolve the discrepancy (free); decide whether to compute TreeSHAP-adv | mostly free |

## 5. Open questions for Peng
1. Test framework: per-seed nonparametric contrasts vs one linear mixed model?
2. Equivalence margin for "Medoid ≈ Spatial" (TOST)?
3. Include the tabular replication as the domain contrast, or just cite Slack and argue geospatial only?
4. Venue scope: is the robustness / worst-case x\* material (C4) in this paper or a second one?
5. How many seeds is "enough" given GeoShapley compute?

## 6. Work sequence
1. Resolve framing caveats (1, 2, 6-discrepancy) — free, do first.
2. Build multi-seed harness wrapping Exp 1/2/3/5/6 (absorbs caveats 3, 4).
3. Run N=10 seeds → distributions + CIs.
4. Run Tests 1–4.
5. GeoShapley coalition-routing probe (caveat 5).
6. Tabular replication for the domain contrast (Test 3).
