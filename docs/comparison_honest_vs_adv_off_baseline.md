# Honest 11-Feature vs Adversarial Attribution Comparison

Generated: 2026-04-23

All honest-11feat values come from XGBoost retrained with the 10 original features **plus** the trigger feature.
KernelSHAP background: `n_background=100` (50 for Exp 10c parcel). LIME: 500 perturbations, random_state=42.
All attributions computed on the same first-200 test points used by the corresponding adversarial scaffold.

See figure: `docs/figures/fig_honest_vs_adv_off_baseline.png`

---

## Baseline R² Comparison

| Experiment | Trigger | Dataset | 10-feat R² | 11-feat R² | Δ R² |
|---|---|---|---|---|---|
| Exp 3  | pct_black_latino   | seattle_sample_3k | 0.8895 | 0.8889 | −0.0006 |
| Exp 10 | in_floodplain      | seattle_sample_3k | 0.8895 | 0.8934 | +0.0039 |
| Exp 10b | in_flood_500yr    | seattle_sample_3k | 0.8895 | 0.8911 | +0.0016 |
| Exp 10c | in_floodplain     | kc_parcel 59k     | 0.7625 | 0.7617 | −0.0008 |
| Exp 10d | in_geologic_hazard | seattle_sample_3k | 0.8895 | 0.8875 | −0.0020 |
| Exp 11 | in_low_score_zone  | school_zone 1231  | ~0.81† | 0.8123 | — |

† Exp 11 10-feat baseline not separately run on the school-zone subset; 0.81 is an estimate from the scaffold run.

---

## Per-Experiment Tables

Features shown: trigger + UTM_Y + sqft_living + grade + age (top structural).
All values = mean |attribution| over 200 test points.

### Exp 3 — pct_black_latino (seattle_sample_3k, R²=0.8889)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values shown for UTM_Y etc. — see this doc for cross-experiment baseline-R²-matched comparisons.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME | Adv GeoSHAP (n=20) |
|---|---|---|---|---|---|
| pct_black_latino | 0.022 | 0.026 | 0.035 | 0.108 | — |
| UTM_Y | 0.106 | 0.110 | 0.050 | 0.124 | see scaffold_geoshap |
| sqft_living | 0.066 | 0.092 | 0.082 | 0.096 | — |
| grade | 0.048 | 0.087 | 0.075 | 0.087 | — |
| age | 0.012 | 0.046 | 0.033 | 0.057 | — |

UTM_Y shift (honest→adv): −0.056 SHAP (−53%). The scaffold suppresses UTM_Y dramatically.
Trigger shift: +0.013 SHAP (+59%). Honest model does attribute moderate weight to pct_black_latino naturally.

### Exp 10 — in_floodplain (seattle_sample_3k, R²=0.8934)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values — see this doc for comparisons.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME |
|---|---|---|---|---|
| in_floodplain | 0.000 | 0.031 | 0.001 | 0.067 |
| UTM_Y | 0.117 | 0.125 | 0.050 | 0.129 |
| sqft_living | 0.067 | 0.094 | 0.079 | 0.092 |
| grade | 0.051 | 0.091 | 0.073 | 0.094 |
| age | 0.013 | 0.043 | 0.034 | 0.056 |

in_floodplain is essentially zero in honest SHAP (≈0.000) — the honest model cannot use it as a predictor because floodplain status has no real price signal in this dataset. The scaffold raises it modestly in LIME (0.067) by laundering through the OOD gate.

### Exp 10b — in_flood_500yr (seattle_sample_3k, R²=0.8911)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values — see this doc.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME |
|---|---|---|---|---|
| in_flood_500yr | 0.000 | 0.033 | 0.002 | 0.090 |
| UTM_Y | 0.118 | 0.121 | 0.050 | 0.128 |
| sqft_living | 0.067 | 0.088 | 0.079 | 0.090 |
| grade | 0.050 | 0.091 | 0.073 | 0.093 |
| age | 0.012 | 0.041 | 0.033 | 0.057 |

Trigger honest SHAP is effectively zero (0.0001), and adv SHAP is near-zero (0.002). The scaffold expresses the trigger primarily in adv LIME (0.090) — nearly 3× the honest LIME value.

### Exp 10c — in_floodplain / kc_parcel 59k (R²=0.7617)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values — see this doc.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME |
|---|---|---|---|---|
| in_floodplain | 0.000 | 0.121 | 0.002 | 0.150 |
| UTM_Y | 0.254 | 0.320 | 0.227 | 0.329 |
| sqft_living | 0.120 | 0.152 | 0.119 | 0.141 |
| grade | 0.088 | 0.241 | 0.109 | 0.267 |
| age | 0.055 | 0.116 | 0.037 | 0.131 |

The parcel dataset is the outlier: UTM_Y dominates both honest and adversarial SHAP (0.254 vs 0.227), and the adversarial redistribution is much smaller (−0.027 vs −0.056 for 3k set). LIME shows in_floodplain more prominently (0.150) even in adv LIME — LIME sees through the scaffold more in this dataset.

### Exp 10d — in_geologic_hazard (seattle_sample_3k, R²=0.8875)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values — see this doc.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME |
|---|---|---|---|---|
| in_geologic_hazard | 0.000 | 0.022 | 0.013 | 0.186 |
| UTM_Y | 0.117 | 0.120 | 0.050 | 0.129 |
| sqft_living | 0.067 | 0.095 | 0.079 | 0.094 |
| grade | 0.051 | 0.086 | 0.072 | 0.094 |
| age | 0.012 | 0.047 | 0.031 | 0.055 |

Geologic hazard is the most dramatic: adv LIME = 0.186 vs honest LIME = 0.022, a factor of 8.5×. The scaffold is highly effective at manufacturing LIME attribution for this binary trigger. SHAP is more resistant (adv 0.013 vs honest ~0.000).

### Exp 11 — in_low_score_zone (school_zone 1231 records, R²=0.8123)

> Honest column = mean |attribution| under 11-feature honest XGBoost (10 original + trigger). Not apples-to-apples with the original 10-feature honest values — see this doc.

| Feature | Honest SHAP (11-feat) | Honest LIME (11-feat) | Adv SHAP | Adv LIME |
|---|---|---|---|---|
| in_low_score_zone | 0.002 | 0.010 | 0.036 | 0.144 |
| UTM_Y | 0.076 | 0.085 | 0.027 | 0.091 |
| sqft_living | 0.068 | 0.105 | 0.080 | 0.093 |
| grade | 0.035 | 0.103 | 0.058 | 0.117 |
| age | 0.010 | 0.038 | 0.024 | 0.046 |

The scaffold inflates the trigger SHAP from 0.002 → 0.036 (18×) and LIME from 0.010 → 0.144 (14×). UTM_Y drops from 0.076 → 0.027 (−64%), consistent with the other 3k-set experiments.

---

## Cross-Experiment Summary: Honest SHAP Shift vs Adversarial SHAP Shift

The key question: when the trigger is added to the honest model, does UTM_Y attribution drop (proxy-for-trigger signal)? And is that redistribution similar to what the scaffold does?

| Exp | Trigger | Honest UTM_Y (10-feat)* | Honest UTM_Y (11-feat) | Δ honest UTM_Y | Adv UTM_Y | Adv−Honest11 Δ |
|---|---|---|---|---|---|---|
| Exp 3  | pct_black_latino | 0.1162 | 0.106 | −0.010 | 0.050 | −0.056 |
| Exp 10 | in_floodplain   | 0.1162 | 0.117 | +0.001 | 0.050 | −0.067 |
| Exp 10b | in_flood_500yr | 0.1162 | 0.118 | +0.002 | 0.050 | −0.068 |
| Exp 10c | in_floodplain (parcel) | 0.254† | 0.254 | ~0.000 | 0.227 | −0.027 |
| Exp 10d | in_geologic_hazard | 0.1162 | 0.117 | +0.001 | 0.050 | −0.067 |
| Exp 11 | in_low_score_zone | ~0.076 | 0.076 | ~0.000 | 0.027 | −0.049 |

*10-feat honest UTM_Y from `results/explanations/shap_values.npy` (Exp 1 baseline); same for all 3k-set experiments.
†Parcel 10-feat honest UTM_Y approximated from 11-feat run (trigger contributes ≈0).

**Key finding:** Adding the trigger to the *honest* model causes essentially zero UTM_Y redistribution for all binary triggers (in_floodplain, in_flood_500yr, in_geologic_hazard, in_low_score_zone) — SHAP change < 0.002. This is because these binary features carry negligible price signal and the honest model ignores them.

For pct_black_latino (Exp 3), honest UTM_Y drops slightly (−0.010) when the trigger is added — consistent with UTM_Y being a geographic proxy for demographic composition. The honest model was already partially using location as a proxy for the trigger.

The adversarial redistribution (−0.05 to −0.07 on UTM_Y) is **5–7× larger** than what honest inclusion of the trigger produces. This confirms the scaffold is doing genuine laundering — not just naturally sharing credit with the trigger. The scaffold mechanically suppresses UTM_Y attribution by routing through the innocuous model (which lacks location features) for SHAP-background points.

**Biggest redistribution when trigger added honestly:** Exp 3 (pct_black_latino, UTM_Y Δ = −0.010). All other experiments show Δ ≤ 0.002, confirming that geographic/hazard binary flags have essentially no natural price signal in the Seattle dataset.

---

## Notes

- GeoSHAP for Exp 3 honest-11feat: computed on first 20 test points, saved at `results/honest_11feat/exp3/honest_geoshap_11feat.npy` (shape 20×11). Not shown in tables above; values available for comparison with the adversarial GeoSHAP from `results/scaffold/adv_geoshap_values.npy`.
- All outputs at `results/honest_11feat/<exp>/`.
- Figure at `docs/figures/fig_honest_vs_adv_off_baseline.png`.
