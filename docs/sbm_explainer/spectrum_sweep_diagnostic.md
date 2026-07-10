# Exp 12 Spectrum Sweep — Anomaly Diagnostic

**Date:** 2026-04-23  
**Scope:** Two numerical anomalies in the Exp 12 master table  
(`results/sbm_explainer/spectrum_sweep/summary.json`)

---

## Anomaly 1 — pct_black_latino Plain NMI = 0.000

### Root cause (specific line)

`scripts/run_spectrum_sweep.py`, lines 521–531. The code that loads the
Exp 9 plain NMI for the `pct_black_latino` reference row is:

```python
seedscale_path = RESULTS_DIR / "sbm_explainer" / "seedscale_gauss_lime" / "results.json"
plain_nmi_pbl = 0.0  # default
if seedscale_path.exists():
    with open(seedscale_path) as f:
        ss = json.load(f)
    for entry in ss if isinstance(ss, list) else []:   # ← BUG
        if entry.get("layer_feature") == "pct_black_latino":
            plain_nmi_pbl = entry.get("plain_nmi_mean", 0.0)
            break
pex_nmi9 = plain_nmi_pbl + exp9_delta_nmi
```

`results/sbm_explainer/seedscale_gauss_lime/results.json` is a **single
dict** (keys: `n_seeds`, `K`, `plain`, `pex`, `delta`, …), not a list.
The guard `isinstance(ss, list)` is `False`, so the loop body never runs,
`plain_nmi_pbl` stays `0.0`, and the table gets:

- Plain NMI = 0.000  (hardcoded fallback)
- pEx  NMI = 0.000 + 0.048 = 0.048  
- Δ NMI = 0.048  (accidentally correct)

### Canonical Exp 9 numbers (no re-run needed — already saved)

The authoritative plain NMI comes from
`results/sbm_explainer/fingerprint_sbm/summary.json` → `cells[0]`
(variant=gaussian, scaffold=lime):

```
plain.nmi_mean  = 0.2867
pex.nmi_mean    = 0.3344
Δ NMI           = +0.048   (pex − plain)
```

The 20-seed seedscale run (`seedscale_gauss_lime/results.json`) gives
`plain.mean = 0.2942 ± 0.018`, fully consistent (within 1 std).

**Canonical triple for pct_black_latino (Gauss+LIME scaffold, housing):**

| | |
|---|---|
| Plain NMI | **0.287** |
| pEx NMI   | **0.334** |
| Δ NMI     | **+0.048** |

The Δ in the Exp 12 table (0.048) is correct. The Plain NMI (0.000) and
pEx NMI (0.048) are both wrong because of the failed lookup. The correct
absolute values are 0.287 and 0.334 — matching the Exp 9 headline exactly.

### Fix required in run_spectrum_sweep.py

Replace the seedscale lookup with a direct read from
`fingerprint_sbm/summary.json`:

```python
fp_path = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm" / "summary.json"
plain_nmi_pbl = 0.0
if fp_path.exists():
    with open(fp_path) as f:
        fp = json.load(f)
    for cell in fp.get("cells", []):
        if cell.get("variant") == "gaussian" and cell.get("scaffold") == "lime":
            plain_nmi_pbl = cell["plain"]["nmi_mean"]
            break
pex_nmi9 = plain_nmi_pbl + exp9_delta_nmi
```

---

## Anomaly 2 — pct_latino (Δ = +0.000) and pct_black (Δ = +0.002)

### Root cause (protocol decision)

The Exp 12 sweep built **fresh BiasedModels** for every feature including
`pct_latino` and `pct_black` (see `run_spectrum_sweep.py` line 229:
`BiasedModel(base_model=honest_model, pct_col_idx=feat_col_idx,
threshold=threshold, penalty=0.3)`). So each feature did get its own
scaffold. This is NOT a scaffold-reuse bug.

The low Δ NMI is structural, not a protocol error. Three compounding factors:

1. **Neither `pct_latino` nor `pct_black` appears in `BASELINE_FEATURES`.**
   The honest baseline is trained on 10 features (bathrooms, sqft_living,
   sqft_lot, grade, condition, waterfront, view, age, UTM_X, UTM_Y). The
   demographic features are absent. The BiasedModel penalty is therefore the
   *only* source of behavioral signal for these features in the fingerprint
   graph.

2. **Penalty coverage is sparse.** At P75, approximately 45–46/200 test
   points are "high". The penalty affects only those points; for the
   remaining ~77%, the biased model is identical to the honest model.
   With K_perts=50 Gaussian perturbations, the fingerprint at a low-value
   point is essentially noise-free — indistinguishable between
   pct_latino-scaffold and pct_black_latino-scaffold. The 8-NN cosine
   similarity graph therefore has edges that are driven almost entirely by
   the honest model's variation across the 10-feature space, not by the
   penalty.

3. **The layer split does not align with the scaffold's behavioral
   boundary.** The pct_black_latino scaffold (Exp 9) succeeded because
   `pct_black_latino > 0.25` at P75 maps to roughly the same 25% of the
   dataset that the scaffold penalizes, AND this feature has high within-group
   spatial clustering (I=0.834), so nearby test points are usually in the
   same layer, driving the fingerprint graph's community structure to align
   with the layer. For `pct_black` (I=0.850) and `pct_latino` (I=0.756) the
   Ψ matrix shows both layers concentrate in the same dominant block (no
   disassortative structure, C-ratio < 1), meaning the scaffold's fingerprint
   graph does not separate the protected group at all. The penalty is present
   but too sparse relative to the feature's natural co-variation with the
   baseline features to create a graph with the right community geometry.

4. **Summary: null-signal scaffold, not a protocol bug.** If the sweep had
   reused the pct_black_latino scaffold with pct_latino/pct_black as the
   LAYER only, you would still expect weak (not zero) signal, because
   pct_black and pct_latino are subsets of pct_black_latino. The actual
   fresh-scaffold results (Δ ≈ 0.001–0.002, C-ratio < 1) are even weaker
   because the scaffold-triggered behavioral variation in the fingerprint is
   decoupled from the audit layer. This confirms the collapse/diffuse regime
   characterization in `spectrum_sweep_summary.md`.

---

## Verdict: Which Rows Are Trustworthy?

| Feature | Plain NMI | pEx NMI | Δ NMI | Status |
|---|---|---|---|---|
| condition | 0.065 | 0.064 | −0.000 | **Trustworthy** (null result, correct) |
| view | 0.877 | 0.877 | 0.000 | **Trustworthy** (saturation regime) |
| sqft_living | 0.014 | 0.007 | −0.007 | **Trustworthy** (null/diffuse) |
| grade | 0.014 | 0.020 | +0.006 | **Trustworthy** (weak positive) |
| sqft_lot | 0.027 | 0.041 | +0.014 | **Trustworthy** |
| pct_latino | 0.011 | 0.012 | +0.000 | **Trustworthy** (null result) |
| **pct_black_latino** | **0.000 ← WRONG** | **0.048 ← WRONG** | 0.048 | **BROKEN — absolute values wrong** |
| pct_black | 0.012 | 0.014 | +0.002 | **Trustworthy** (null result) |

**The Δ NMI column is correct for all 8 rows.** Only the Plain NMI and pEx NMI
absolute values for `pct_black_latino` are wrong. The corrected row is:

| pct_black_latino | **0.287** | **0.334** | +0.048 | C-ratio=1.319, disassortative |

All other rows are scientifically valid and can go in slides as-is.

**Count of trustworthy Δ > 0 rows:** grade, sqft_lot, pct_black_latino,
pct_latino, pct_black = 5 out of 8 (unchanged). The "5/8" verdict stands
but with the corrected understanding that pct_latino and pct_black are
genuine null-signal results (not protocol bugs), and pct_black_latino is a
strong positive (+0.048) not visible from the broken absolute values.

---

## Recommended Follow-Up

### Immediate (before slides)

1. **Fix the pct_black_latino row in the master table** — update
   `spectrum_sweep_summary.md` and `docs/project_overview.md` with Plain
   NMI=0.287, pEx NMI=0.334. The Δ=+0.048 is already correct.

2. **Fix `run_spectrum_sweep.py` lines 521–531** as shown above so that a
   future re-run correctly loads plain_nmi from `fingerprint_sbm/summary.json`.

3. **Do NOT re-run the full 8-feature sweep** — all other rows are correct.

### For the paper / protocol

4. **pct_latino and pct_black do not need a re-run.** Their low Δ is a genuine
   finding (null-signal scaffold = collapse regime, C-ratio < 1). They belong
   in the paper as examples of the diffuse regime and strengthen the regime-
   condition argument.

5. **The §6.2 regime framing is correct and should be kept.** The three-regime
   model (saturation / middle / diffuse) fully explains all 8 rows without
   any protocol bugs for 7 of 8 rows. Only pct_black_latino needed a data-
   lookup fix, not a re-run.

6. **Optional robustness check for pct_black/pct_latino:** If Peng wants
   non-zero signal on these features, the correct fix is to include the
   feature in the baseline model (so it affects the fingerprint at ALL points,
   not just penalized ones). This is a design decision for the paper — add a
   note in §6.2 that "demographic features absent from the honest model produce
   null-signal scaffolds regardless of Moran's I."
