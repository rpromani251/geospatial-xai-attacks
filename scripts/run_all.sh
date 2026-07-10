#!/usr/bin/env bash
set -e  # exit on any error
set -u  # treat unset variables as error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Geospatial XAI Attack Framework: Full Reproducibility Run ==="
echo "Working dir: $REPO_ROOT"
echo ""
echo "This runs all experiments documented in project_summary.md through"
echo "Experiment 12. Full runtime is approximately 1–2 hours depending on"
echo "hardware (GeoSHAP + SBM Gibbs sampling dominate). Set SKIP_SLOW=1 to"
echo "run only the fast core (~30 min, Exps 1–6 + baseline replication)."
echo ""

SKIP_SLOW="${SKIP_SLOW:-0}"

# --- Core pipeline (Exps 1-6) ---

echo "Step 1: Baseline XGBoost (Exp 1; 10 features incl. UTM)"
python3 scripts/run_baseline.py

echo "Step 2: Honest explainer baselines (Exp 2; KernelSHAP + TreeSHAP + GeoSHAP + LIME)"
python3 scripts/run_explanations.py

echo "Step 3: Adversarial scaffold attack (Exp 3; pct_black_latino>0.25 trigger)"
python3 scripts/run_scaffold_attack.py

echo "Step 4: No-location honest ablation (Exp 4)"
python3 scripts/run_no_location_experiment.py

echo "Step 4b: No-location scaffold ablation (Exp 4b)"
python3 scripts/run_scaffold_no_location.py

echo "Step 5: Robustness matrix (Exp 5; 3x5 incl. Ensemble)"
python3 scripts/run_robustness.py

echo "Step 5b: Slack-suite robustness replication (Exp 5b)"
python3 scripts/fix_treeshap_and_cc.py

echo "Step 6: Worst-case x* attack (Exp 6; eps sweep)"
python3 scripts/run_worst_case.py

# --- Moran's I thread (Exps 7, 8, 8b) ---

echo "Step 7: Moran's I spatial-autocorrelation survey (Exp 7)"
python3 scripts/run_morans_i.py

echo "Step 8: Feature sweep across Moran's I (Exp 8)"
python3 scripts/run_feature_sweep.py

# NB: Exp 8b (base-importance-normalized 8-feature sweep) currently has
# its artifacts in results/feature_sweep_a2/ but no dedicated script in
# this repo. To regenerate, run_feature_sweep.py needs a flag for the
# extended 8-feature set with normalized_visibility columns added.

# --- Non-autocorrelated bias-feature attempts (Exps 10, 10b, 10c, 10d, 11) ---

if [ "$SKIP_SLOW" = "0" ]; then
    echo "Step 10: Scaffold on FEMA 100-year floodplain (Exp 10)"
    python3 scripts/process_flood_zone.py 2>/dev/null || true
    python3 scripts/run_scaffold_flood_zone.py

    echo "Step 10b: Scaffold on FEMA 500-year flood zone (Exp 10b)"
    python3 scripts/process_flood_zone_500yr.py 2>/dev/null || true
    python3 scripts/run_scaffold_flood_500yr.py

    echo "Step 10c: Scaffold on KC parcel dataset (59k; Exp 10c)"
    # NB: requires results/scaffold_parcel_flood/baseline_* from build_kc_parcel_dataset.py
    python3 scripts/build_kc_parcel_dataset.py 2>/dev/null || true
    python3 scripts/run_scaffold_parcel_flood.py

    echo "Step 10d: Scaffold on geologic hazard (Exp 10d; 6.5% cov, the one that fires)"
    python3 scripts/run_scaffold_geologic_hazard.py

    echo "Step 11: Scaffold on SPS school zone (Exp 11)"
    python3 scripts/process_school_zone.py 2>/dev/null || true
    python3 scripts/run_school_zone_attack.py
fi

# --- SBM-as-explainer thread (Exp 9 + 12) ---

if [ "$SKIP_SLOW" = "0" ]; then
    echo "Step 9: Rung-3 fingerprint SBM (Exp 9 main; sqft_lot>P75 layer for housing)"
    python3 scripts/sbm_explainer/run_fingerprint_sbm.py
    python3 scripts/sbm_explainer/run_ood_flip_probe.py
    python3 scripts/sbm_explainer/run_honest_fingerprint_check.py
    python3 scripts/sbm_explainer/run_gauss_lime_seedscale.py

    echo "Step 9.4: 4-dataset replication (Slack COMPAS / German / CC)"
    python3 scripts/sbm_explainer/run_compas_fingerprint.py
    python3 scripts/sbm_explainer/run_german_fingerprint.py
    python3 scripts/sbm_explainer/run_cc_fingerprint.py

    echo "Step 9.5: Mechanism tests (K-sweep + tanh Lipschitz)"
    python3 scripts/sbm_explainer/run_k_scaling_sweep.py
    python3 scripts/sbm_explainer/run_tanh_sweep.py
    python3 scripts/sbm_explainer/run_tanh_seedscale.py

    echo "Step 9.6: H-DP extension (R sampler)"
    # Requires R + mcclust + mcclust.ext. Skip gracefully if R is missing.
    if command -v Rscript >/dev/null 2>&1; then
        python3 scripts/sbm_explainer/export_hdp_inputs.py
        Rscript scripts/sbm_explainer/run_hdp_pex.R "$REPO_ROOT"
        Rscript scripts/sbm_explainer/run_hdp_pex_sweep.R "$REPO_ROOT"
        python3 scripts/sbm_explainer/analyze_hdp_pex.py
        python3 scripts/sbm_explainer/analyze_hdp_refinement.py
        python3 scripts/sbm_explainer/analyze_hdp_attributions.py
    else
        echo "  (Rscript not found — skipping H-DP. Install R + mcclust.ext to enable.)"
    fi

    echo "Step 12: pEx-SBM across Moran's I spectrum (Exp 12; 8-feature sweep)"
    python3 scripts/run_spectrum_sweep.py
fi

# --- Figures ---

echo "Figures: diagnostic plots"
python3 scripts/plot_results.py
python3 scripts/plot_no_location.py
python3 scripts/plot_robustness.py
python3 scripts/plot_worst_case.py

echo "Figures: paper figures"
python3 scripts/plot_paper_figures.py

# Figures for the non-autocorrelated bias experiments (best-effort; scripts may
# have been deleted or refactored — wrap in guards)
python3 scripts/plot_parcel_flood_figures.py 2>/dev/null || true
python3 scripts/plot_flood_500yr.py 2>/dev/null || true
python3 scripts/plot_sbm_heatmap.py 2>/dev/null || true
python3 scripts/plot_demographics.py 2>/dev/null || true

echo ""
echo "=== All done. Results in results/ ==="
echo "  results/baseline/                        Exp 1"
echo "  results/explanations/                    Exp 2"
echo "  results/scaffold/                        Exp 3"
echo "  results/no_location/                     Exp 4"
echo "  results/scaffold_no_location/            Exp 4b"
echo "  results/robustness/                      Exp 5 (3x5 incl. Ensemble)"
echo "  results/slack_robustness/                Exp 5b"
echo "  results/worst_case/                      Exp 6"
echo "  results/morans_i/                        Exp 7"
echo "  results/feature_sweep/                   Exp 8"
echo "  results/feature_sweep_a2/                Exp 8b (artifacts only; see note above)"
echo "  results/sbm_explainer/                   Exp 9 + Exp 12 (rung-3 + spectrum sweep)"
echo "  results/scaffold_{flood_zone,flood_500yr,parcel_flood,geologic_hazard,school_zone}/  Exps 10-11"
echo "  results/paper_figures/                   Publication figures"
