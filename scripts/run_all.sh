#!/usr/bin/env bash
set -e  # exit on any error
set -u  # treat unset variables as error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Geospatial XAI Attack Framework: Full Reproducibility Run ==="
echo "Working dir: $REPO_ROOT"
echo ""

echo "Step 1/8: Baseline XGBoost (10 features, UTM)"
python3 scripts/run_baseline.py

echo "Step 2/8: KernelSHAP + LIME explanations (baseline)"
python3 scripts/run_explanations.py

echo "Step 3/8: Adversarial scaffold attack (KernelSHAP / LIME / GeoSHAP targets)"
python3 scripts/run_scaffold_attack.py

echo "Step 4/8: No-location experiment (TreeSHAP / KernelSHAP / GeoSHAP / LIME)"
python3 scripts/run_no_location_experiment.py

echo "Step 5/8: Robustness analysis (3x4 matrix)"
python3 scripts/run_robustness.py

echo "Step 6/8: Worst-case x* attack (eps sweep)"
python3 scripts/run_worst_case.py

echo "Step 7/8: Diagnostic plots"
python3 scripts/plot_results.py
python3 scripts/plot_no_location.py
python3 scripts/plot_robustness.py
python3 scripts/plot_worst_case.py

echo "Step 8/8: Paper figures"
python3 scripts/plot_paper_figures.py

echo ""
echo "=== All done. Results in results/ ==="
echo "  results/baseline/        -- model + metrics"
echo "  results/explanations/    -- KernelSHAP + LIME baselines"
echo "  results/scaffold/        -- adversarial attack outputs"
echo "  results/no_location/     -- no-UTM experiment"
echo "  results/robustness/      -- 3x4 matrix + figures"
echo "  results/worst_case/      -- x* sweep + figures"
echo "  results/paper_figures/   -- publication-ready figures"
