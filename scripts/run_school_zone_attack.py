"""Run the adversarial scaffold attack with the SPS school-zone binary trigger.

Experiment 11 — Non-Autocorrelated Bias (SPS School Zone).

Prerequisites:
  python scripts/process_school_zone.py     # builds seattle_housing_with_school_zone.csv

Outputs (all in results/scaffold_school_zone/):
  adv_shap_values.npy
  adv_lime_values.npy
  adv_geoshapley_{primary,geo,interaction,flat}.npy
  metrics.json
  spatial_correlation.json
  morans_i.json              (copied from process_school_zone.py output if present)

Key comparison: does GeoShapley reverse from most-compromised to most-resistant
when the bias feature (in_low_score_zone) has low spatial autocorrelation?
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel, get_safe_feature_indices
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.explainers.geoshap_runner import explain_geoshapley, geoshapley_to_flat, save_geoshapley
from src.utils.io import load_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mean_abs_attr(arr: np.ndarray, idx: int) -> float:
    return float(np.abs(arr[:, idx]).mean())


def compare_attributions(honest, adversarial, feature_names, label):
    """Print side-by-side honest vs adversarial mean |attribution|."""
    mean_h = np.abs(honest).mean(axis=0)
    mean_a = np.abs(adversarial).mean(axis=0)
    rank_h = np.argsort(mean_h)[::-1]
    print(f"\n{label} — mean |attribution|")
    print(f"  {'Feature':>22}  {'Honest':>9}  {'Adversarial':>12}  {'Shift':>8}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*12}  {'-'*8}")
    for idx in rank_h:
        if idx >= len(mean_a):
            continue
        name = feature_names[idx]
        h, a = mean_h[idx], mean_a[idx]
        print(f"  {name:>22}  {h:9.4f}  {a:12.4f}  {a-h:+8.4f}")
    rank_a = np.argsort(mean_a)[::-1]
    print(f"  Honest top-3:      {[feature_names[i] for i in rank_h[:3]]}")
    print(f"  Adversarial top-3: {[feature_names[i] for i in rank_a[:3]]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ── 1. Config ────────────────────────────────────────────────────────────
    cfg_path = CONFIGS_DIR / "school_zone.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    bias_feature = config.get("bias_feature", "in_low_score_zone")
    out_dir = RESULTS_DIR / config.get("output_dir", "scaffold_school_zone")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Load data ─────────────────────────────────────────────────────────
    X, y = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=config["features"],
        target_column=config["target"],
    )
    feature_names = list(X.columns)
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # ── 3. Train/test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    X_train_np = X_train.values
    X_test_np = X_test.values

    # ── 4. Honest baseline model ─────────────────────────────────────────────
    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Honest baseline model loaded.")

    # ── 5. Biased model ──────────────────────────────────────────────────────
    baseline_features = [f for f in feature_names if f != bias_feature]
    base_feature_indices = [feature_names.index(f) for f in baseline_features]
    bias_col_idx = feature_names.index(bias_feature)

    f_biased = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=bias_col_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )
    n_penalized = int(np.sum(X_test_np[:, bias_col_idx] > config["biased_model"]["threshold"]))
    pct_penalized = n_penalized / len(X_test)
    print(f"Biased model: penalty={config['biased_model']['penalty']}, "
          f"threshold={config['biased_model']['threshold']}")
    print(f"  {n_penalized}/{len(X_test)} test points in low-score zone ({pct_penalized*100:.1f}%)")
    if pct_penalized < 0.05:
        print("  WARNING: fewer than 5% of test points are penalized — low statistical power.")

    # ── 6. Innocuous model ───────────────────────────────────────────────────
    safe_indices = get_safe_feature_indices(
        feature_names,
        exclude=["UTM_X", "UTM_Y", bias_feature],
    )
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])
    psi_r2 = r2_score(y_test, psi.predict(X_test_np))
    print(f"Innocuous model R²: {psi_r2:.4f} (on {len(safe_indices)} safe features)")

    # ── 7. OOD detectors + scaffolds ────────────────────────────────────────
    print("\nTraining LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(f_biased, psi, X_train_np, config["lime_ood"])
    lime_fidelity = lime_scaffold.fidelity(X_test_np)
    print(f"  OOD accuracy: {lime_scaffold.ood_diagnostics['accuracy']:.4f}")
    print(f"  Fidelity: r={lime_fidelity['pearson_r']:.4f}, "
          f"routed_biased={lime_fidelity['pct_routed_to_biased']*100:.1f}%")

    print("\nTraining SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(f_biased, psi, X_train_np, config["shap_ood"])
    shap_fidelity = shap_scaffold.fidelity(X_test_np)
    print(f"  OOD accuracy: {shap_scaffold.ood_diagnostics['accuracy']:.4f}")
    print(f"  Fidelity: r={shap_fidelity['pearson_r']:.4f}, "
          f"routed_biased={shap_fidelity['pct_routed_to_biased']*100:.1f}%")

    # ── 8. Generate adversarial explanations ─────────────────────────────────
    n_explain = config["explanation"]["n_explain"]
    X_explain = X_test.iloc[:n_explain]
    X_explain_np = X_explain.values

    print(f"\nGenerating adversarial explanations for {n_explain} test points …")

    print("  SHAP (adversarial)...")
    adv_shap = explain_shap(
        shap_scaffold, X_train, X_explain,
        n_background=config["explanation"]["shap_n_background"],
    )

    print("  LIME (adversarial)...")
    adv_lime = explain_lime(
        lime_scaffold, X_train, X_explain,
        num_samples=config["explanation"]["lime_num_samples"],
        random_state=config["explanation"]["lime_random_state"],
    )

    # GeoShapley uses shap_scaffold (which routes via SHAP OOD detector)
    print("  GeoShapley (adversarial, via shap_scaffold)...")
    adv_geo_results = explain_geoshapley(shap_scaffold, X_train, X_explain)
    adv_geo_flat = geoshapley_to_flat(adv_geo_results)

    # ── 9. Load honest baseline explanations ─────────────────────────────────
    expl_dir = RESULTS_DIR / "explanations"
    honest_shap = np.load(expl_dir / "shap_values.npy")       # (200, 10)
    honest_lime = np.load(expl_dir / "lime_values.npy")       # (200, 10)
    honest_geo_flat = np.load(expl_dir / "geoshapley_flat.npy")
    honest_features = np.load(expl_dir / "feature_names.npy").tolist()

    # Align adversarial (11-feature) with honest (10-feature)
    shared_idx_adv = [feature_names.index(f) for f in honest_features if f in feature_names]

    compare_attributions(honest_shap, adv_shap[:, shared_idx_adv], honest_features, "KernelSHAP")
    compare_attributions(honest_lime, adv_lime[:, shared_idx_adv], honest_features, "LIME")
    # GeoShapley flat comparison: non-geo features only
    non_geo_honest = [f for f in honest_features if f not in ("UTM_X", "UTM_Y")]
    n_shared = len(non_geo_honest)
    compare_attributions(honest_geo_flat[:, :n_shared], adv_geo_flat[:, :n_shared], non_geo_honest, "GeoShapley")

    bias_idx_adv = feature_names.index(bias_feature)
    non_geo_adv = [f for f in feature_names if f not in ("UTM_X", "UTM_Y")]
    bias_flat_idx = non_geo_adv.index(bias_feature)
    print(f"\n  {bias_feature} mean |SHAP| in adversarial:        {mean_abs_attr(adv_shap, bias_idx_adv):.4f}")
    print(f"  {bias_feature} mean |LIME| in adversarial:        {mean_abs_attr(adv_lime, bias_idx_adv):.4f}")
    print(f"  {bias_feature} mean |GeoShapley| in adversarial:  {np.abs(adv_geo_flat[:, bias_flat_idx]).mean():.4f}")

    # ── 10. Key comparison table (bias feature + geo_effect + sqft_living) ──
    print("\n─── Key Attribution Table ───────────────────────────────────────")
    key_features = [bias_feature, "geo_effect", "sqft_living"]
    print(f"  {'Feature':>22}  {'H-SHAP':>7}  {'A-SHAP':>7}  {'H-LIME':>7}  {'A-LIME':>7}  {'H-GeoS':>7}  {'A-GeoS':>7}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    attribution_table = {}
    for feat in key_features:
        if feat == bias_feature:
            h_s, h_l, h_g = 0.0, 0.0, 0.0
            a_s = mean_abs_attr(adv_shap, bias_idx_adv)
            a_l = mean_abs_attr(adv_lime, bias_idx_adv)
            a_g = float(np.abs(adv_geo_flat[:, bias_flat_idx]).mean())
        elif feat == "geo_effect":
            h_s, h_l = float("nan"), float("nan")
            h_g = float(np.abs(honest_geo_flat[:, -1]).mean())
            a_s, a_l = float("nan"), float("nan")
            a_g = float(np.abs(adv_geo_flat[:, -1]).mean())
        elif feat in non_geo_honest:
            hi = honest_features.index(feat)
            ai = feature_names.index(feat)
            fi = non_geo_honest.index(feat)
            h_s = float(np.abs(honest_shap[:, hi]).mean())
            h_l = float(np.abs(honest_lime[:, hi]).mean())
            h_g = float(np.abs(honest_geo_flat[:, fi]).mean())
            a_s = mean_abs_attr(adv_shap, ai)
            a_l = mean_abs_attr(adv_lime, ai)
            a_fi = non_geo_adv.index(feat)
            a_g = float(np.abs(adv_geo_flat[:, a_fi]).mean())
        else:
            continue
        def _fmt(v):
            return f"{v:7.4f}" if not (isinstance(v, float) and np.isnan(v)) else "    —  "
        print(f"  {feat:>22}  {_fmt(h_s)}  {_fmt(a_s)}  {_fmt(h_l)}  {_fmt(a_l)}  {_fmt(h_g)}  {_fmt(a_g)}")
        attribution_table[feat] = {
            "honest_shap": round(h_s, 4) if not np.isnan(h_s) else None,
            "adv_shap": round(a_s, 4) if not np.isnan(a_s) else None,
            "honest_lime": round(h_l, 4) if not np.isnan(h_l) else None,
            "adv_lime": round(a_l, 4) if not np.isnan(a_l) else None,
            "honest_geoshapley": round(h_g, 4), "adv_geoshapley": round(a_g, 4),
        }

    # ── 11. Spatial correlation analysis ─────────────────────────────────────
    from scipy.stats import pearsonr
    import pandas as pd

    bias_vals = X_test_np[:, bias_col_idx]
    biased_preds = f_biased.predict(X_test_np)
    honest_preds = honest_model.predict(X_test_np[:, base_feature_indices])

    r_biased, p_biased = pearsonr(bias_vals, biased_preds)
    r_honest, p_honest = pearsonr(bias_vals, honest_preds)
    print(f"\n  Pearson r(in_low_score_zone, biased_pred): {r_biased:+.4f} (p={p_biased:.4e})")
    print(f"  Pearson r(in_low_score_zone, honest_pred): {r_honest:+.4f} (p={p_honest:.4e})")

    corr_metrics = {
        "pearson_r_biased_vs_school_zone": round(float(r_biased), 4),
        "pearson_p_biased": float(p_biased),
        "pearson_r_honest_vs_school_zone": round(float(r_honest), 4),
        "pearson_p_honest": float(p_honest),
    }

    # ── 12. GeoShapley geo_effect check ─────────────────────────────────────
    # GeoShapley decomposes spatial effects into a single geo_effect column.
    # Check whether the geo effect is suppressed under the adversarial scaffold.
    geo_honest_effect = float(np.abs(honest_geo_flat[:, -1]).mean())
    geo_adv_effect = float(np.abs(adv_geo_flat[:, -1]).mean())
    geo_pct_change = (geo_adv_effect - geo_honest_effect) / (geo_honest_effect + 1e-9) * 100
    print(f"\n  GeoShapley geo_effect: honest={geo_honest_effect:.4f} → adversarial={geo_adv_effect:.4f} "
          f"({geo_pct_change:+.1f}%)")
    if geo_pct_change > -30:
        print("  GeoShapley REVERSAL CONFIRMED: geo_effect is NOT suppressed (school-zone bias is low-autocorr)")
    else:
        print("  GeoShapley not reversed — geo_effect still suppressed.")

    # ── 13. Save all outputs ─────────────────────────────────────────────────
    np.save(out_dir / "adv_shap_values.npy", adv_shap)
    np.save(out_dir / "adv_lime_values.npy", adv_lime)
    save_geoshapley(adv_geo_results, out_dir, prefix="adv_geoshapley")
    np.save(out_dir / "feature_names.npy", np.array(feature_names))

    metrics = {
        "innocuous_model_r2": float(psi_r2),
        "n_penalized_test": n_penalized,
        "pct_penalized_test": round(pct_penalized, 4),
        "lime": {
            "ood_accuracy": lime_scaffold.ood_diagnostics["accuracy"],
            **lime_fidelity,
        },
        "shap": {
            "ood_accuracy": shap_scaffold.ood_diagnostics["accuracy"],
            **shap_fidelity,
        },
        "geoshap_ood_accuracy": shap_scaffold.ood_diagnostics["accuracy"],
        "n_explain": n_explain,
        "attribution_table": attribution_table,
    }
    save_metrics(metrics, out_dir / "metrics.json")
    save_metrics(corr_metrics, out_dir / "spatial_correlation.json")

    # Copy Moran's I if already computed by process_school_zone.py
    morans_src = out_dir / "morans_i.json"
    if not morans_src.exists():
        print("  morans_i.json not found — run process_school_zone.py first.")

    print(f"\nAll outputs saved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
