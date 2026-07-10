"""Experiment 4b — No-Location Scaffold Ablation.

Runs the Exp 3 scaffold attack with UTM_X and UTM_Y excluded from ALL model
inputs (biased model, innocuous model, OOD detectors). The demographic trigger
(pct_black_latino > 0.25) is still computed via the tract spatial join — the
penalty mechanism is unchanged. This tests whether coordinate laundering is the
mechanism that makes the scaffold work, or whether the attack can still hide
bias without spatial features.

Outputs land in results/scaffold_no_location/.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel, get_safe_feature_indices
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.utils.io import save_metrics
from src.utils.paths import RESULTS_DIR

import pandas as pd


def compare_attributions(honest, adversarial, feature_names, label):
    """Print comparison of feature importance rankings."""
    mean_honest = np.abs(honest).mean(axis=0)
    mean_adv = np.abs(adversarial).mean(axis=0)

    rank_honest = np.argsort(mean_honest)[::-1]
    rank_adv = np.argsort(mean_adv)[::-1]

    print(f"\n{label} Feature Importance Comparison:")
    print(f"  {'Feature':>20}  {'Honest':>10}  {'Adversarial':>12}  {'Shift':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*8}")

    for idx in rank_honest:
        name = feature_names[idx]
        h = mean_honest[idx]
        a = mean_adv[idx] if idx < len(mean_adv) else 0.0
        shift = a - h
        print(f"  {name:>20}  {h:10.4f}  {a:12.4f}  {shift:+8.4f}")

    print(f"\n  Honest top-3:      {[feature_names[i] for i in rank_honest[:3]]}")
    print(f"  Adversarial top-3: {[feature_names[i] for i in rank_adv[:3]]}")


def main():
    # ------------------------------------------------------------------ #
    # 1. Feature lists — no UTM_X / UTM_Y                                #
    # ------------------------------------------------------------------ #
    # 9 features total: 8 structural + pct_black_latino (no coordinates)
    NO_LOC_FEATURES = [
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "grade",
        "condition",
        "waterfront",
        "view",
        "age",
        "pct_black_latino",
    ]

    # Safe features for innocuous model: structural only (no coords, no demo)
    SAFE_FEATURES = [
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "grade",
        "condition",
        "waterfront",
        "view",
        "age",
    ]

    # Scaffold attack config (reuse Exp 3 OOD/explanation settings, but
    # use intentionally weaker hyperparams for the innocuous model so
    # psi and honest_no_loc are genuinely distinct predictors — Exp 4b
    # audit 2026-04-23 found that the prior (n_est=300, depth=5,
    # seed=42) matched honest_no_loc bit-for-bit, collapsing the
    # scaffold into "penalty on/off" rather than "biased vs different
    # innocuous model." The weaker psi (n_est=100, depth=3, seed=7)
    # diverges from honest_no_loc's R²=0.64 by about 5 percentage
    # points, restoring the structural distinction this ablation needs.
    config = {
        "biased_model": {"penalty": 0.3, "threshold": 0.25},
        "innocuous_model": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 7,
        },
        "lime_ood": {
            "perturbation_std": 0.3,
            "perturbation_multiplier": 30,
            "rf_estimators": 100,
        },
        "shap_ood": {
            "n_kmeans": 10,
            "perturbation_multiplier": 10,
            "n_samples": 20000,
            "rf_estimators": 100,
        },
        "explanation": {
            "n_explain": 200,
            "shap_n_background": 100,
            "lime_num_samples": 500,
            "lime_random_state": 42,
        },
    }

    # ------------------------------------------------------------------ #
    # 2. Load data — demographics dataset, no UTM columns                #
    # ------------------------------------------------------------------ #
    X, y = load_housing(
        dataset="demographics",
        feature_columns=NO_LOC_FEATURES,
        target_column="log_price",
    )
    feature_names = list(X.columns)
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # ------------------------------------------------------------------ #
    # 3. Train/test split                                                 #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    X_train_np = X_train.values
    X_test_np = X_test.values

    # ------------------------------------------------------------------ #
    # 4. Load honest baseline model (10-feature, includes UTM_X/Y)       #
    # The baseline model was trained on 10 features including UTM_X/Y.   #
    # For the no-location biased model, we train a fresh honest model on  #
    # the 8 structural features only, then wrap it in BiasedModel.       #
    # ------------------------------------------------------------------ #
    import xgboost as xgb

    print("\nTraining honest no-location model (8 structural features) for biased wrapper...")
    safe_indices_np = [feature_names.index(f) for f in SAFE_FEATURES]
    pct_col_idx = feature_names.index("pct_black_latino")

    X_train_safe = X_train_np[:, safe_indices_np]
    X_test_safe = X_test_np[:, safe_indices_np]

    honest_no_loc = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    honest_no_loc.fit(X_train_safe, y_train.values)
    honest_preds = honest_no_loc.predict(X_test_safe)
    honest_r2 = r2_score(y_test, honest_preds)
    print(f"Honest no-location model R²: {honest_r2:.6f}  "
          f"(n_est=300, max_depth=5, seed=42)")

    # ------------------------------------------------------------------ #
    # 5. Build biased model                                               #
    # BiasedModel.predict(X) expects full 9-feature array;               #
    # base_feature_indices selects the 8 safe features inside the model. #
    # ------------------------------------------------------------------ #
    f_biased = BiasedModel(
        base_model=honest_no_loc,
        base_feature_indices=safe_indices_np,
        pct_col_idx=pct_col_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )
    print(f"\nBiased model: penalty={config['biased_model']['penalty']}, "
          f"threshold={config['biased_model']['threshold']}")

    pct_vals = X_test_np[:, pct_col_idx]
    n_penalized = int(np.sum(pct_vals > config["biased_model"]["threshold"]))
    print(f"  {n_penalized}/{len(X_test)} test points in high-minority tracts "
          f"({n_penalized/len(X_test)*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 6. Build innocuous model                                            #
    # Same 8 safe features; no coordinates, no demographics.             #
    # ------------------------------------------------------------------ #
    psi = InnocuousModel(safe_col_indices=safe_indices_np)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])

    psi_preds = psi.predict(X_test_np)
    psi_r2 = r2_score(y_test, psi_preds)
    ic = config["innocuous_model"]
    print(f"Innocuous model R²:           {psi_r2:.6f}  "
          f"(n_est={ic['n_estimators']}, max_depth={ic['max_depth']}, "
          f"seed={ic['random_state']})")
    # Check that psi and honest_no_loc are genuinely distinct predictors
    # (the Exp 4b audit needed their predictions to diverge, not
    # specifically for psi to be weaker). |ΔR²| > 0.01 is the threshold.
    r2_gap = honest_r2 - psi_r2
    if abs(r2_gap) < 1e-9:
        verdict = "← COLLAPSED: bit-identical R² → models are the same XGB"
    elif abs(r2_gap) < 0.01:
        verdict = "← WARNING: R² near-identical (0.01); predictions may still be too similar"
    else:
        direction = "psi weaker" if r2_gap > 0 else "psi stronger"
        verdict = f"← models genuinely distinct ✓ ({direction}: |ΔR²| = {abs(r2_gap):.4f})"
    print(f"  ΔR² (honest_no_loc − psi) = {r2_gap:+.6f}  {verdict}")

    # ------------------------------------------------------------------ #
    # 7. Build LIME scaffold                                              #
    # ------------------------------------------------------------------ #
    print("\nTraining LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(
        f_biased, psi, X_train_np, config["lime_ood"],
    )
    print(f"  OOD detector accuracy: {lime_scaffold.ood_diagnostics['accuracy']:.4f}")

    lime_fidelity = lime_scaffold.fidelity(X_test_np)
    print(f"  Fidelity: r={lime_fidelity['pearson_r']:.4f}, "
          f"MSE={lime_fidelity['mse']:.6f}, "
          f"routed to biased={lime_fidelity['pct_routed_to_biased']*100:.1f}%")

    # ------------------------------------------------------------------ #
    # 8. Build SHAP scaffold                                              #
    # ------------------------------------------------------------------ #
    print("\nTraining SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, config["shap_ood"],
    )
    print(f"  OOD detector accuracy: {shap_scaffold.ood_diagnostics['accuracy']:.4f}")

    shap_fidelity = shap_scaffold.fidelity(X_test_np)
    print(f"  Fidelity: r={shap_fidelity['pearson_r']:.4f}, "
          f"MSE={shap_fidelity['mse']:.6f}, "
          f"routed to biased={shap_fidelity['pct_routed_to_biased']*100:.1f}%")

    # ------------------------------------------------------------------ #
    # 9. Generate adversarial explanations                                #
    # ------------------------------------------------------------------ #
    n_explain = config["explanation"]["n_explain"]
    X_explain = X_test.iloc[:n_explain]
    X_explain_np = X_explain.values

    print(f"\nGenerating adversarial explanations for {n_explain} test points...")

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

    # ------------------------------------------------------------------ #
    # 10. Compute HONEST explanations on honest_no_loc (the biased-branch #
    # base model) — this matches hyperparams, feature set, and training  #
    # data exactly, so the honest-vs-adv comparison isolates the scaffold #
    # attack effect (not UTM-removal or hyperparam changes).              #
    #                                                                      #
    # Fixed 2026-04-23: prior version loaded 10-feature Exp 2 baseline    #
    # (n=500, depth=6, R²=0.89) as the honest reference, which conflated  #
    # scaffold effect with UTM-removal and hyperparameter mismatch.       #
    # ------------------------------------------------------------------ #
    print("\n--- Computing honest baseline explanations on honest_no_loc ---")
    print("(same 9-feature model that the biased branch wraps; matches")
    print(" hyperparams and features so shifts isolate the attack effect)")

    # BiasedModel-style interface: honest_no_loc takes 8 safe features, but
    # we want explanations in the 9-feature space for alignment with adv
    # outputs. Wrap honest_no_loc so it accepts the full 9-feature input
    # and drops the pct_black_latino column internally.
    class HonestNoLocWrapper:
        def __init__(self, model, safe_indices):
            self.model = model
            self.safe_indices = safe_indices
        def predict(self, X):
            X = np.asarray(X)
            return self.model.predict(X[:, self.safe_indices])

    honest_wrapper = HonestNoLocWrapper(honest_no_loc, safe_indices_np)

    print("  honest SHAP on honest_no_loc ...")
    honest_shap_vals = explain_shap(
        honest_wrapper, X_train, X_explain,
        n_background=config["explanation"]["shap_n_background"],
    )
    print("  honest LIME on honest_no_loc ...")
    honest_lime_vals = explain_lime(
        honest_wrapper, X_train, X_explain,
        num_samples=config["explanation"]["lime_num_samples"],
        random_state=config["explanation"]["lime_random_state"],
    )

    # Save for reproducibility / external comparison
    out_dir = RESULTS_DIR / "scaffold_no_location"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "honest_shap_values.npy", honest_shap_vals)
    np.save(out_dir / "honest_lime_values.npy", honest_lime_vals)

    # Comparison on the shared 9 features (both arrays use the same columns)
    shared_features = feature_names   # 9 features
    honest_shared_idx = list(range(len(feature_names)))
    honest_shap_full = honest_shap_vals
    honest_lime_full = honest_lime_vals
    adv_shared_idx = [feature_names.index(f) for f in shared_features]

    print(f"\nShared features for comparison ({len(shared_features)}): {shared_features}")

    compare_attributions(
        honest_shap_full[:, honest_shared_idx],
        adv_shap[:, adv_shared_idx],
        shared_features, "SHAP",
    )
    compare_attributions(
        honest_lime_full[:, honest_shared_idx],
        adv_lime[:, adv_shared_idx],
        shared_features, "LIME",
    )

    pct_idx = feature_names.index("pct_black_latino")
    print(f"\n  pct_black_latino mean |SHAP| in adversarial: {np.abs(adv_shap[:, pct_idx]).mean():.4f}")
    print(f"  pct_black_latino mean |LIME| in adversarial: {np.abs(adv_lime[:, pct_idx]).mean():.4f}")

    # ------------------------------------------------------------------ #
    # 11. Comparison table for key features                               #
    # ------------------------------------------------------------------ #
    KEY_FEATURES = ["sqft_living", "pct_black_latino", "age", "grade", "bathrooms"]
    # UTM_Y not in no-loc feature set — it's what we removed

    print("\n--- Comparison Table (mean |attribution|) ---")
    header = f"  {'Feature':>20}  {'Honest SHAP':>12}  {'Adv SHAP':>10}  {'Honest LIME':>12}  {'Adv LIME':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # honest_shap_full / honest_lime_full now live in the 9-feature no-loc
    # column order (they were computed on honest_no_loc inside this script
    # — see §10 above). That matches feature_names exactly. `honest_features_full`
    # is no longer a separate 10-feat array; it IS feature_names.
    for feat in KEY_FEATURES:
        if feat in feature_names:
            ai = feature_names.index(feat)
            hs = np.abs(honest_shap_full[:, ai]).mean()
            as_ = np.abs(adv_shap[:, ai]).mean()
            hl = np.abs(honest_lime_full[:, ai]).mean()
            al = np.abs(adv_lime[:, ai]).mean()
            print(f"  {feat:>20}  {hs:12.4f}  {as_:10.4f}  {hl:12.4f}  {al:10.4f}")
        else:
            # UTM_Y: not in no-loc at all (it's what we removed)
            print(f"  {feat:>20}  {'—':>12}  {'—':>10}  {'—':>12}  {'—':>10}")

    # ------------------------------------------------------------------ #
    # 12. Spatial correlation analysis                                    #
    # ------------------------------------------------------------------ #
    print("\n--- Spatial Correlation Analysis ---")
    biased_preds_test = f_biased.predict(X_test_np)
    honest_no_loc_preds = honest_no_loc.predict(X_test_safe)

    r_biased, p_biased = pearsonr(pct_vals, biased_preds_test)
    r_honest, p_honest = pearsonr(pct_vals, honest_no_loc_preds)
    print(f"  Pearson r(pct_minority, biased_pred):  {r_biased:+.4f} (p={p_biased:.4f})")
    print(f"  Pearson r(pct_minority, honest_pred):  {r_honest:+.4f} (p={p_honest:.4f})")

    bins = [0, 0.1, 0.25, 0.5, 1.01]
    labels = ["<10%", "10-25%", "25-50%", ">50%"]
    df_corr = pd.DataFrame({
        "pct": pct_vals,
        "biased": biased_preds_test,
        "honest": honest_no_loc_preds,
    })
    df_corr["bin"] = pd.cut(df_corr["pct"], bins=bins, labels=labels)
    grouped = df_corr.groupby("bin")[["biased", "honest"]].mean()
    print("\n  Avg log_price prediction by minority percentage:")
    print(f"  {'Tract':>10}  {'Biased':>8}  {'Honest':>8}  {'Delta':>8}")
    for label, row in grouped.iterrows():
        delta = row["biased"] - row["honest"]
        print(f"  {str(label):>10}  {row['biased']:8.4f}  {row['honest']:8.4f}  {delta:+8.4f}")

    # ------------------------------------------------------------------ #
    # 13. Save outputs                                                    #
    # ------------------------------------------------------------------ #
    out_dir = RESULTS_DIR / "scaffold_no_location"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "adv_shap_values.npy", adv_shap)
    np.save(out_dir / "adv_lime_values.npy", adv_lime)
    np.save(out_dir / "feature_names.npy", np.array(feature_names))

    metrics = {
        "honest_no_loc_r2": float(honest_r2),
        "innocuous_model_r2": float(psi_r2),
        "n_features": len(feature_names),
        "features": feature_names,
        "lime": {
            "ood_accuracy": lime_scaffold.ood_diagnostics["accuracy"],
            **lime_fidelity,
        },
        "shap": {
            "ood_accuracy": shap_scaffold.ood_diagnostics["accuracy"],
            **shap_fidelity,
        },
        "n_explain": n_explain,
        "n_penalized_test": n_penalized,
        "pct_penalized_test": float(n_penalized / len(X_test)),
    }
    save_metrics(metrics, out_dir / "metrics.json")

    corr_metrics = {
        "pearson_r_biased_vs_pct_minority": float(r_biased),
        "pearson_p_biased": float(p_biased),
        "pearson_r_honest_vs_pct_minority": float(r_honest),
        "pearson_p_honest": float(p_honest),
    }
    save_metrics(corr_metrics, out_dir / "spatial_correlation.json")

    print(f"\nSaved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
