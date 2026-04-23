"""Run the adversarial scaffold attack end-to-end.

Builds biased + innocuous models, trains OOD detectors for LIME and SHAP,
generates adversarial explanations, and compares to honest baseline.
"""

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
from src.utils.io import load_model, save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


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
    # 1. Load configs
    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        config = yaml.safe_load(f)

    # 2. Load data with demographics
    X, y = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=config["features"],
        target_column=config["target"],
    )
    feature_names = list(X.columns)
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # 3. Train/test split (same seed as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Convert to numpy for the attack pipeline
    X_train_np = X_train.values
    X_test_np = X_test.values

    # 4. Load honest baseline model
    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Honest model loaded.")

    # 5. Build biased model
    baseline_features = [f for f in feature_names if f != "pct_black_latino"]
    base_feature_indices = [feature_names.index(f) for f in baseline_features]
    pct_col_idx = feature_names.index("pct_black_latino")

    f_biased = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=pct_col_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )
    print(f"Biased model: penalty={config['biased_model']['penalty']}, "
          f"threshold={config['biased_model']['threshold']}")

    # Check: how many test points get penalized?
    pct_vals = X_test_np[:, pct_col_idx]
    n_penalized = np.sum(pct_vals > config["biased_model"]["threshold"])
    print(f"  {n_penalized}/{len(X_test)} test points in high-minority tracts "
          f"({n_penalized/len(X_test)*100:.1f}%)")

    # 6. Build innocuous model
    safe_indices = get_safe_feature_indices(
        feature_names,
        exclude=["UTM_X", "UTM_Y", "pct_black_latino"],
    )
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])

    # Report innocuous model quality
    psi_preds = psi.predict(X_test_np)
    psi_r2 = r2_score(y_test, psi_preds)
    print(f"Innocuous model R²: {psi_r2:.4f} (trained on {len(safe_indices)} safe features)")

    # 7. Build LIME scaffold
    print("\nTraining LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(
        f_biased, psi, X_train_np, config["lime_ood"],
    )
    print(f"  OOD detector accuracy: {lime_scaffold.ood_diagnostics['accuracy']:.4f}")

    lime_fidelity = lime_scaffold.fidelity(X_test_np)
    print(f"  Fidelity on test data: r={lime_fidelity['pearson_r']:.4f}, "
          f"MSE={lime_fidelity['mse']:.6f}, "
          f"routed to biased={lime_fidelity['pct_routed_to_biased']*100:.1f}%")

    # 8. Build SHAP scaffold
    print("\nTraining SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, config["shap_ood"],
    )
    print(f"  OOD detector accuracy: {shap_scaffold.ood_diagnostics['accuracy']:.4f}")

    shap_fidelity = shap_scaffold.fidelity(X_test_np)
    print(f"  Fidelity on test data: r={shap_fidelity['pearson_r']:.4f}, "
          f"MSE={shap_fidelity['mse']:.6f}, "
          f"routed to biased={shap_fidelity['pct_routed_to_biased']*100:.1f}%")

    # 9. Generate adversarial explanations
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

    # 10. Load honest baseline explanations
    expl_dir = RESULTS_DIR / "explanations"
    honest_shap = np.load(expl_dir / "shap_values.npy")
    honest_lime = np.load(expl_dir / "lime_values.npy")
    honest_features = np.load(expl_dir / "feature_names.npy").tolist()

    # 11. Compare
    # Honest explanations have 10 features, adversarial have 11
    # For comparison, align on the shared 10 features
    shared_indices_adv = [feature_names.index(f) for f in honest_features]

    compare_attributions(
        honest_shap, adv_shap[:, shared_indices_adv],
        honest_features, "SHAP",
    )
    compare_attributions(
        honest_lime, adv_lime[:, shared_indices_adv],
        honest_features, "LIME",
    )

    # Also show pct_black_latino attribution in adversarial
    pct_idx = feature_names.index("pct_black_latino")
    print(f"\n  pct_black_latino mean |SHAP| in adversarial: {np.abs(adv_shap[:, pct_idx]).mean():.4f}")
    print(f"  pct_black_latino mean |LIME| in adversarial: {np.abs(adv_lime[:, pct_idx]).mean():.4f}")

    # Section A: GeoSHAP adversarial explanation
    utmx_idx = feature_names.index("UTM_X")
    utmy_idx = feature_names.index("UTM_Y")
    coords_train_np = X_train_np[:, [utmx_idx, utmy_idx]]
    coords_explain_np = X_explain_np[:, [utmx_idx, utmy_idx]]

    from src.explainers.geoshap_runner import explain_geoshap

    print("  GeoSHAP (adversarial, on shap_scaffold)...")
    adv_geo_shap = explain_geoshap(
        shap_scaffold,
        X_train,          # full 11-feature DataFrame
        coords_train_np,  # UTM only, shape (n_train, 2)
        X_explain,        # full 11-feature DataFrame
        coords_explain_np, # UTM only
        k=100, verbose=True,
    )
    compare_attributions(
        honest_shap[:, shared_indices_adv],  # use same alignment as SHAP compare
        adv_geo_shap[:, shared_indices_adv],
        honest_features, "GeoSHAP (adversarial)",
    )
    pct_geo_idx = feature_names.index("pct_black_latino")
    print(f"\n  pct_black_latino mean |GeoSHAP| adversarial: {np.abs(adv_geo_shap[:, pct_geo_idx]).mean():.4f}")

    # 12. Save outputs
    out_dir = RESULTS_DIR / "scaffold"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "adv_shap_values.npy", adv_shap)
    np.save(out_dir / "adv_lime_values.npy", adv_lime)
    np.save(out_dir / "adv_geoshap_values.npy", adv_geo_shap)

    metrics = {
        "innocuous_model_r2": float(psi_r2),
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
        "n_penalized_test": int(n_penalized),
        "pct_penalized_test": float(n_penalized / len(X_test)),
    }
    save_metrics(metrics, out_dir / "metrics.json")

    print(f"\nSaved to {out_dir}/")

    # Section B: Spatial correlation analysis
    print("\n--- Spatial Correlation Analysis ---")
    biased_preds_test = f_biased.predict(X_test_np)
    honest_preds_test = honest_model.predict(X_test_np[:, base_feature_indices])
    pct_vals_all = X_test_np[:, pct_col_idx]

    from scipy.stats import pearsonr
    r_biased, p_biased = pearsonr(pct_vals_all, biased_preds_test)
    r_honest, p_honest = pearsonr(pct_vals_all, honest_preds_test)
    print(f"  Pearson r(pct_minority, biased_pred):  {r_biased:+.4f} (p={p_biased:.4f})")
    print(f"  Pearson r(pct_minority, honest_pred):  {r_honest:+.4f} (p={p_honest:.4f})")

    bins = [0, 0.1, 0.25, 0.5, 1.01]
    labels = ["<10%", "10-25%", "25-50%", ">50%"]
    import pandas as pd
    df_corr = pd.DataFrame({
        "pct": pct_vals_all,
        "biased": biased_preds_test,
        "honest": honest_preds_test,
    })
    df_corr["bin"] = pd.cut(df_corr["pct"], bins=bins, labels=labels)
    grouped = df_corr.groupby("bin")[["biased", "honest"]].mean()
    print("\n  Avg log_price prediction by minority percentage:")
    print(f"  {'Tract':>10}  {'Biased':>8}  {'Honest':>8}  {'Delta':>8}")
    for label, row in grouped.iterrows():
        delta = row["biased"] - row["honest"]
        print(f"  {str(label):>10}  {row['biased']:8.4f}  {row['honest']:8.4f}  {delta:+8.4f}")

    corr_metrics = {
        "pearson_r_biased_vs_pct_minority": float(r_biased),
        "pearson_p_biased": float(p_biased),
        "pearson_r_honest_vs_pct_minority": float(r_honest),
        "pearson_p_honest": float(p_honest),
    }
    save_metrics(corr_metrics, RESULTS_DIR / "scaffold" / "spatial_correlation.json")
    print("\nSaved spatial_correlation.json")

    print("Done!")


if __name__ == "__main__":
    main()
