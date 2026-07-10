"""Run the adversarial scaffold attack with the FEMA flood zone bias feature.

Mirrors run_scaffold_attack.py but uses configs/flood_zone.yaml and
data/processed/seattle_housing_with_flood_zone.csv.

Outputs saved to results/scaffold_flood_zone/:
  adv_shap_values.npy
  adv_lime_values.npy
  adv_geoshapley_*.npy
  metrics.json
  spatial_correlation.json
  morans_i.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

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


# ── Moran's I helpers (mirrored from run_morans_i.py) ────────────────────────

def build_knn_weights(coords: np.ndarray, k: int = 8) -> np.ndarray:
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]
    W = np.zeros((n, n))
    for i, nbr_idx in enumerate(indices):
        W[i, nbr_idx] = 1.0 / k
    return W


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    z = x - x.mean()
    denom = z @ z
    if denom == 0:
        return 0.0
    return float((z @ W @ z) / denom)


def morans_i_permutation(x: np.ndarray, W: np.ndarray, n_perm: int = 499, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    observed = morans_i(x, W)
    perm_vals = np.array([morans_i(rng.permutation(x), W) for _ in range(n_perm)])
    mu = perm_vals.mean()
    sigma = perm_vals.std(ddof=1)
    z = (observed - mu) / sigma if sigma > 0 else 0.0
    p = float((np.sum(perm_vals >= observed) + 1) / (n_perm + 1))
    return {"morans_i": float(observed), "z_score": float(z), "p_value": p,
            "n_permutations": n_perm, "k_neighbors": 8}


# ── Attribution comparison ────────────────────────────────────────────────────

def compare_attributions(honest, adversarial, feature_names, label):
    mean_honest = np.abs(honest).mean(axis=0)
    mean_adv = np.abs(adversarial).mean(axis=0)
    rank_honest = np.argsort(mean_honest)[::-1]
    rank_adv = np.argsort(mean_adv)[::-1]

    print(f"\n{label} Feature Importance Comparison:")
    print(f"  {'Feature':>22}  {'Honest':>10}  {'Adversarial':>12}  {'Shift':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*8}")
    for idx in rank_honest:
        name = feature_names[idx]
        h = mean_honest[idx]
        a = mean_adv[idx] if idx < len(mean_adv) else 0.0
        shift = a - h
        print(f"  {name:>22}  {h:10.4f}  {a:12.4f}  {shift:+8.4f}")

    print(f"\n  Honest top-3:      {[feature_names[i] for i in rank_honest[:3]]}")
    print(f"  Adversarial top-3: {[feature_names[i] for i in rank_adv[:3]]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # 1. Load config
    config_path = CONFIGS_DIR / "flood_zone.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trigger_col = config["biased_model"]["trigger_column"]  # "in_floodplain"
    print(f"Flood zone scaffold attack — trigger: '{trigger_col}'")
    print(f"Config: {config_path}")

    # 2. Load data
    X, y = load_housing(
        dataset=config["data"]["dataset"],
        feature_columns=config["features"],
        target_column=config["target"],
    )
    feature_names = list(X.columns)
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    X_train_np = X_train.values
    X_test_np = X_test.values

    # 4. Load honest baseline model (trained on 10-feature raw data, re-use it)
    baseline_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")

    # The baseline model uses the original 10 features (no flood zone).
    # Extract indices that align with those features (all except in_floodplain).
    base_features = [f for f in feature_names if f != trigger_col]
    base_feature_indices = [feature_names.index(f) for f in base_features]
    trigger_idx = feature_names.index(trigger_col)

    # Wrap the baseline model to accept 10-feature slice
    # (same pattern as run_scaffold_attack.py)
    print("Honest model loaded (baseline 10-feature XGBoost).")

    # 5. Build biased model
    f_biased = BiasedModel(
        base_model=baseline_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=trigger_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )
    print(f"Biased model: penalty={config['biased_model']['penalty']}, "
          f"threshold={config['biased_model']['threshold']} on '{trigger_col}'")

    # Coverage check
    trigger_vals = X_test_np[:, trigger_idx]
    n_penalized = int(np.sum(trigger_vals > config["biased_model"]["threshold"]))
    pct_penalized = n_penalized / len(X_test)
    print(f"  {n_penalized}/{len(X_test)} test points in flood zone ({pct_penalized*100:.1f}%)")
    if pct_penalized < 0.03:
        print(f"  [WARN] Only {pct_penalized*100:.1f}% in floodplain — below 3% threshold.")

    # 6. Build innocuous model (safe features only, excludes UTM + trigger)
    safe_indices = get_safe_feature_indices(
        feature_names,
        exclude=["UTM_X", "UTM_Y", trigger_col],
    )
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])

    psi_preds = psi.predict(X_test_np)
    psi_r2 = r2_score(y_test, psi_preds)
    print(f"Innocuous model R²: {psi_r2:.4f}")

    # 7. Compute Moran's I for in_floodplain on the full dataset
    print("\nComputing Moran's I for in_floodplain (full dataset, 8-NN, 499 permutations)...")
    all_coords = X.values[:, [feature_names.index("UTM_X"), feature_names.index("UTM_Y")]]
    flood_vals = X[trigger_col].values.astype(float)
    W_full = build_knn_weights(all_coords, k=8)
    morans_result = morans_i_permutation(flood_vals, W_full, n_perm=499, seed=42)
    print(f"  Moran's I = {morans_result['morans_i']:+.4f}  z = {morans_result['z_score']:+.2f}  "
          f"p = {morans_result['p_value']:.4f}")

    if morans_result['morans_i'] < 0.3:
        print("  [OK] in_floodplain has low spatial autocorrelation (I < 0.3) — hypothesis supported.")
    else:
        print("  [NOTE] in_floodplain has moderate/high spatial autocorrelation.")

    # 8. Build OOD scaffolds
    print("\nTraining LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(
        f_biased, psi, X_train_np, config["lime_ood"],
    )
    print(f"  OOD accuracy: {lime_scaffold.ood_diagnostics['accuracy']:.4f}")
    lime_fidelity = lime_scaffold.fidelity(X_test_np)
    print(f"  Fidelity: r={lime_fidelity['pearson_r']:.4f}, "
          f"routed={lime_fidelity['pct_routed_to_biased']*100:.1f}%")

    print("\nTraining SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, config["shap_ood"],
    )
    print(f"  OOD accuracy: {shap_scaffold.ood_diagnostics['accuracy']:.4f}")
    shap_fidelity = shap_scaffold.fidelity(X_test_np)
    print(f"  Fidelity: r={shap_fidelity['pearson_r']:.4f}, "
          f"routed={shap_fidelity['pct_routed_to_biased']*100:.1f}%")

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

    print("  GeoShapley (adversarial)...")
    adv_geo_results = explain_geoshapley(shap_scaffold, X_train, X_explain)
    adv_geo_flat = geoshapley_to_flat(adv_geo_results)

    # 10. Load honest baseline explanations (10-feature)
    expl_dir = RESULTS_DIR / "explanations"
    honest_shap = np.load(expl_dir / "shap_values.npy")
    honest_lime = np.load(expl_dir / "lime_values.npy")
    honest_features = np.load(expl_dir / "feature_names.npy").tolist()

    # Align adversarial (11-feature) to honest (10-feature) shared features
    shared_indices_adv = [feature_names.index(f) for f in honest_features]

    compare_attributions(
        honest_shap, adv_shap[:, shared_indices_adv],
        honest_features, "SHAP",
    )
    compare_attributions(
        honest_lime, adv_lime[:, shared_indices_adv],
        honest_features, "LIME",
    )
    # GeoShapley comparison: non-geo features only
    non_geo_honest = [f for f in honest_features if f not in ("UTM_X", "UTM_Y")]
    non_geo_adv = [f for f in feature_names if f not in ("UTM_X", "UTM_Y")]
    n_shared = len(non_geo_honest)
    compare_attributions(
        np.load(expl_dir / "geoshapley_flat.npy")[:, :n_shared],
        adv_geo_flat[:, :n_shared],
        non_geo_honest, "GeoShapley (adversarial)",
    )

    # Key: attribution on trigger column itself
    trigger_flat_idx = non_geo_adv.index(trigger_col)
    print(f"\n  {trigger_col} mean |SHAP| (adversarial):         {np.abs(adv_shap[:, trigger_idx]).mean():.4f}")
    print(f"  {trigger_col} mean |LIME| (adversarial):         {np.abs(adv_lime[:, trigger_idx]).mean():.4f}")
    print(f"  {trigger_col} mean |GeoShapley| (adversarial):   {np.abs(adv_geo_flat[:, trigger_flat_idx]).mean():.4f}")

    print(f"\n  geo_effect mean |GeoShapley| adversarial: {np.abs(adv_geo_flat[:, -1]).mean():.4f}")

    # 11. Key comparison table (acid test)
    print("\n── Acid Test: Attribution Table ────────────────────────────────────────────")
    print(f"  {'Feature':>22}  {'Honest|SHAP|':>13}  {'Adv|SHAP|':>10}  "
          f"{'Adv|LIME|':>10}  {'Adv|GeoShapley|':>13}")
    print(f"  {'-'*22}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*13}")

    focus = [trigger_col, "geo_effect", "sqft_living", "grade", "age"]
    rows = []
    for feat in focus:
        if feat == "geo_effect":
            a_geo = np.abs(adv_geo_flat[:, -1]).mean()
            rows.append({
                "feature": feat,
                "honest_shap": float("nan"),
                "adv_shap": float("nan"),
                "adv_lime": float("nan"),
                "adv_geoshapley": a_geo,
            })
            print(f"  {'geo_effect':>22}  {'   —  ':>13}  {'   —   ':>10}  {'   —   ':>10}  {'   ' + f'{a_geo:.4f}':>13}")
        elif feat in feature_names:
            fi = feature_names.index(feat)
            if feat in honest_features:
                hi = honest_features.index(feat)
                h_shap = np.abs(honest_shap[:, hi]).mean()
            else:
                h_shap = float("nan")
            a_shap = np.abs(adv_shap[:, fi]).mean()
            a_lime = np.abs(adv_lime[:, fi]).mean()
            a_geo = np.abs(adv_geo_flat[:, non_geo_adv.index(feat)]).mean() if feat in non_geo_adv else float("nan")
            rows.append({
                "feature": feat,
                "honest_shap": h_shap,
                "adv_shap": a_shap,
                "adv_lime": a_lime,
                "adv_geoshapley": a_geo,
            })
            h_str = f"{h_shap:.4f}" if not np.isnan(h_shap) else "   —  "
            a_geo_str = f"{a_geo:13.4f}" if not np.isnan(a_geo) else "       —     "
            print(f"  {feat:>22}  {h_str:>13}  {a_shap:10.4f}  {a_lime:10.4f}  {a_geo_str}")

    # GeoShapley reversal check
    trigger_geo = np.abs(adv_geo_flat[:, trigger_flat_idx]).mean()
    trigger_shap = np.abs(adv_shap[:, trigger_idx]).mean()
    print(f"\n  GeoShapley reversal check:")
    print(f"    {trigger_col} |GeoShapley| = {trigger_geo:.4f}  vs  |SHAP| = {trigger_shap:.4f}")
    if trigger_geo > trigger_shap:
        print("    [REVERSAL CONFIRMED] GeoShapley exposes the bias more than SHAP.")
    elif trigger_geo < trigger_shap * 0.5:
        print("    [NO REVERSAL] GeoShapley still hides the bias.")
    else:
        print("    [PARTIAL] GeoShapley neither strongly conceals nor reveals the bias.")

    # 12. Save outputs
    out_dir = RESULTS_DIR / "scaffold_flood_zone"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "adv_shap_values.npy", adv_shap)
    np.save(out_dir / "adv_lime_values.npy", adv_lime)
    save_geoshapley(adv_geo_results, out_dir, prefix="adv_geoshapley")

    # metrics.json
    metrics = {
        "config": str(config_path),
        "trigger_column": trigger_col,
        "innocuous_model_r2": float(psi_r2),
        "n_penalized_test": n_penalized,
        "pct_penalized_test": float(pct_penalized),
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
        "attribution_table": rows,
        "geoshap_reversal": bool(trigger_geo > trigger_shap),
    }
    save_metrics(metrics, out_dir / "metrics.json")

    # spatial_correlation.json
    biased_preds_test = f_biased.predict(X_test_np)
    honest_preds_test = baseline_model.predict(X_test_np[:, base_feature_indices])
    trigger_vals_test = X_test_np[:, trigger_idx]

    # Only correlate with flood zone binary (biserial-like with Pearson)
    nonzero_mask = trigger_vals_test > 0
    r_biased, p_biased = pearsonr(trigger_vals_test, biased_preds_test)
    r_honest, p_honest = pearsonr(trigger_vals_test, honest_preds_test)
    print(f"\n  Pearson r(in_floodplain, biased_pred):  {r_biased:+.4f} (p={p_biased:.4f})")
    print(f"  Pearson r(in_floodplain, honest_pred):  {r_honest:+.4f} (p={p_honest:.4f})")

    # Mean prediction by flood zone status
    df_corr = pd.DataFrame({
        "flood": trigger_vals_test,
        "biased": biased_preds_test,
        "honest": honest_preds_test,
    })
    grouped = df_corr.groupby("flood")[["biased", "honest"]].mean()
    print("\n  Mean log_price prediction by flood zone:")
    print(f"  {'Zone':>10}  {'Biased':>8}  {'Honest':>8}  {'Delta':>8}")
    for zone_val, row in grouped.iterrows():
        label = "in flood" if zone_val > 0 else "outside"
        delta = row["biased"] - row["honest"]
        print(f"  {label:>10}  {row['biased']:8.4f}  {row['honest']:8.4f}  {delta:+8.4f}")

    corr_metrics = {
        "pearson_r_biased_vs_floodzone": float(r_biased),
        "pearson_p_biased": float(p_biased),
        "pearson_r_honest_vs_floodzone": float(r_honest),
        "pearson_p_honest": float(p_honest),
        "mean_pred_in_flood": {
            "biased": float(df_corr[df_corr["flood"] > 0]["biased"].mean()) if nonzero_mask.any() else None,
            "honest": float(df_corr[df_corr["flood"] > 0]["honest"].mean()) if nonzero_mask.any() else None,
        },
        "mean_pred_outside_flood": {
            "biased": float(df_corr[df_corr["flood"] == 0]["biased"].mean()),
            "honest": float(df_corr[df_corr["flood"] == 0]["honest"].mean()),
        },
    }
    save_metrics(corr_metrics, out_dir / "spatial_correlation.json")

    # morans_i.json
    save_metrics(morans_result, out_dir / "morans_i.json")

    print(f"\nSaved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
