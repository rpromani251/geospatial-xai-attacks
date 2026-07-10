"""Experiment 10c — Parcel-scale flood zone scaffold attack.

Adapts run_scaffold_flood_zone.py to use the expanded KC Assessor parcel
dataset (59,490 records) via configs/scaffold_parcel_flood.yaml.

Key differences from Exp 10 (3k sample):
  - Dataset: data/processed/kc_housing_with_flood.csv (59,490 rows)
  - Baseline model: trained fresh on this dataset (same hyperparams as baseline.yaml)
  - Moran's I: subsampled to 5,000 pts (N² weight matrix infeasible on 59k)
  - GeoShapley n_explain=200 (reduce to 100 if runtime >10 min)

Outputs:  results/scaffold_parcel_flood/
  baseline_model.pkl, baseline_metrics.json
  morans_i.json
  adv_shap_values.npy, adv_lime_values.npy
  adv_geoshapley_{primary,geo,interaction,flat}.npy
  metrics.json, spatial_correlation.json
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks.biased_model import BiasedModel, InnocuousModel, get_safe_feature_indices
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.explainers.geoshap_runner import explain_geoshapley, geoshapley_to_flat, save_geoshapley
from src.utils.io import save_metrics
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


# ── Moran's I helpers ────────────────────────────────────────────────────────

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
            "n_permutations": n_perm, "k_neighbors": 8, "subsample_n": len(x)}


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
    t_start = time.time()
    np.random.seed(42)

    # 1. Load config
    config_path = CONFIGS_DIR / "scaffold_parcel_flood.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    trigger_col = config["biased_model"]["trigger_column"]
    print(f"Parcel-scale flood zone scaffold attack — trigger: '{trigger_col}'")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")

    # 2. Load parcel dataset
    parcel_path = ROOT / "data" / "processed" / "kc_housing_with_flood.csv"
    df_full = pd.read_csv(parcel_path)
    print(f"\nLoaded parcel dataset: {len(df_full)} rows")
    print(f"  in_floodplain: {df_full['in_floodplain'].sum()} positives "
          f"({df_full['in_floodplain'].mean()*100:.2f}%)")

    feature_cols = config["features"]
    target_col = config["target"]

    X_full = df_full[feature_cols]
    y_full = df_full[target_col]
    feature_names = list(X_full.columns)
    print(f"  Features: {feature_names}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  In-flood test points: {int(X_test['in_floodplain'].sum())}")

    X_train_np = X_train.values
    X_test_np = X_test.values

    # 4. Train baseline XGBoost on 10 features (no in_floodplain)
    base_features = [f for f in feature_names if f != trigger_col]
    base_feature_indices = [feature_names.index(f) for f in base_features]
    trigger_idx = feature_names.index(trigger_col)

    print(f"\n[Step 4] Training baseline XGBoost on {len(base_features)} features...")
    baseline_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    baseline_model.fit(X_train_np[:, base_feature_indices], y_train.values)

    base_preds = baseline_model.predict(X_test_np[:, base_feature_indices])
    r2 = r2_score(y_test, base_preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))
    mae = float(mean_absolute_error(y_test, base_preds))
    print(f"  Baseline R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    if r2 < 0.5:
        print(f"[STOP] Baseline R² = {r2:.4f} is below 0.5 — likely schema mismatch.")
        print("  Inspect feature distributions before proceeding.")
        return

    # Save baseline
    with open(out_dir / "baseline_model.pkl", "wb") as f:
        pickle.dump(baseline_model, f)

    baseline_metrics = {
        "r2": float(r2), "rmse": rmse, "mae": mae,
        "n_train": len(X_train), "n_test": len(X_test),
        "features": base_features,
    }
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"  Saved baseline_model.pkl, baseline_metrics.json")

    # 5. Moran's I for in_floodplain (subsample 5000)
    print(f"\n[Step 5] Computing Moran's I for in_floodplain (subsample 5000, 499 perms)...")
    rng_sub = np.random.default_rng(42)
    sub_idx = rng_sub.choice(len(X_full), size=5000, replace=False)
    coords_sub = X_full.values[sub_idx][:, [feature_names.index("UTM_X"), feature_names.index("UTM_Y")]]
    flood_sub = X_full.values[sub_idx, trigger_idx].astype(float)

    W_sub = build_knn_weights(coords_sub, k=8)
    morans_result = morans_i_permutation(flood_sub, W_sub, n_perm=499, seed=42)
    print(f"  Moran's I = {morans_result['morans_i']:+.4f}  z = {morans_result['z_score']:+.2f}  "
          f"p = {morans_result['p_value']:.4f}  (n=5000 subsample)")

    save_metrics(morans_result, out_dir / "morans_i.json")

    # 6. Biased model
    f_biased = BiasedModel(
        base_model=baseline_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=trigger_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )
    print(f"\n[Step 6] Biased model: penalty={config['biased_model']['penalty']}, "
          f"threshold={config['biased_model']['threshold']} on '{trigger_col}'")

    trigger_vals = X_test_np[:, trigger_idx]
    n_penalized = int(np.sum(trigger_vals > config["biased_model"]["threshold"]))
    pct_penalized = n_penalized / len(X_test)
    print(f"  {n_penalized}/{len(X_test)} test points in flood zone ({pct_penalized*100:.2f}%)")
    if pct_penalized < 0.03:
        print(f"  [WARN] Only {pct_penalized*100:.2f}% in floodplain — below 3% threshold.")

    # 7. Innocuous model
    safe_indices = get_safe_feature_indices(
        feature_names,
        exclude=["UTM_X", "UTM_Y", trigger_col],
    )
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])

    psi_preds = psi.predict(X_test_np)
    psi_r2 = r2_score(y_test, psi_preds)
    print(f"  Innocuous model R²: {psi_r2:.4f}")

    # 8. OOD scaffolds
    print("\n[Step 8] Training LIME OOD detector...")
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
    print(f"\n[Step 9] Generating adversarial explanations for {n_explain} test points...")

    print("  SHAP (adversarial)...")
    t0 = time.time()
    adv_shap = explain_shap(
        shap_scaffold, X_train, X_explain,
        n_background=config["explanation"]["shap_n_background"],
    )
    print(f"  SHAP done in {time.time()-t0:.0f}s")

    print("  LIME (adversarial)...")
    t0 = time.time()
    adv_lime = explain_lime(
        lime_scaffold, X_train, X_explain,
        num_samples=config["explanation"]["lime_num_samples"],
        random_state=config["explanation"]["lime_random_state"],
    )
    print(f"  LIME done in {time.time()-t0:.0f}s")

    print("  GeoShapley (adversarial)...")
    t0 = time.time()
    adv_geo_results = explain_geoshapley(shap_scaffold, X_train, X_explain)
    adv_geo_flat = geoshapley_to_flat(adv_geo_results)
    geo_time = time.time() - t0
    print(f"  GeoShapley done in {geo_time:.0f}s")

    # 10. Load honest baseline explanations (10-feature, from Exp 1)
    expl_dir = RESULTS_DIR / "explanations"
    honest_shap = np.load(expl_dir / "shap_values.npy")
    honest_lime = np.load(expl_dir / "lime_values.npy")
    honest_geo_flat = np.load(expl_dir / "geoshapley_flat.npy")
    honest_features = np.load(expl_dir / "feature_names.npy").tolist()
    print(f"\nHonest baseline loaded: {honest_shap.shape[0]} points, features={honest_features}")

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
        honest_geo_flat[:, :n_shared],
        adv_geo_flat[:, :n_shared],
        non_geo_honest, "GeoShapley (adversarial)",
    )

    # 11. Attribution table
    print("\n── Acid Test: Attribution Table ────────────────────────────────────────────")
    print(f"  {'Feature':>22}  {'Honest|SHAP|':>13}  {'Adv|SHAP|':>10}  "
          f"{'Adv|LIME|':>10}  {'Adv|GeoShapley|':>15}")
    print(f"  {'-'*22}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*15}")

    focus = [trigger_col, "sqft_living", "grade", "age", "pct_black_latino"]
    rows = []
    for feat in focus:
        if feat not in feature_names:
            continue
        fi = feature_names.index(feat)
        if feat in honest_features:
            hi = honest_features.index(feat)
            h_shap = np.abs(honest_shap[:, hi]).mean()
        else:
            h_shap = float("nan")
        a_shap = np.abs(adv_shap[:, fi]).mean()
        a_lime = np.abs(adv_lime[:, fi]).mean()
        # GeoShapley flat: non-geo features indexed through non_geo_adv
        if feat in non_geo_adv:
            a_geo = np.abs(adv_geo_flat[:, non_geo_adv.index(feat)]).mean()
        else:
            a_geo = float("nan")
        rows.append({
            "feature": feat,
            "honest_shap": float(h_shap) if not np.isnan(h_shap) else None,
            "adv_shap": float(a_shap),
            "adv_lime": float(a_lime),
            "adv_geoshapley": float(a_geo) if not np.isnan(a_geo) else None,
        })
        h_str = f"{h_shap:.4f}" if not np.isnan(h_shap) else "   —  "
        a_geo_str = f"{a_geo:.4f}" if not np.isnan(a_geo) else "   —  "
        print(f"  {feat:>22}  {h_str:>13}  {a_shap:10.4f}  {a_lime:10.4f}  {a_geo_str:>15}")

    # GeoShapley reversal check (compare to Exp 10 / Exp 3 benchmarks)
    # geo_effect is the last column of the flat array
    geo_effect_col = adv_geo_flat.shape[1] - 1
    geo_effect_mean = np.abs(adv_geo_flat[:, geo_effect_col]).mean()
    trigger_shap = np.abs(adv_shap[:, trigger_idx]).mean()
    if trigger_col in non_geo_adv:
        trigger_geo = np.abs(adv_geo_flat[:, non_geo_adv.index(trigger_col)]).mean()
    else:
        trigger_geo = float("nan")

    print(f"\n  GeoShapley reversal check:")
    print(f"    {trigger_col} |GeoShapley| = {trigger_geo:.4f}  vs  |SHAP| = {trigger_shap:.4f}")
    print(f"    geo_effect (pure location) = {geo_effect_mean:.4f}")

    geoshap_reversal = bool(not np.isnan(trigger_geo) and trigger_geo > trigger_shap)
    if geoshap_reversal:
        reversal_verdict = "REVERSAL CONFIRMED — GeoShapley exposes bias more than SHAP."
    elif not np.isnan(trigger_geo) and trigger_geo < trigger_shap * 0.5:
        reversal_verdict = "NO REVERSAL — GeoShapley still hides the bias."
    else:
        reversal_verdict = "PARTIAL — GeoShapley neither strongly conceals nor reveals."
    print(f"    Verdict: {reversal_verdict}")

    # 12. Save NPY outputs
    np.save(out_dir / "adv_shap_values.npy", adv_shap)
    np.save(out_dir / "adv_lime_values.npy", adv_lime)
    save_geoshapley(adv_geo_results, out_dir, prefix="adv_geoshapley")

    # metrics.json
    metrics = {
        "config": str(config_path),
        "dataset": "parcel_flood",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_r2": float(r2),
        "baseline_rmse": rmse,
        "trigger_column": trigger_col,
        "n_penalized_test": n_penalized,
        "pct_penalized_test": float(pct_penalized),
        "innocuous_model_r2": float(psi_r2),
        "lime": {
            "ood_accuracy": lime_scaffold.ood_diagnostics["accuracy"],
            **lime_fidelity,
        },
        "shap": {
            "ood_accuracy": shap_scaffold.ood_diagnostics["accuracy"],
            **shap_fidelity,
        },
        "n_explain": n_explain,
        "attribution_table": rows,
        "geoshapley_reversal": geoshap_reversal,
        "geoshapley_reversal_verdict": reversal_verdict,
        "geo_effect_mean": float(geo_effect_mean),
        "runtime_seconds": float(time.time() - t_start),
    }
    save_metrics(metrics, out_dir / "metrics.json")

    # spatial_correlation.json
    biased_preds_test = f_biased.predict(X_test_np)
    honest_preds_test = baseline_model.predict(X_test_np[:, base_feature_indices])
    trigger_vals_test = X_test_np[:, trigger_idx]
    nonzero_mask = trigger_vals_test > 0

    r_biased, p_biased = pearsonr(trigger_vals_test, biased_preds_test)
    r_honest, p_honest = pearsonr(trigger_vals_test, honest_preds_test)
    print(f"\n  Pearson r(in_floodplain, biased_pred):  {r_biased:+.4f} (p={p_biased:.4f})")
    print(f"  Pearson r(in_floodplain, honest_pred):  {r_honest:+.4f} (p={p_honest:.4f})")

    corr_metrics = {
        "pearson_r_biased_vs_floodzone": float(r_biased),
        "pearson_p_biased": float(p_biased),
        "pearson_r_honest_vs_floodzone": float(r_honest),
        "pearson_p_honest": float(p_honest),
        "mean_pred_in_flood": {
            "biased": float(biased_preds_test[nonzero_mask].mean()) if nonzero_mask.any() else None,
            "honest": float(honest_preds_test[nonzero_mask].mean()) if nonzero_mask.any() else None,
        },
        "mean_pred_outside_flood": {
            "biased": float(biased_preds_test[~nonzero_mask].mean()),
            "honest": float(honest_preds_test[~nonzero_mask].mean()),
        },
    }
    save_metrics(corr_metrics, out_dir / "spatial_correlation.json")

    total_time = time.time() - t_start
    print(f"\n[DONE] Total runtime: {total_time:.0f}s")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
