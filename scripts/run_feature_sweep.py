"""Scaffold attack sweep across 5 features spanning the Moran's I spectrum.

For each candidate feature (read from results/morans_i/sweep_features.txt, or
overridden via --features), builds a BiasedModel that penalizes instances where
that feature exceeds its 75th-percentile threshold, trains LIME + SHAP OOD
detectors, generates adversarial explanations, and records how much each
explainer's attribution of the biased feature is suppressed.

Run from repo root:
    python3 scripts/run_feature_sweep.py
    python3 scripts/run_feature_sweep.py --features pct_black_latino grade waterfront

Outputs:
    results/feature_sweep/sweep_summary.json   — per-feature metrics
    results/feature_sweep/sweep_plot.png        — suppression vs Moran's I
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.shap_runner import explain_shap
from src.explainers.lime_runner import explain_lime
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR

MORANS_DIR = RESULTS_DIR / "morans_i"
SWEEP_DIR = RESULTS_DIR / "feature_sweep"
SWEEP_DIR_EXTENDED = RESULTS_DIR / "feature_sweep_a2"  # Exp 8b outputs

# Exp 8b default feature set. 8 features spanning Moran's I ∈ [0.05, 0.85]:
# 6 structural (waterfront, condition, view, sqft_living, grade, sqft_lot)
# plus 2 demographic (pct_latino, pct_black). `pct_black_latino` is Exp 3's
# canonical trigger and is handled separately; `age` is excluded for
# parallelism with the original 5-feature sweep.
EXTENDED_FEATURES = [
    "waterfront", "condition", "view", "sqft_living",
    "grade", "sqft_lot", "pct_latino", "pct_black",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sweep_features(override: list[str] | None) -> list[str]:
    if override:
        return override
    features_file = MORANS_DIR / "sweep_features.txt"
    if not features_file.exists():
        raise FileNotFoundError(
            f"{features_file} not found. Run run_morans_i.py first, "
            "or pass --features explicitly."
        )
    return [l.strip() for l in features_file.read_text().splitlines() if l.strip()]


def load_morans_i_map() -> dict[str, float]:
    json_path = MORANS_DIR / "morans_i_results.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        data = json.load(f)
    return {r["feature"]: r["morans_i"] for r in data["all_features"]}


def attribution_suppression(honest: np.ndarray, adversarial: np.ndarray, feat_idx: int) -> float:
    """Fractional drop in mean |attribution| for feat_idx under the scaffold.

    Positive = suppression (attribution went down).
    """
    h = np.abs(honest[:, feat_idx]).mean()
    a = np.abs(adversarial[:, feat_idx]).mean()
    if h == 0:
        return 0.0
    return float((h - a) / h)


def attribution_visibility(adversarial: np.ndarray, feat_idx: int) -> float:
    """Share of total |attribution| assigned to feat_idx by the adversarial explainer.

    visibility = mean_i(|adv[i, feat]|) / mean_i(sum_j|adv[i, j]|)

    Well-defined whether or not feat is in the honest baseline: 0 means the
    biased feature is invisible in the adversarial explanation; higher means
    the attack leaks. Replaces the suppression metric for demographic features
    (where honest attr = 0 makes suppression degenerate to 0).
    """
    feat_attr = np.abs(adversarial[:, feat_idx]).mean()
    total_attr = np.abs(adversarial).sum(axis=1).mean()
    if total_attr == 0:
        return 0.0
    return float(feat_attr / total_attr)


# ── Single-feature scaffold run ───────────────────────────────────────────────

def run_scaffold_for_feature(
    bias_feature: str,
    df: pd.DataFrame,
    y: pd.Series,
    honest_model,
    base_config: dict,
    baseline_features: list[str],
    n_explain: int = 100,
) -> dict:
    """Run full scaffold attack for one biased feature. Returns metrics dict."""
    all_feature_cols = list(df.columns)

    if bias_feature not in df.columns:
        raise ValueError(f"Feature '{bias_feature}' not in dataset columns: {list(df.columns)}")

    # Build a focused dataset: exactly the 10 baseline features + bias feature (if not
    # already in baseline). This prevents passing extraneous demographic columns to
    # SHAP / XGBoost, which causes shape mismatches when models expect different sizes.
    if bias_feature in baseline_features:
        sweep_cols = list(baseline_features)          # bias feature already present
    else:
        sweep_cols = list(baseline_features) + [bias_feature]  # append demographic col

    df_sweep = df[sweep_cols]
    sweep_feature_cols = list(df_sweep.columns)

    # Threshold at 75th percentile of the bias feature
    threshold = float(np.percentile(df_sweep[bias_feature].values, 75))
    penalty = base_config["biased_model"]["penalty"]

    print(f"\n  ── Feature: {bias_feature}  (threshold={threshold:.4f}, penalty={penalty}) ──")

    X_train, X_test, y_train, y_test = train_test_split(
        df_sweep, y, test_size=base_config["data"]["test_size"],
        random_state=base_config["data"]["random_state"],
    )
    X_train_np = X_train.values
    X_test_np = X_test.values

    bias_col_idx = sweep_feature_cols.index(bias_feature)
    # The honest base model always receives all 10 baseline features — never fewer.
    # (If bias feature is structural it stays in the base model input; the penalty
    # is applied on top, not by removing it from predictions.)
    base_feature_indices = [sweep_feature_cols.index(f) for f in baseline_features]

    # Biased model: penalize instances above threshold
    f_biased = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=bias_col_idx,
        threshold=threshold,
        penalty=penalty,
    )
    n_penalized = int(np.sum(X_test_np[:, bias_col_idx] > threshold))
    print(f"    Penalized test pts: {n_penalized}/{len(X_test)} ({n_penalized/len(X_test)*100:.1f}%)")

    # Innocuous model: all sweep features except the biased one and UTM coords
    safe_indices = [i for i, f in enumerate(sweep_feature_cols)
                    if f not in {bias_feature, "UTM_X", "UTM_Y"}]
    psi = InnocuousModel(safe_col_indices=safe_indices)
    psi.fit(X_train_np, y_train.values, base_config["innocuous_model"])
    psi_r2 = r2_score(y_test, psi.predict(X_test_np))
    print(f"    Innocuous model R²: {psi_r2:.4f}")

    # SHAP scaffold
    print("    Training SHAP OOD detector...")
    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, base_config["shap_ood"],
    )
    shap_ood_acc = shap_scaffold.ood_diagnostics["accuracy"]
    print(f"    SHAP OOD accuracy: {shap_ood_acc:.4f}")

    # LIME scaffold
    print("    Training LIME OOD detector...")
    lime_scaffold = AdversarialScaffold.for_lime(
        f_biased, psi, X_train_np, base_config["lime_ood"],
    )
    lime_ood_acc = lime_scaffold.ood_diagnostics["accuracy"]
    print(f"    LIME OOD accuracy: {lime_ood_acc:.4f}")

    # Generate adversarial explanations
    X_explain = X_test.iloc[:n_explain]
    X_train_df = pd.DataFrame(X_train_np, columns=sweep_feature_cols)
    X_explain_df = pd.DataFrame(X_explain.values, columns=sweep_feature_cols)

    print(f"    Running SHAP adversarial ({n_explain} pts)...")
    adv_shap = explain_shap(
        shap_scaffold, X_train_df, X_explain_df,
        n_background=base_config["explanation"]["shap_n_background"],
    )

    print(f"    Running LIME adversarial ({n_explain} pts)...")
    adv_lime = explain_lime(
        lime_scaffold, X_train_df, X_explain_df,
        num_samples=base_config["explanation"]["lime_num_samples"],
        random_state=base_config["explanation"]["lime_random_state"],
    )

    # Honest baseline: TreeSHAP on the 10 baseline features
    # CAVEAT (2026-04-23): we only build a TreeSHAP honest baseline here,
    # not an honest LIME baseline. The lime_supp computation below reuses
    # honest_shap_full, which makes lime_supp a cross-method comparison
    # (honest TreeSHAP vs adversarial LIME) rather than a within-method
    # suppression (honest LIME vs adversarial LIME). Do not interpret
    # lime_supp as a LIME-specific effect; use lime_vis (visibility) as
    # the headline LIME metric instead. shap_supp is a valid SHAP
    # before/after.
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(honest_model)
    honest_shap_vals = explainer.shap_values(X_explain_df[baseline_features])
    # Pad to sweep_feature_cols width so indices align with adv_shap / adv_lime
    honest_shap_full = np.zeros((n_explain, len(sweep_feature_cols)))
    baseline_mask = [sweep_feature_cols.index(f) for f in baseline_features]
    honest_shap_full[:, baseline_mask] = honest_shap_vals

    shap_supp = attribution_suppression(honest_shap_full, adv_shap, bias_col_idx)
    lime_supp = attribution_suppression(honest_shap_full, adv_lime, bias_col_idx)  # NB: cross-method baseline; see caveat above

    shap_vis = attribution_visibility(adv_shap, bias_col_idx)
    lime_vis = attribution_visibility(adv_lime, bias_col_idx)

    adv_shap_mean = float(np.abs(adv_shap[:, bias_col_idx]).mean())
    adv_lime_mean = float(np.abs(adv_lime[:, bias_col_idx]).mean())

    feature_role = "structural" if bias_feature in baseline_features else "demographic"

    print(f"    Adv SHAP |attr| for {bias_feature}: {adv_shap_mean:.4f}  "
          f"(vis={shap_vis:.3f}, suppression={shap_supp:+.3f})")
    print(f"    Adv LIME |attr| for {bias_feature}: {adv_lime_mean:.4f}  "
          f"(vis={lime_vis:.3f}, suppression={lime_supp:+.3f})")

    return {
        "feature": bias_feature,
        "feature_role": feature_role,
        "threshold_p75": threshold,
        "pct_penalized_test": round(n_penalized / len(X_test), 4),
        "innocuous_r2": round(psi_r2, 4),
        "shap_ood_accuracy": round(shap_ood_acc, 4),
        "lime_ood_accuracy": round(lime_ood_acc, 4),
        "adv_shap_mean_attr_biased": round(adv_shap_mean, 4),
        "adv_lime_mean_attr_biased": round(adv_lime_mean, 4),
        "shap_visibility": round(shap_vis, 4),
        "lime_visibility": round(lime_vis, 4),
        "shap_suppression": round(shap_supp, 4),
        "lime_suppression": round(lime_supp, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", default=None,
                   help="Override feature list (space-separated). "
                        "Default: read from results/morans_i/sweep_features.txt")
    p.add_argument("--n_explain", type=int, default=100,
                   help="Points to explain per feature (default 100; full=200 is slower)")
    p.add_argument("--extended", action="store_true",
                   help="Run Exp 8b: 8 features (waterfront, condition, view, "
                        "sqft_living, grade, sqft_lot, pct_latino, pct_black), "
                        "add normalized_visibility_* columns, and write to "
                        "results/feature_sweep_a2/ instead of results/feature_sweep/. "
                        "`--features` and `--n_explain` still apply.")
    return p.parse_args()


def honest_importance_for(bias_feature: str, X_explain_df: pd.DataFrame,
                          honest_model) -> float:
    """Mean |honest TreeSHAP attribution| on the biased feature itself.

    Used to normalize visibility so the I-trend is separated from the
    bias feature's natural predictive importance (Exp 8b). The honest
    baseline was trained on the 10 baseline features; `bias_feature` is
    either in that list (structural → nonzero honest_importance) or
    outside it (demographic → zero, because the honest model doesn't
    see the feature).
    """
    if bias_feature not in X_explain_df.columns:
        return 0.0
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(honest_model)
    baseline_cols = list(honest_model.get_booster().feature_names) \
        if hasattr(honest_model, "get_booster") else list(X_explain_df.columns)
    if bias_feature not in baseline_cols:
        return 0.0
    honest_vals = explainer.shap_values(X_explain_df[baseline_cols])
    bias_idx = baseline_cols.index(bias_feature)
    return float(np.abs(honest_vals[:, bias_idx]).mean())


def main():
    args = parse_args()
    np.random.seed(42)

    # Load base config from scaffold.yaml (reuse OOD / innocuous settings)
    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        base_config = yaml.safe_load(f)

    # Load baseline feature list — the honest model was trained on exactly these 10 features
    with open(CONFIGS_DIR / "baseline.yaml") as f:
        baseline_config = yaml.safe_load(f)
    baseline_features = baseline_config["features"]

    # Load full dataset (with demographics)
    X_full, y = load_housing(
        dataset="demographics",
        feature_columns=None,
        target_column="log_price",
    )
    # Drop non-numeric / identifier columns that aren't features
    drop_cols = {"tract_id"}
    X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns])
    print(f"Full dataset: {X_full.shape[0]} rows, {X_full.shape[1]} features")
    print(f"  Columns: {list(X_full.columns)}")

    # Load honest baseline model
    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Honest baseline model loaded.")

    # Load sweep features.  --extended wins over --features wins over
    # results/morans_i/sweep_features.txt.
    if args.extended and args.features is None:
        sweep_features = EXTENDED_FEATURES
        out_dir = SWEEP_DIR_EXTENDED
        summary_name = "combined_8feature_summary.json"
        print("\n--- EXTENDED mode (Exp 8b) ---")
    else:
        sweep_features = load_sweep_features(args.features)
        out_dir = SWEEP_DIR_EXTENDED if args.extended else SWEEP_DIR
        summary_name = "combined_8feature_summary.json" if args.extended else "sweep_summary.json"

    morans_map = load_morans_i_map()
    print(f"\nSweep features: {sweep_features}")
    for f in sweep_features:
        I = morans_map.get(f, "N/A")
        print(f"  {f}: Moran's I = {I}")

    # Run sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_results = []

    # Load X_explain once for honest_importance computation (matches what
    # run_scaffold_for_feature uses internally: first n_explain rows of
    # the test split under the same 80/20 random_state=42 split).
    X_train_explain, X_test_explain, _, _ = train_test_split(
        X_full, y,
        test_size=base_config["data"]["test_size"],
        random_state=base_config["data"]["random_state"],
    )
    X_explain_honest = X_test_explain.iloc[:args.n_explain]

    for feat in sweep_features:
        try:
            metrics = run_scaffold_for_feature(
                bias_feature=feat,
                df=X_full,
                y=y,
                honest_model=honest_model,
                base_config=base_config,
                baseline_features=baseline_features,
                n_explain=args.n_explain,
            )
            metrics["morans_i"] = morans_map.get(feat, None)

            if args.extended:
                # Compute normalized-visibility (Exp 8b): visibility /
                # honest_importance. Divides out the biased feature's
                # natural predictive importance in the honest baseline.
                hi = honest_importance_for(feat, X_explain_honest, honest_model)
                metrics["honest_importance"] = round(hi, 6)
                metrics["delta_shap_importance"] = round(
                    metrics["adv_shap_mean_attr_biased"] - hi, 6)
                metrics["delta_lime_importance"] = round(
                    metrics["adv_lime_mean_attr_biased"] - hi, 6)
                if hi > 0:
                    metrics["normalized_visibility_shap"] = round(
                        metrics["adv_shap_mean_attr_biased"] / hi, 4)
                    metrics["normalized_visibility_lime"] = round(
                        metrics["adv_lime_mean_attr_biased"] / hi, 4)
                else:
                    # Demographic feature: not in honest model → no
                    # natural attribution. Normalized-visibility is
                    # undefined; any adversarial signal is pure leakage.
                    metrics["normalized_visibility_shap"] = None
                    metrics["normalized_visibility_lime"] = None

            sweep_results.append(metrics)
        except Exception as e:
            print(f"  ERROR on {feat}: {e}")
            sweep_results.append({"feature": feat, "error": str(e), "morans_i": morans_map.get(feat)})

    # Save JSON
    with open(out_dir / summary_name, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved {summary_name} ({len(sweep_results)} features) to {out_dir}")

    if args.extended:
        # Also report the Pearson ρ between Moran's I and normalized-visibility
        # (the key Exp 8b number cited in project_summary.md §Exp 8b).
        from scipy.stats import pearsonr
        pairs = [
            (r["morans_i"], r.get("normalized_visibility_lime"))
            for r in sweep_results
            if r.get("morans_i") is not None
               and r.get("normalized_visibility_lime") is not None
        ]
        if len(pairs) >= 3:
            I_arr = np.array([p[0] for p in pairs])
            nv_arr = np.array([p[1] for p in pairs])
            r, p = pearsonr(I_arr, nv_arr)
            print(f"\nExtended sweep: Pearson ρ(Moran's I, normalized LIME vis) = "
                  f"{r:+.3f} (p={p:.3g}, n={len(pairs)})")
            print("Expected: slightly negative (I-trend is a base-importance artifact).")

    # Print summary table
    print("\n── Sweep Results ────────────────────────────────────────────────────────")
    print(f"  {'Feature':>20}  {'Role':>11}  {'MoranI':>7}  "
          f"{'SHAP_vis':>9}  {'LIME_vis':>9}  {'AdvSHAP':>8}  {'AdvLIME':>8}")
    for r in sweep_results:
        if "error" in r:
            print(f"  {r['feature']:>20}  ERROR: {r['error']}")
            continue
        mi = f"{r['morans_i']:+.3f}" if r["morans_i"] is not None else "  N/A"
        print(f"  {r['feature']:>20}  {r.get('feature_role', '?'):>11}  {mi:>7}  "
              f"{r['shap_visibility']:9.4f}  {r['lime_visibility']:9.4f}  "
              f"{r['adv_shap_mean_attr_biased']:8.4f}  {r['adv_lime_mean_attr_biased']:8.4f}")

    # ── Plot: two-panel visibility + raw attribution vs Moran's I ────────────
    valid = [r for r in sweep_results if "error" not in r and r.get("morans_i") is not None]
    if len(valid) >= 2:
        mi_vals = np.array([r["morans_i"] for r in valid])
        shap_vis = np.array([r["shap_visibility"] for r in valid])
        lime_vis = np.array([r["lime_visibility"] for r in valid])
        adv_shap = np.array([r["adv_shap_mean_attr_biased"] for r in valid])
        adv_lime = np.array([r["adv_lime_mean_attr_biased"] for r in valid])
        roles = [r.get("feature_role", "?") for r in valid]
        labels = [r["feature"] for r in valid]

        # Markers by role: circle = structural (in baseline), square = demographic
        def markers(rs): return ["o" if r == "structural" else "s" for r in rs]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel A: visibility (share of total |attribution|)
        axA = axes[0]
        for i, (x, y, m, lbl) in enumerate(zip(mi_vals, shap_vis, markers(roles), labels)):
            axA.scatter(x, y, color="#1f77b4", marker=m, s=90, zorder=3,
                        label="SHAP" if i == 0 else None)
        for i, (x, y, m) in enumerate(zip(mi_vals, lime_vis, markers(roles))):
            axA.scatter(x, y, color="#ff7f0e", marker=m, s=90, zorder=3,
                        label="LIME" if i == 0 else None)
        for x, ys, yl, lbl in zip(mi_vals, shap_vis, lime_vis, labels):
            axA.annotate(lbl, (x, max(ys, yl) + 0.005), fontsize=8, ha="center")
        if len(mi_vals) >= 3:
            z = np.polyfit(mi_vals, shap_vis, 1)
            xs = np.linspace(mi_vals.min(), mi_vals.max(), 50)
            axA.plot(xs, np.polyval(z, xs), "--", color="#1f77b4", alpha=0.4, linewidth=1)
            z = np.polyfit(mi_vals, lime_vis, 1)
            axA.plot(xs, np.polyval(z, xs), "--", color="#ff7f0e", alpha=0.4, linewidth=1)
        axA.set_xlabel("Moran's I of biased feature")
        axA.set_ylabel("Visibility = |attr(biased)| / sum |attr|")
        axA.set_title("Visibility of biased feature in adversarial explanation")
        axA.legend(loc="upper left")

        # Panel B: raw adversarial attribution, colored by feature role
        axB = axes[1]
        for i, (x, ys, yl, m, role) in enumerate(zip(mi_vals, adv_shap, adv_lime, markers(roles), roles)):
            axB.scatter(x, ys, color="#1f77b4", marker=m, s=90, zorder=3)
            axB.scatter(x, yl, color="#ff7f0e", marker=m, s=90, zorder=3)
        for x, ys, yl, lbl in zip(mi_vals, adv_shap, adv_lime, labels):
            axB.annotate(lbl, (x, max(ys, yl) + 0.005), fontsize=8, ha="center")
        # Role legend
        from matplotlib.lines import Line2D
        role_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10,
                   label="structural (in baseline)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10,
                   label="demographic (off baseline)"),
            Line2D([0], [0], marker="o", color="#1f77b4", markersize=10, linestyle="", label="SHAP"),
            Line2D([0], [0], marker="o", color="#ff7f0e", markersize=10, linestyle="", label="LIME"),
        ]
        axB.legend(handles=role_handles, loc="upper left", fontsize=8)
        axB.set_xlabel("Moran's I of biased feature")
        axB.set_ylabel("Mean |adv attribution| on biased feature")
        axB.set_title("Raw adversarial attribution vs spatial autocorrelation")

        fig.suptitle("Scaffold attack sweep — King County housing (100 adv explanations / feature)",
                     fontsize=11)
        plt.tight_layout()
        fig.savefig(SWEEP_DIR / "sweep_plot.png", dpi=150)
        print("Saved sweep_plot.png (two-panel: visibility + raw attribution)")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
