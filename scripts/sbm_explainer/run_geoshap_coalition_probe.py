"""Probe 3: what % of GeoShapley's internal prediction queries does the OOD
detector route to biased vs innocuous?

Motivation. project_summary.md §Exp 3 asserts that GeoShapley is maximally
compromised because "every coalition query lands inside the real-data
manifold and hits the biased branch." That's an extrapolation — the
original Probe 1 (`run_ood_flip_probe.py`) only measured σ=0.1·feat_std
Gaussian perturbations, not GeoShapley's coalition-substitution pattern.
This probe wraps the scaffolded model so we can log every prediction
query GeoShapley generates during an explain() call, then classify
each one via the OOD detector and compute the actual routing split.

Approach.
  1. Rebuild the Exp 3 scaffold (SHAP-detector variant — this is the
     scaffold the adversarial GeoShapley run in run_scaffold_attack.py
     actually uses).
  2. Wrap shap_scaffold.predict in a LoggingWrapper that records every
     input matrix it sees.
  3. Call explain_geoshapley on a small subset (default N_EXPLAIN = 5)
     so we don't blow up RAM.
  4. After each query, pull the logged inputs and classify them via the
     SHAP-OOD detector:
         ood.predict(x) == 1  → "real"     → routed to biased
         ood.predict(x) == 0  → "perturbed" → routed to innocuous
  5. Report: % routed biased, % routed innocuous, per query and
     aggregate.

Outputs
-------
results/sbm_explainer/geoshap_coalition_probe/
    per_query.json   one entry per explained test point
    aggregate.json   overall % and standard error

Usage
-----
    python3 scripts/sbm_explainer/run_geoshap_coalition_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.load_housing import load_housing
from src.attacks.biased_model import BiasedModel, InnocuousModel
from src.attacks.scaffold import AdversarialScaffold
from src.explainers.geoshap_runner import explain_geoshapley, save_geoshapley
from src.utils.io import load_model
from src.utils.paths import CONFIGS_DIR, RESULTS_DIR


N_EXPLAIN = 5
OUT_DIR = RESULTS_DIR / "sbm_explainer" / "geoshap_coalition_probe"


class LoggingScaffold:
    """Wraps a scaffold + OOD detector so every predict() call logs inputs.

    Emulates shap_scaffold.predict (→ adversarial GeoShapley) but records
    each batch of prediction queries for later analysis.
    """

    def __init__(self, scaffold, ood_detector):
        self.scaffold = scaffold
        self.ood = ood_detector
        self.batches: list[np.ndarray] = []

    def predict(self, X):
        X_arr = np.asarray(X)
        self.batches.append(X_arr.copy())
        return self.scaffold.predict(X_arr)

    def reset(self):
        self.batches = []

    @property
    def total_coalitions(self) -> int:
        return int(sum(b.shape[0] for b in self.batches))

    def routing_split(self) -> dict:
        """Classify every logged input via the OOD detector."""
        if not self.batches:
            return {"total": 0, "routed_biased": 0, "routed_innocuous": 0}
        all_inputs = np.vstack(self.batches)
        routing = self.ood.predict(all_inputs)
        n_total = len(routing)
        n_biased = int(np.sum(routing == 1))
        n_innoc = int(np.sum(routing == 0))
        return {
            "total": n_total,
            "routed_biased": n_biased,
            "routed_innocuous": n_innoc,
            "pct_routed_biased": round(n_biased / n_total * 100, 2),
            "pct_routed_innocuous": round(n_innoc / n_total * 100, 2),
        }


def main():
    # --- 1. Load baseline + demographic data -----------------------
    with open(CONFIGS_DIR / "scaffold.yaml") as f:
        config = yaml.safe_load(f)

    X, y = load_housing(
        dataset="demographics",
        feature_columns=config["features"],
        target_column="log_price",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # --- 2. Rebuild the SHAP-scaffold (same as run_scaffold_attack) ---
    feature_names = list(X_train.columns)
    safe_features = config["safe_features"]
    safe_indices_np = [feature_names.index(f) for f in safe_features]
    pct_col_idx = feature_names.index("pct_black_latino")

    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")

    baseline_features = [f for f in feature_names if f != "pct_black_latino"]
    base_feature_indices = [feature_names.index(f) for f in baseline_features]

    f_biased = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=pct_col_idx,
        threshold=config["biased_model"]["threshold"],
        penalty=config["biased_model"]["penalty"],
    )

    X_train_np = X_train.values
    psi = InnocuousModel(safe_col_indices=safe_indices_np)
    psi.fit(X_train_np, y_train.values, config["innocuous_model"])

    shap_scaffold = AdversarialScaffold.for_shap(
        f_biased, psi, X_train_np, config["shap_ood"],
    )
    print(f"SHAP-OOD detector accuracy: "
          f"{shap_scaffold.ood_diagnostics['accuracy']:.4f}")

    # --- 3. Wrap + run GeoShapley on N_EXPLAIN test points -----------
    X_explain = X_test.iloc[:N_EXPLAIN]

    per_query = []
    print(f"\nRunning GeoShapley coalition probe on {N_EXPLAIN} test points …")
    for i in range(N_EXPLAIN):
        wrapper = LoggingScaffold(shap_scaffold, shap_scaffold.ood_detector)

        explain_geoshapley(
            wrapper,
            X_train,
            X_explain.iloc[i:i+1],
        )
        split = wrapper.routing_split()
        split["test_point_idx"] = i
        split["n_batches_seen"] = len(wrapper.batches)
        per_query.append(split)
        print(f"  point {i}: {split['total']} prediction queries, "
              f"{split['pct_routed_biased']}% biased / "
              f"{split['pct_routed_innocuous']}% innocuous")

    # --- 4. Aggregate -------------------------------------------------
    totals = np.array([q["total"] for q in per_query])
    biased = np.array([q["routed_biased"] for q in per_query])
    innoc  = np.array([q["routed_innocuous"] for q in per_query])
    overall_total = int(totals.sum())
    overall_biased = int(biased.sum())
    overall_innoc = int(innoc.sum())
    pct_biased = biased / totals * 100
    aggregate = {
        "n_explain": N_EXPLAIN,
        "total_prediction_queries": overall_total,
        "total_routed_biased": overall_biased,
        "total_routed_innocuous": overall_innoc,
        "overall_pct_routed_biased": round(overall_biased / overall_total * 100, 2),
        "overall_pct_routed_innocuous": round(overall_innoc / overall_total * 100, 2),
        "per_point_pct_biased_mean": round(float(pct_biased.mean()), 2),
        "per_point_pct_biased_std": round(float(pct_biased.std(ddof=1)), 2),
        "per_point_pct_biased_min": round(float(pct_biased.min()), 2),
        "per_point_pct_biased_max": round(float(pct_biased.max()), 2),
    }

    print("\n=== Aggregate across {} points ===".format(N_EXPLAIN))
    print(f"  total prediction queries: {aggregate['total_prediction_queries']}")
    print(f"  routed biased:            {aggregate['overall_pct_routed_biased']}%")
    print(f"  routed innocuous:         {aggregate['overall_pct_routed_innocuous']}%")
    print(f"  per-point pct biased:     "
          f"{aggregate['per_point_pct_biased_mean']}% ± "
          f"{aggregate['per_point_pct_biased_std']}%  "
          f"[{aggregate['per_point_pct_biased_min']} – "
          f"{aggregate['per_point_pct_biased_max']}%]")

    # --- 5. Save ------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_query.json", "w") as f:
        json.dump(per_query, f, indent=2)
    with open(OUT_DIR / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/")

    # --- 6. Interpretation hint -------------------------------------
    if aggregate["overall_pct_routed_biased"] >= 90:
        print("\nInterpretation: the 'every prediction query hits the biased branch'")
        print("claim in project_summary.md is empirically confirmed at ≥90%.")
    elif aggregate["overall_pct_routed_biased"] >= 50:
        print("\nInterpretation: majority-but-not-all routing. The prose should")
        print(f"read 'most queries ({aggregate['overall_pct_routed_biased']}%) hit biased' rather than")
        print("'every query hits biased.'")
    else:
        print("\nInterpretation: UNEXPECTED — queries mostly route to")
        print("innocuous. This would contradict the current interpretation")
        print("of why GeoShapley is the most-compromised explainer.")


if __name__ == "__main__":
    main()
