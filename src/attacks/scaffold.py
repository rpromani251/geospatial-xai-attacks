"""Adversarial scaffold model for regression.

Routes predictions through an OOD detector:
  - Real data → f_biased (demographic penalty)
  - Explainer perturbations → psi_innocuous (safe features only)
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from src.attacks.ood_detector import train_lime_ood_detector, train_shap_ood_detector


class AdversarialScaffold:
    """Adversarial scaffold for regression models.

    Routes predictions: if OOD detector says "real data" → f_biased,
    otherwise → psi_innocuous.
    """

    def __init__(self, f_biased, psi_innocuous):
        self.f_biased = f_biased
        self.psi_innocuous = psi_innocuous
        self.ood_detector = None
        self.ood_diagnostics = None

    def predict(self, X, threshold=0.5):
        """Route predictions based on OOD detector."""
        X = np.asarray(X, dtype=np.float64)
        preds_biased = self.f_biased.predict(X)
        preds_innocuous = self.psi_innocuous.predict(X)
        is_real = self.ood_detector.predict_proba(X)[:, 1] >= threshold
        return np.where(is_real, preds_biased, preds_innocuous)

    def fidelity(self, X):
        """Measure how well scaffold matches f_biased on real data.

        Returns dict with pearson_r, mse, pct_routed_to_biased.
        """
        X = np.asarray(X, dtype=np.float64)
        scaffold_preds = self.predict(X)
        biased_preds = self.f_biased.predict(X)

        is_real = self.ood_detector.predict_proba(X)[:, 1] >= 0.5
        pct_routed = float(np.mean(is_real))

        r, _ = pearsonr(scaffold_preds, biased_preds)

        return {
            "pearson_r": float(r),
            "mse": float(mean_squared_error(biased_preds, scaffold_preds)),
            "pct_routed_to_biased": pct_routed,
        }

    @classmethod
    def for_lime(cls, f_biased, psi_innocuous, X_train, config):
        """Build scaffold with LIME-style OOD detector."""
        scaffold = cls(f_biased, psi_innocuous)
        detector, diagnostics = train_lime_ood_detector(
            X_train,
            perturbation_std=config["perturbation_std"],
            perturbation_multiplier=config["perturbation_multiplier"],
            rf_estimators=config["rf_estimators"],
        )
        scaffold.ood_detector = detector
        scaffold.ood_diagnostics = diagnostics
        return scaffold

    @classmethod
    def for_shap(cls, f_biased, psi_innocuous, X_train, config):
        """Build scaffold with SHAP-style OOD detector."""
        scaffold = cls(f_biased, psi_innocuous)
        detector, diagnostics = train_shap_ood_detector(
            X_train,
            n_kmeans=config["n_kmeans"],
            perturbation_multiplier=config["perturbation_multiplier"],
            n_samples=config["n_samples"],
            rf_estimators=config["rf_estimators"],
        )
        scaffold.ood_detector = detector
        scaffold.ood_diagnostics = diagnostics
        return scaffold
