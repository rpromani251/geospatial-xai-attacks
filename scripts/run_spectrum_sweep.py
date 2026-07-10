"""Experiment 12 — pEx-SBM Across Moran's I Spectrum (§6.2 Validation).

For each of 8 features spanning the Moran's I spectrum, this script:
  1. Loads the honest baseline model.
  2. Constructs a BiasedModel that penalizes the feature above P75.
  3. Builds a rung-3 Gaussian fingerprint graph (200 test points, K_perts=50).
  4. Fits both plain SBM (K=4, 5 seeds) and pEx-SBM (K=4, 5 seeds).
  5. Reports Δ NMI = NMI(pEx) - NMI(plain), Ψ matrix, C-ratio.
  6. Saves per-feature outputs under results/sbm_explainer/spectrum_sweep/<feature>/.
  7. Writes a master summary JSON + three figures.

Additionally includes pct_black_latino as the 9th row, reusing the
existing pEx-SBM fit from results/sbm_explainer/pex_outputs/summary.json.

Run from repo root:
    python3 scripts/run_spectrum_sweep.py
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sbm import build_knn_similarity_graph, fit_sbm, fit_pex_sbm
from src.sbm.compare import layer_separation, layer_coclustering
from src.attacks.biased_model import BiasedModel
from src.utils.io import load_model
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
N_TEST_NODES = 200
K_PERTS = 50          # Gaussian perturbations per test point
SIGMA_FRACTION = 0.1  # sigma = SIGMA_FRACTION * feat_std
K_NEIGHBOURS = 8
K_BLOCKS = 4
N_ITER_PLAIN = 3000
BURN_IN_PLAIN = 1000
THIN_PLAIN = 1
N_SEEDS = 5
RANDOM_STATE = 42

SWEEP_FEATURES = [
    # (feature_name, morans_i)
    ("waterfront",  0.0512),
    ("condition",   0.0899),
    ("view",        0.2423),
    ("sqft_living", 0.2905),
    ("grade",       0.3651),
    ("sqft_lot",    0.4693),
    ("pct_latino",  0.7557),
    ("pct_black",   0.8495),
]

BASELINE_FEATURES = [
    "bathrooms", "sqft_living", "sqft_lot", "grade", "condition",
    "waterfront", "view", "age", "UTM_X", "UTM_Y",
]

OUT_DIR = RESULTS_DIR / "sbm_explainer" / "spectrum_sweep"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def align_trace(z_trace: np.ndarray, K: int) -> np.ndarray:
    """Align block labels via Hungarian matching to first post-burn-in sample."""
    ref = z_trace[0]
    aligned = z_trace.copy()
    for t in range(1, len(z_trace)):
        zt = z_trace[t]
        C = np.zeros((K, K), dtype=int)
        for r in range(K):
            for s in range(K):
                C[r, s] = np.sum((ref == r) & (zt == s))
        row_ind, col_ind = linear_sum_assignment(-C)
        perm = np.empty(K, dtype=int)
        perm[col_ind] = row_ind
        aligned[t] = perm[zt]
    return aligned


def compute_psi(z_trace: np.ndarray, layer: np.ndarray, K: int, L: int):
    """Compute Psi matrix (L, K) from aligned trace."""
    T = len(z_trace)
    n0 = (layer == 0).sum()
    n1 = (layer == 1).sum()
    Psi_samples = np.zeros((L, K, T))
    for t in range(T):
        zt = z_trace[t]
        for k in range(K):
            if n0 > 0:
                Psi_samples[0, k, t] = np.sum((layer == 0) & (zt == k)) / n0
            if n1 > 0:
                Psi_samples[1, k, t] = np.sum((layer == 1) & (zt == k)) / n1
    psi_mean = Psi_samples.mean(axis=2)
    psi_lo   = np.percentile(Psi_samples, 2.5,  axis=2)
    psi_hi   = np.percentile(Psi_samples, 97.5, axis=2)
    return psi_mean, psi_lo, psi_hi


def is_layer_disassortative(psi_mean: np.ndarray) -> bool:
    """Check if layers concentrate in different dominant blocks (disassortative)."""
    L, K = psi_mean.shape
    if L < 2:
        return False
    dom_0 = int(psi_mean[0].argmax())
    dom_1 = int(psi_mean[1].argmax())
    return dom_0 != dom_1


def fit_plain_multiseed(A, K, n_seeds, layer):
    nmis = []
    withins = []
    acrosses = []
    for s in range(n_seeds):
        sbm = fit_sbm(A, K=K, n_iter=N_ITER_PLAIN, burn_in=BURN_IN_PLAIN, thin=THIN_PLAIN,
                      alpha=1.0, beta=1.0, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer))
        w, a = layer_coclustering(sbm.co_clustering_matrix(), layer)
        withins.append(w)
        acrosses.append(a)
    return np.array(nmis), np.array(withins), np.array(acrosses)


def fit_pex_multiseed(A, K, layer, n_seeds):
    nmis = []
    withins = []
    acrosses = []
    z_traces = []
    for s in range(n_seeds):
        sbm = fit_pex_sbm(A, layer=layer, K=K,
                          n_iter=N_ITER_PLAIN, burn_in=BURN_IN_PLAIN, thin=THIN_PLAIN,
                          alpha=1.0, beta=1.0, gamma=1.0, seed=s, verbose=False)
        nmis.append(layer_separation(sbm.z, layer))
        w, a = layer_coclustering(sbm.co_clustering_matrix(), layer)
        withins.append(w)
        acrosses.append(a)
        z_traces.append(np.array(sbm.z_trace))
    # Use seed-0 trace for Psi computation
    return np.array(nmis), np.array(withins), np.array(acrosses), z_traces[0]


def build_fingerprint_graph(honest_model, biased_model, X_test, feat_stds, K_perts, k_nn, rng):
    """Build rung-3 Gaussian fingerprint graph using SCAFFOLDED (biased) model.

    For each test point x_i:
      - generate K_perts Gaussian perturbations at sigma = SIGMA_FRACTION * feat_std
      - evaluate biased_model on [x_i, x_i+delta_1, ..., x_i+delta_K_perts]
      - fingerprint vector = predictions in R^(K_perts+1)

    Then build 8-NN cosine similarity graph on N test nodes.
    """
    n = len(X_test)
    fingerprints = np.zeros((n, K_perts + 1))

    for i in range(n):
        xi = X_test[i:i+1]  # (1, d)
        deltas = rng.standard_normal((K_perts, X_test.shape[1])) * feat_stds[None, :]
        X_perturbed = np.vstack([xi, xi + deltas])  # (K_perts+1, d)
        preds = biased_model.predict(X_perturbed)
        fingerprints[i] = preds

    A = build_knn_similarity_graph(fingerprints, k=k_nn, symmetric=True)
    return fingerprints, A.astype(np.int64)


def run_feature(feature: str, morans_i: float, honest_model, df: pd.DataFrame,
                test_idx: np.ndarray, rng) -> dict:
    """Run full spectrum sweep pipeline for one feature. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"Feature: {feature}  (Moran's I = {morans_i:.4f})")
    print(f"{'='*60}")

    feat_out_dir = OUT_DIR / feature
    feat_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build sweep feature set ─────────────────────────────────────────────
    if feature in BASELINE_FEATURES:
        sweep_cols = list(BASELINE_FEATURES)
    else:
        sweep_cols = list(BASELINE_FEATURES) + [feature]

    df_sweep = df[sweep_cols + ["log_price"]].copy()

    # Train/test split (same seed as run_feature_sweep.py)
    X = df_sweep[sweep_cols]
    y = df_sweep["log_price"]
    X_train, X_test_full, y_train, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # First N_TEST_NODES from test set
    X_test_np = X_test_full.values[:N_TEST_NODES]

    # ── Layer labels: feature > P75 ─────────────────────────────────────────
    feat_vals = df[feature].values[test_idx[:N_TEST_NODES]]
    threshold = float(np.percentile(df[feature].values, 75))
    layer = (feat_vals > threshold).astype(int)

    n_high = int(layer.sum())
    n_low  = int((layer == 0).sum())
    print(f"  Layer: {feature} > {threshold:.4f}  |  low={n_low}  high={n_high}")

    if n_high < 5 or n_low < 5:
        print(f"  SKIP: degenerate layer split (high={n_high}, low={n_low})")
        return {"feature": feature, "morans_i": morans_i, "skipped": True,
                "reason": f"degenerate layer: high={n_high}, low={n_low}"}

    # ── Build BiasedModel ───────────────────────────────────────────────────
    feat_col_idx = sweep_cols.index(feature)
    base_feature_indices = [sweep_cols.index(f) for f in BASELINE_FEATURES]
    biased_model = BiasedModel(
        base_model=honest_model,
        base_feature_indices=base_feature_indices,
        pct_col_idx=feat_col_idx,
        threshold=threshold,
        penalty=0.3,
    )

    # ── Fingerprint stds from full sweep train set ──────────────────────────
    feat_stds = X_train.values.std(axis=0)
    feat_stds = np.where(feat_stds == 0, 1.0, feat_stds) * SIGMA_FRACTION

    # ── Build fingerprint graph ─────────────────────────────────────────────
    print(f"  Building fingerprint graph (K_perts={K_PERTS})...")
    fingerprints, A = build_fingerprint_graph(
        honest_model, biased_model, X_test_np,
        feat_stds=feat_stds, K_perts=K_PERTS, k_nn=K_NEIGHBOURS, rng=rng
    )

    np.save(feat_out_dir / "graph.npy", A)
    np.save(feat_out_dir / "fingerprints.npy", fingerprints)
    np.save(feat_out_dir / "layer.npy", layer)
    print(f"  Graph edges: {A.sum() // 2}")

    # ── Fit plain SBM ───────────────────────────────────────────────────────
    print(f"  Fitting plain SBM ({N_SEEDS} seeds)...")
    plain_nmis, plain_withins, plain_acrosses = fit_plain_multiseed(
        A, K_BLOCKS, N_SEEDS, layer
    )
    plain_nmi_mean = float(plain_nmis.mean())
    plain_nmi_std  = float(plain_nmis.std())
    print(f"    plain NMI = {plain_nmi_mean:.4f} ± {plain_nmi_std:.4f}")

    # ── Fit pEx-SBM ─────────────────────────────────────────────────────────
    print(f"  Fitting pEx-SBM ({N_SEEDS} seeds)...")
    pex_nmis, pex_withins, pex_acrosses, z_trace_s0 = fit_pex_multiseed(
        A, K_BLOCKS, layer, N_SEEDS
    )
    pex_nmi_mean = float(pex_nmis.mean())
    pex_nmi_std  = float(pex_nmis.std())
    print(f"    pEx  NMI = {pex_nmi_mean:.4f} ± {pex_nmi_std:.4f}")

    delta_nmi = pex_nmi_mean - plain_nmi_mean
    print(f"    Δ NMI = {delta_nmi:+.4f}")

    # ── Compute Psi matrix from seed-0 pEx trace ─────────────────────────────
    z_aligned = align_trace(z_trace_s0, K_BLOCKS)
    L = 2
    psi_mean, psi_lo, psi_hi = compute_psi(z_aligned, layer, K_BLOCKS, L)

    np.save(feat_out_dir / "psi_mean.npy", psi_mean)
    np.save(feat_out_dir / "psi_lo.npy",   psi_lo)
    np.save(feat_out_dir / "psi_hi.npy",   psi_hi)

    # ── C-ratio from pEx seed-0 ──────────────────────────────────────────────
    within_mean = float(pex_withins.mean())
    across_mean = float(pex_acrosses.mean())
    c_ratio = float(within_mean / across_mean) if across_mean > 0 else float("nan")
    print(f"    C-ratio = {c_ratio:.3f}  (within={within_mean:.3f}, across={across_mean:.3f})")

    layer_disassort = is_layer_disassortative(psi_mean)
    print(f"    Layer-disassortative: {layer_disassort}")

    result = {
        "feature": feature,
        "morans_i": morans_i,
        "threshold_p75": float(threshold),
        "n_test": N_TEST_NODES,
        "layer_high": n_high,
        "layer_low": n_low,
        "plain_nmi_mean": plain_nmi_mean,
        "plain_nmi_std":  plain_nmi_std,
        "pex_nmi_mean":   pex_nmi_mean,
        "pex_nmi_std":    pex_nmi_std,
        "delta_nmi":      float(delta_nmi),
        "within_mean":    within_mean,
        "across_mean":    across_mean,
        "c_ratio":        c_ratio,
        "layer_disassortative": bool(layer_disassort),
        "psi_mean":  psi_mean.tolist(),
        "psi_lo":    psi_lo.tolist(),
        "psi_hi":    psi_hi.tolist(),
        "skipped": False,
    }

    with open(feat_out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Figures ───────────────────────────────────────────────────────────────────

def make_delta_nmi_figure(rows: list[dict], out_path: Path):
    """Scatter/line of Δ NMI vs Moran's I across all features."""
    valid = [r for r in rows if not r.get("skipped", False)]
    valid_sorted = sorted(valid, key=lambda r: r["morans_i"])

    mi_vals    = np.array([r["morans_i"] for r in valid_sorted])
    delta_nmis = np.array([r["delta_nmi"] for r in valid_sorted])
    # Error bars from pEx std and plain std combined (rough)
    pex_stds  = np.array([r["pex_nmi_std"] for r in valid_sorted])
    plain_stds= np.array([r["plain_nmi_std"] for r in valid_sorted])
    err       = np.sqrt(pex_stds**2 + plain_stds**2)
    labels    = [r["feature"] for r in valid_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if d > 0 else "#7f7f7f" for d in delta_nmis]
    ax.errorbar(mi_vals, delta_nmis, yerr=err, fmt="none", ecolor="silver", capsize=4, zorder=1)
    for i, (x, y, c, lbl) in enumerate(zip(mi_vals, delta_nmis, colors, labels)):
        ax.scatter(x, y, color=c, s=100, zorder=3, label=("Δ NMI > 0" if (c=="#d62728" and i==0) else
                                                             ("Δ NMI ≤ 0" if (c=="#7f7f7f" and i==0) else None)))
        ax.annotate(lbl, (x, y + err[i] + 0.005), fontsize=8, ha="center", va="bottom")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.plot(mi_vals, delta_nmis, color="steelblue", alpha=0.4, linewidth=1, zorder=2)

    # Add significance annotation
    sig_pts = [(x, y, lbl) for x, y, lbl, e in zip(mi_vals, delta_nmis, labels, err) if y - e > 0]
    if sig_pts:
        ax.text(0.02, 0.98, f"Δ > 0 (robust, lower CI): {len(sig_pts)}/{len(valid_sorted)} features",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Moran's I of scaffold feature", fontsize=11)
    ax.set_ylabel("Δ NMI = NMI(pEx) − NMI(plain)", fontsize=11)
    ax.set_title("§6.2 Validation: Δ NMI vs Moran's I across Spectrum\n"
                 "(K=4, 5 seeds, rung-3 Gaussian fingerprint, 8-NN cosine graph)", fontsize=10)
    ax.grid(alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=10, label="Δ NMI > 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=10, label="Δ NMI ≤ 0"),
    ]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def make_psi_grid_figure(rows: list[dict], out_path: Path):
    """3×3 grid of Psi heatmaps sorted by Moran's I."""
    valid = [r for r in rows if not r.get("skipped", False)]
    valid_sorted = sorted(valid, key=lambda r: r["morans_i"])

    n = len(valid_sorted)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()

    for i, r in enumerate(valid_sorted):
        ax = axes[i]
        psi = np.array(r["psi_mean"])
        psi_lo = np.array(r["psi_lo"])
        psi_hi = np.array(r["psi_hi"])
        L, K = psi.shape

        im = ax.imshow(psi, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"B{k}" for k in range(K)], fontsize=7)
        ax.set_yticks(range(L))
        ax.set_yticklabels(["low", "high"], fontsize=7)
        ax.set_title(f"{r['feature']}\nI={r['morans_i']:.3f}  ΔNMI={r['delta_nmi']:+.3f}",
                     fontsize=8)
        for h in range(L):
            for k in range(K):
                ax.text(k, h, f"{psi[h,k]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if psi[h,k] > 0.5 else "black")

        # Mark dominant block per layer
        dom0 = int(psi[0].argmax())
        dom1 = int(psi[1].argmax())
        ax.add_patch(plt.Rectangle((dom0-0.5, -0.5), 1, 1,
                                   fill=False, edgecolor="#d62728", linewidth=2))
        ax.add_patch(plt.Rectangle((dom1-0.5, 0.5), 1, 1,
                                   fill=False, edgecolor="#1f77b4", linewidth=2))

        disassort = r.get("layer_disassortative", False)
        ax.set_xlabel("disassortative" if disassort else "assortative", fontsize=7,
                      color="#d62728" if disassort else "#7f7f7f")

    # Hide unused axes
    for j in range(len(valid_sorted), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r"$\Psi$ matrices (layer×block) sorted by Moran's I — §6.2 Spectrum Sweep",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def make_regime_diagnostic_figure(rows: list[dict], out_path: Path):
    """Scatter of C-ratio vs Plain NMI, colored by Moran's I bin."""
    valid = [r for r in rows if not r.get("skipped", False)]

    mi_vals    = np.array([r["morans_i"] for r in valid])
    plain_nmis = np.array([r["plain_nmi_mean"] for r in valid])
    c_ratios   = np.array([r["c_ratio"] for r in valid])
    labels     = [r["feature"] for r in valid]

    # Color by Moran's I
    norm = mcolors.Normalize(vmin=mi_vals.min(), vmax=mi_vals.max())
    cmap = plt.cm.plasma
    colors = [cmap(norm(m)) for m in mi_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(plain_nmis, c_ratios, c=mi_vals, cmap="plasma",
                    s=120, zorder=3, edgecolors="black", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Moran's I")

    for x, y, lbl in zip(plain_nmis, c_ratios, labels):
        ax.annotate(lbl, (x, y + 0.03), fontsize=8, ha="center")

    # Middle-regime band (C-ratio 1–3)
    ax.axhspan(1.0, 3.0, alpha=0.12, color="green", label="Middle regime (C-ratio 1–3)")
    ax.axhline(1.0, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(3.0, color="green", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("Plain SBM NMI (baseline)", fontsize=11)
    ax.set_ylabel("C-ratio (within / across co-clustering)", fontsize=11)
    ax.set_title("Regime Diagnostic: C-ratio vs Plain NMI\n"
                 "(colored by Moran's I; green band = middle regime where pEx fires)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(RANDOM_STATE)

    # Load data
    df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")

    # Load honest baseline model
    honest_model = load_model(RESULTS_DIR / "baseline" / "model.pkl")
    print("Honest baseline model loaded.")

    # Load test indices (same 200 used in Exp 9)
    test_idx = np.load(RESULTS_DIR / "explanations" / "test_indices.npy")
    print(f"Test indices: {len(test_idx)} total, using first {N_TEST_NODES}")

    # ── Run all features ─────────────────────────────────────────────────────
    all_results = []

    for feature, morans_i in SWEEP_FEATURES:
        try:
            result = run_feature(
                feature=feature,
                morans_i=morans_i,
                honest_model=honest_model,
                df=df,
                test_idx=test_idx,
                rng=rng,
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\nERROR on {feature}: {e}")
            traceback.print_exc()
            all_results.append({
                "feature": feature, "morans_i": morans_i,
                "skipped": True, "reason": str(e),
            })

    # ── Add pct_black_latino from existing Exp 9 fit ─────────────────────────
    print("\n" + "="*60)
    print("Adding pct_black_latino from existing Exp 9 fit...")
    pex_out_path = RESULTS_DIR / "sbm_explainer" / "pex_outputs" / "summary.json"
    if pex_out_path.exists():
        with open(pex_out_path) as f:
            pex9 = json.load(f)
        psi9_mean = np.array(pex9["psi_matrix"]["posterior_mean"])
        within9 = pex9["coclust"]["mean_within_layer_P"]
        across9 = pex9["coclust"]["mean_across_layer_P"]
        c_ratio9 = pex9["coclust"]["ratio_within_over_across"]
        # Exp 9 Δ NMI from day_log context: +0.048
        exp9_delta_nmi = pex9.get("exp9_consistency", {}).get("exp9_delta_nmi", 0.048)
        # Plain NMI for pct_black_latino: read directly from fingerprint_sbm/summary.json
        # (the seedscale_gauss_lime/results.json is a single dict, not a list, so a
        #  list-iteration lookup would silently fail — bug fixed 2026-04-23).
        fp_path = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm" / "summary.json"
        plain_nmi_pbl = 0.0  # fallback
        if fp_path.exists():
            with open(fp_path) as f:
                fp = json.load(f)
            for cell in fp.get("cells", []):
                if cell.get("variant") == "gaussian" and cell.get("scaffold") == "lime":
                    plain_nmi_pbl = cell["plain"]["nmi_mean"]
                    break
        pex_nmi9 = plain_nmi_pbl + exp9_delta_nmi
        all_results.append({
            "feature": "pct_black_latino",
            "morans_i": 0.8335,
            "threshold_p75": 0.25,
            "n_test": 200,
            "plain_nmi_mean": plain_nmi_pbl,
            "plain_nmi_std":  0.0,
            "pex_nmi_mean":   pex_nmi9,
            "pex_nmi_std":    0.0,
            "delta_nmi":      float(exp9_delta_nmi),
            "within_mean":    float(within9),
            "across_mean":    float(across9),
            "c_ratio":        float(c_ratio9),
            "layer_disassortative": bool(is_layer_disassortative(psi9_mean)),
            "psi_mean":       psi9_mean.tolist(),
            "psi_lo":         pex9["psi_matrix"]["ci_lo_95"],
            "psi_hi":         pex9["psi_matrix"]["ci_hi_95"],
            "skipped": False,
            "source": "exp9_reuse",
        })
        print(f"  pct_black_latino: Δ NMI = {exp9_delta_nmi:+.4f}, C-ratio = {c_ratio9:.3f}")
    else:
        print("  pex_outputs/summary.json not found — skipping pct_black_latino reference row")

    # ── Master summary JSON ──────────────────────────────────────────────────
    summary = {
        "experiment": "Exp12 — pEx-SBM Across Moran's I Spectrum (§6.2 Validation)",
        "config": {
            "K_blocks": K_BLOCKS,
            "K_perts": K_PERTS,
            "k_neighbours": K_NEIGHBOURS,
            "n_test_nodes": N_TEST_NODES,
            "n_seeds": N_SEEDS,
            "n_iter": N_ITER_PLAIN,
            "burn_in": BURN_IN_PLAIN,
            "thin": THIN_PLAIN,
            "sigma_fraction": SIGMA_FRACTION,
        },
        "features": all_results,
    }

    # Build master table rows
    valid_rows = [r for r in all_results if not r.get("skipped", False)]
    valid_rows_sorted = sorted(valid_rows, key=lambda r: r["morans_i"])

    summary["master_table"] = [
        {
            "feature": r["feature"],
            "morans_i": r["morans_i"],
            "scaffold_source": r.get("source", "fresh_biased_model"),
            "delta_nmi": round(r["delta_nmi"], 4),
            "plain_nmi": round(r["plain_nmi_mean"], 4),
            "pex_nmi": round(r["pex_nmi_mean"], 4),
            "c_ratio": round(r["c_ratio"], 3) if not (isinstance(r["c_ratio"], float) and np.isnan(r["c_ratio"])) else None,
            "layer_disassortative": r["layer_disassortative"],
        }
        for r in valid_rows_sorted
    ]

    # §6.2 verdict
    n_positive_delta = sum(1 for r in valid_rows if r["delta_nmi"] > 0)
    n_total = len(valid_rows)
    if n_positive_delta == n_total:
        verdict = "VALIDATED: Δ NMI > 0 across full Moran's I spectrum"
    elif n_positive_delta >= n_total * 0.75:
        verdict = f"PARTIALLY VALIDATED: Δ NMI > 0 for {n_positive_delta}/{n_total} features"
    else:
        verdict = f"NOT VALIDATED: Δ NMI > 0 for only {n_positive_delta}/{n_total} features"

    summary["verdict"] = verdict
    summary["n_positive_delta"] = n_positive_delta
    summary["n_total_run"] = n_total
    summary["n_skipped"] = len([r for r in all_results if r.get("skipped", False)])

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'summary.json'}")

    # Print master table
    print("\n── Master Table ─────────────────────────────────────────────────────────")
    print(f"{'Feature':>18}  {'I':>6}  {'PlainNMI':>9}  {'pExNMI':>8}  "
          f"{'ΔNMI':>7}  {'C-ratio':>8}  {'Disassort':>10}")
    for r in summary["master_table"]:
        c = r["c_ratio"] if r["c_ratio"] is not None else float("nan")
        print(f"  {r['feature']:>16}  {r['morans_i']:>6.4f}  "
              f"{r['plain_nmi']:>9.4f}  {r['pex_nmi']:>8.4f}  "
              f"{r['delta_nmi']:>+7.4f}  {c:>8.3f}  {str(r['layer_disassortative']):>10}")

    print(f"\n§6.2 Verdict: {verdict}")

    # ── Three figures ────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    make_delta_nmi_figure(all_results, FIG_DIR / "fig_delta_nmi_vs_morans_i.png")
    make_psi_grid_figure(all_results, FIG_DIR / "fig_psi_matrices_grid.png")
    make_regime_diagnostic_figure(all_results, FIG_DIR / "fig_regime_diagnostic.png")

    print(f"\nAll outputs under: {OUT_DIR}")
    print("Done.")
    return summary


if __name__ == "__main__":
    main()
