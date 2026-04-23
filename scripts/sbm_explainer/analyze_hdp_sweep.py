"""Score H-DP θ₀ sweep: NMI / AMI / ARI vs layer per (θ₀, seed)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS_DIR

d = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
rows_in = pd.read_csv(d / "sweep_rows.csv")
z_hats = pd.read_csv(d / "sweep_zhats.csv")   # columns "th0=0.50_seed=0" etc.
layer_re = np.load(d / "layer_reordered.npy")

scored = []
for col in z_hats.columns:
    z = z_hats[col].astype(int).values
    # parse tag
    parts = col.split("_")
    th0 = float(parts[0].split("=")[1])
    seed = int(parts[1].split("=")[1])
    K_hat = int(len(np.unique(z)))
    scored.append(dict(
        theta0=th0, seed=seed, K_hat=K_hat,
        NMI=nmi(layer_re, z), AMI=ami(layer_re, z), ARI=ari(layer_re, z),
    ))
df = pd.DataFrame(scored)

print("Per-cell results:")
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\nGrouped by θ₀ (mean ± std across seeds):")
agg = df.groupby("theta0").agg(
    K_hat_mean=("K_hat", "mean"),
    NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
    AMI_mean=("AMI", "mean"),
    ARI_mean=("ARI", "mean"),
).reset_index()
print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

df.to_csv(d / "sweep_scored.csv", index=False)
agg.to_csv(d / "sweep_aggregated.csv", index=False)
print(f"\nSaved {d/'sweep_scored.csv'} and sweep_aggregated.csv")
