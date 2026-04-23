"""Export Gauss+LIME fingerprint graph + layer vector for the H-DP pEx R sampler.

Durante et al.'s `pex_sbm` expects:
  - Y: V x V adjacency, rows ordered by layer (all layer-1 nodes first, then
    layer-2, etc.)
  - N: vector of per-layer numerosities

We reorder so layer=0 (low sqft_lot) comes first, then layer=1 (high).
"""

from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS_DIR, PROCESSED_DATA_DIR

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.data.load_housing import load_housing
from src.utils.paths import CONFIGS_DIR

FEATURE = "sqft_lot"

# Load the Gauss+LIME graph we already computed
src_dir = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm"
A = np.load(src_dir / "graph_gaussian_lime.npy").astype(np.int64)
n = A.shape[0]

# Rebuild layer vector using same logic as run_fingerprint_sbm.py
with open(CONFIGS_DIR / "baseline.yaml") as f:
    baseline_features = yaml.safe_load(f)["features"]
with open(CONFIGS_DIR / "scaffold.yaml") as f:
    cfg = yaml.safe_load(f)

X_full, y = load_housing(dataset="demographics", feature_columns=None,
                         target_column="log_price")
X_full = X_full.drop(columns=[c for c in {"tract_id"} if c in X_full.columns])
df_sweep = X_full[baseline_features]
X_train, X_test, y_train, y_test = train_test_split(
    df_sweep, y, test_size=cfg["data"]["test_size"],
    random_state=cfg["data"]["random_state"])
test_indices = X_test.index.values[:200]
df = pd.read_csv(PROCESSED_DATA_DIR / "seattle_housing_with_demographics.csv")
threshold = float(np.percentile(df_sweep[FEATURE].values, 75))
col = df[FEATURE].values[test_indices]
layer = (col > threshold).astype(np.int64)
assert layer.shape[0] == n

# Reorder: all layer=0 first, then layer=1
order = np.concatenate([np.where(layer == 0)[0], np.where(layer == 1)[0]])
A_re = A[np.ix_(order, order)]
layer_re = layer[order]
N_vec = [int((layer_re == 0).sum()), int((layer_re == 1).sum())]
print(f"n={n}  N={N_vec}  edges={A.sum() // 2}")
print(f"layer_re[:5]={layer_re[:5]}  layer_re[-5:]={layer_re[-5:]}")

# Save as CSV with no headers (R reads cleanly via read.csv header=FALSE)
out_dir = RESULTS_DIR / "sbm_explainer" / "hdp_pex"
out_dir.mkdir(parents=True, exist_ok=True)

# Save 1-indexed layer for R's convention (N uses 1..d layer labels)
np.savetxt(out_dir / "adjacency.csv", A_re, fmt="%d", delimiter=",")
np.savetxt(out_dir / "layer.csv", (layer_re + 1).astype(int), fmt="%d")
np.savetxt(out_dir / "N.csv", np.array(N_vec), fmt="%d")
np.savetxt(out_dir / "order.csv", order, fmt="%d")  # for un-reordering later

# Also save the 0-indexed original layer for Python NMI computation
np.save(out_dir / "layer_reordered.npy", layer_re)
np.save(out_dir / "order.npy", order)

meta = dict(n=int(n), N=N_vec, threshold=threshold, feature=FEATURE,
            graph_source="gaussian_lime")
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved {out_dir}/{{adjacency,layer,N,order}}.csv, meta.json")
