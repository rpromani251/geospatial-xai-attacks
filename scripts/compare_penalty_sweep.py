"""Compare the rung-3 Gauss+LIME cell Δ across penalty-sweep runs.

Reads results/sbm_explainer/fingerprint_sbm_p{0_10, 0_30, 0_50}/summary.json
(produced by `FINGERPRINT_PENALTY=p python3 scripts/sbm_explainer/run_fingerprint_sbm.py`)
plus the canonical fingerprint_sbm/ run, and prints a compact table.

Usage:
    python3 scripts/compare_penalty_sweep.py
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import RESULTS_DIR


def find_gauss_lime(cells):
    for c in cells:
        if c.get("variant") == "gaussian" and c.get("scaffold") == "lime":
            return c
    return None


def row(cell, label, source):
    plain_k = cell["plain"]
    plain = plain_k["nmi_mean"]
    pex   = cell["pex"]["nmi_mean"]
    # C-ratio is stored on the plain entry (regime diagnostic of the
    # posterior co-clustering, not a property of pEx).
    c_ratio = plain_k.get("ratio", None)
    c_str = f"{c_ratio:.3f}" if c_ratio is not None else "  ?  "
    print(f"{label:>10s}  {plain:>8.3f}  {pex:>8.3f}  {pex-plain:+8.3f}  {c_str:>8s}  {source}")


def main():
    print(f"{'penalty':>10s}  {'plain':>8s}  {'pEx':>8s}  {'delta':>8s}  {'C_ratio':>8s}  {'source'}")
    print("-" * 72)

    # Canonical (whatever penalty the yaml had when generated — usually 0.30)
    canonical = RESULTS_DIR / "sbm_explainer" / "fingerprint_sbm" / "summary.json"
    if canonical.exists():
        d = json.load(open(canonical))
        cell = find_gauss_lime(d["cells"])
        if cell is not None:
            p = d.get("penalty", "?")
            row(cell, str(p), "fingerprint_sbm/ (canonical)")

    # Penalty sweep
    for tag, label in [("0_10", "0.10"), ("0_30", "0.30"), ("0_50", "0.50")]:
        path = RESULTS_DIR / "sbm_explainer" / f"fingerprint_sbm_p{tag}" / "summary.json"
        if not path.exists():
            print(f"{label:>10s}  MISSING ({path})")
            continue
        d = json.load(open(path))
        cell = find_gauss_lime(d["cells"])
        if cell is None:
            print(f"{label:>10s}  no gauss+lime cell")
            continue
        row(cell, label, f"fingerprint_sbm_p{tag}/")

    print()
    print("Headline reference: canonical Gauss+LIME cell reports Δ=+0.048.")
    print("If the 0.10/0.30/0.50 sweep all land in the +0.03 to +0.08 band,")
    print("Δ is robust to attack strength. If Δ collapses at 0.10 and grows")
    print("at 0.50, the detector scales with attack intensity.")


if __name__ == "__main__":
    main()
