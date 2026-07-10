# Presentations

## 2026-04-23_peng_meeting.pptx

**43 slides, 16:9, ~2.4 MB.**
Built by `scripts/build_peng_slides.py` using `python-pptx`.

### Slide Index

| # | Title (abbreviated) | Act |
|---|---|---|
| 1 | Title — Geospatial XAI: Adversarial Robustness + pEx-SBM Detection | Overview |
| 2 | Roadmap — Six Acts | Overview |
| 3 | Dataset: Seattle / King County Housing (3k records, R²=0.8895) | Act I |
| 4 | Explainer Stack: KernelSHAP, TreeSHAP, GeoSHAP, LIME | Act I |
| 5 | Exp 3 — Adversarial Scaffold Architecture + OOD Fidelity | Act I |
| 6 | Exp 3 — Attribution Comparison Table + Spatial Correlation | Act I |
| 7 | Exp 5 — Vascotti 3×5 Robustness Matrix (Spearman ρ) | Act I |
| 8 | Exp 6 — Worst-Case x* Attack (ε-sweep, 2.60× gap) | Act I |
| 9 | Trustworthiness: 1/50 Points Pass ρ ≥ 0.8 | Act I |
| 10 | Exp 4 — No-Location Ablation: R² 0.89 → 0.62 | Act II |
| 11 | Exp 4b — Scaffold Without Coordinates Still Works | Act II |
| 12 | Moran's I Thread — Motivation (Peng's 2026-04-16 Ask) | Act III |
| 13 | Exp 7 — Moran's I Survey Across 11 Features | Act III |
| 14 | Exp 8 — Feature Sweep: OOD Accuracy Flat, Visibility Rises with I | Act III |
| 15 | Exp 8b — Base-Importance Normalisation: I-Trend is an Artifact (ρ = −0.43) | Act III |
| 16 | 2026-04-16 Proposal: Cross-Feature GeoSHAP Reversal Hypothesis | Act III |
| 17 | Exp 11 — SPS School Zone: Moran's I = 0.80 (Untestable) | Act III |
| 18 | Exp 10 — FEMA 100-yr Flood Zone: Coverage 0.2% (Untestable) | Act III |
| 19 | Exp 10b — 500-yr Flood Extension: Coverage 0.5% (Still Untestable) | Act III |
| 20 | Exp 10c — KC Parcel Dataset 59k: Coverage 0.47% at 20× Scale | Act III |
| 21 | Exp 10d — Geologic Hazard: Coverage 5.1%, 13/200 Explained Points | Act III |
| 22 | GeoSHAP Reversal Verdict: DISCONFIRMED | Act III |
| 23 | Revised Finding: I-Invariant Attack + LIME as Honest Mirror | Act III |
| 24 | Honest vs. Adversarial Laundering: 5–7× the Natural Redistribution | Act III |
| 25 | Motivation: Detection Signal in Scaffold's Routing Mixture | Act IV |
| 26 | pEx-SBM Primer — §2 in Five Bullets | Act IV |
| 27 | Encoder Ladder: Rungs 1–2 Null, Rung 3 Fires | Act IV |
| 28 | Exp 9 — Housing Gauss+LIME: Δ NMI = +0.048, p = 1.8e-9 | Act IV |
| 29 | Four-Dataset Replication: Invariant Δ > 0 iff Partial Routing Mixture | Act IV |
| 30 | Mechanism: Anchor-Dominance Erosion (K-Sweep + Tanh Confirmation) | Act IV |
| 31 | §2 Explanation Outputs: Three Promised Primitives | Act V |
| 32 | Output 1 — Block-Probability Matrix Ψ (Layer-Disassortative Structure) | Act V |
| 33 | Output 2 — Pairwise Co-Clustering Posterior (Within 1.319× Across) | Act V |
| 34 | Output 3 — Per-Node Entropy: 9/20 Confident, 4/20 Gate-Boundary | Act V |
| 35 | Block Characterisation — Fingerprint Shapes + Geography | Act V |
| 36 | Novel Finding: Block 1 Fingerprint Flip (+0.34 log-price) | Act V |
| 37 | Exp 12 — pEx-SBM Across Moran's I Spectrum (§6.2 Master Table) | Act V |
| 38 | §6.2 Reframing: Middle Regime (C-ratio 1–3) as Actionable Pre-Diagnostic | Act V |
| 39 | Three Theoretical Gaps (pEPPF→functional CIs, projectivity, SNR ceiling) | Act VI |
| 40 | Empirical Gaps and Open Questions | Act VI |
| 41 | Three Asks for Today's Discussion | Act VI |
| 42 | Summary: What Is Established (four headline claims) | Act VI |
| 43 | Acknowledgments / Q&A | Act VI |
