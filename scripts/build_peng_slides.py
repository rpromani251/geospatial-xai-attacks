"""
build_peng_slides.py
--------------------
Builds a 43-slide python-pptx deck for the 2026-04-23 Peng meeting.
Run from any working directory; all paths are resolved relative to REPO_ROOT.

Usage:
    pip install python-pptx
    python scripts/build_peng_slides.py
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"
DOCS_FIGS = REPO_ROOT / "docs" / "figures"
SBM_FIGS  = RESULTS / "sbm_explainer" / "pex_outputs" / "figures"
SBM_SWP   = RESULTS / "sbm_explainer" / "spectrum_sweep" / "figures"
SBM_DOCS  = REPO_ROOT / "docs" / "sbm_explainer" / "figures"
OUT_DIR   = REPO_ROOT / "docs" / "presentations"
OUT_PATH  = OUT_DIR / "2026-04-23_peng_meeting.pptx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Colours / fonts
# ──────────────────────────────────────────────
BG_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
NAVY         = RGBColor(0x1A, 0x2E, 0x4A)
ACCENT_BLUE  = RGBColor(0x2F, 0x6E, 0xB2)
ACCENT_RED   = RGBColor(0xB2, 0x2F, 0x2F)
MID_GRAY     = RGBColor(0x55, 0x55, 0x55)
LIGHT_GRAY   = RGBColor(0xE8, 0xE8, 0xE8)
HEADER_GRAY  = RGBColor(0x2C, 0x3E, 0x50)
TABLE_HEADER = RGBColor(0x2F, 0x6E, 0xB2)
TABLE_EVEN   = RGBColor(0xF0, 0xF4, 0xF8)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ──────────────────────────────────────────────
# Figure existence check
# ──────────────────────────────────────────────
def fig(path: Path) -> Path | None:
    """Return path if file exists, else None (degrade gracefully)."""
    return path if path.exists() else None

# ──────────────────────────────────────────────
# Presentation object
# ──────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H
blank_layout = prs.slide_layouts[6]  # completely blank

SLIDE_COUNT = 0

# ──────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────

def add_rect(slide, left, top, width, height, fill_rgb=None, line_rgb=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    fill = shape.fill
    if fill_rgb:
        fill.solid()
        fill.fore_color.rgb = fill_rgb
    else:
        fill.background()
    line = shape.line
    if line_rgb:
        line.color.rgb = line_rgb
    else:
        line.fill.background()
    return shape


def add_textbox(slide, left, top, width, height, text,
                font_size=Pt(14), bold=False, italic=False,
                color=NAVY, align=PP_ALIGN.LEFT, word_wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = word_wrap
    tf  = txb.text_frame
    tf.word_wrap = word_wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = font_size
    run.font.bold  = bold
    run.font.color.rgb = color
    if italic:
        run.font.italic = True
    return txb


def add_header(slide, title_text, act_label=""):
    """Dark top bar with title + act label."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.85), fill_rgb=HEADER_GRAY)
    add_textbox(slide, Inches(0.25), Inches(0.08), Inches(9.5), Inches(0.7),
                title_text, font_size=Pt(22), bold=True, color=RGBColor(0xFF,0xFF,0xFF))
    if act_label:
        add_textbox(slide, Inches(10.0), Inches(0.15), Inches(3.1), Inches(0.55),
                    act_label, font_size=Pt(11), color=RGBColor(0xCC,0xDD,0xEE),
                    align=PP_ALIGN.RIGHT)


def add_footer(slide, slide_num):
    add_rect(slide, 0, Inches(7.22), SLIDE_W, Inches(0.28), fill_rgb=LIGHT_GRAY)
    add_textbox(slide, Inches(0.15), Inches(7.23), Inches(8), Inches(0.25),
                "Thursday 2026-04-23 — Peng Meeting",
                font_size=Pt(9), color=MID_GRAY)
    add_textbox(slide, Inches(12.3), Inches(7.23), Inches(0.9), Inches(0.25),
                str(slide_num), font_size=Pt(9), color=MID_GRAY, align=PP_ALIGN.RIGHT)


def bullets(slide, left, top, width, height, items, base_size=Pt(13.5), indent_char="•"):
    """Add a multi-line bullet block. items = list of (indent_level, text)."""
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for level, text in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.level = level
        run = p.add_run()
        bullet_str = ("  " * level) + indent_char + " " + text
        run.text = bullet_str
        run.font.size  = Pt(base_size.pt - level * 1.0)
        run.font.color.rgb = NAVY


def embed_figure(slide, left, top, width, height, fig_path, fallback_text="[Figure not found]"):
    if fig_path and Path(fig_path).exists():
        try:
            slide.shapes.add_picture(str(fig_path), left, top, width, height)
            return True
        except Exception as e:
            pass
    # fallback: grey box + text
    add_rect(slide, left, top, width, height, fill_rgb=LIGHT_GRAY)
    add_textbox(slide, left + Inches(0.1), top + Inches(0.1),
                width - Inches(0.2), height - Inches(0.2),
                fallback_text, font_size=Pt(11), color=MID_GRAY, italic=True)
    return False


def pptx_table(slide, left, top, width, headers, rows,
               col_widths=None, font_size=Pt(10.5)):
    """Insert a formatted table."""
    n_cols = len(headers)
    n_rows = len(rows) + 1  # +1 for header
    row_h  = Inches(0.28)
    tbl_h  = row_h * n_rows
    tbl = slide.shapes.add_table(n_rows, n_cols, left, top, width, tbl_h).table

    # column widths
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw
    else:
        unit = width // n_cols
        for i in range(n_cols):
            tbl.columns[i].width = unit

    # header row
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_HEADER
        p  = cell.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = h
        run.font.bold  = True
        run.font.size  = font_size
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = PP_ALIGN.CENTER

    # data rows
    for i, row in enumerate(rows):
        fill_rgb = TABLE_EVEN if i % 2 == 0 else BG_WHITE
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill_rgb
            p  = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = font_size
            run.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.CENTER
    return tbl


def new_slide(title, act="", has_figure=False):
    """Create a new blank slide with header and footer. Returns (slide, slide_num)."""
    global SLIDE_COUNT
    SLIDE_COUNT += 1
    slide = prs.slides.add_slide(blank_layout)
    add_header(slide, title, act)
    add_footer(slide, SLIDE_COUNT)
    # White body background
    add_rect(slide, 0, Inches(0.85), SLIDE_W, Inches(6.37), fill_rgb=BG_WHITE)
    return slide, SLIDE_COUNT


# Content area constants
BODY_TOP = Inches(0.95)
BODY_H   = Inches(6.1)
FULL_W   = SLIDE_W - Inches(0.4)
LEFT_COL = Inches(0.25)
LEFT_W   = Inches(6.6)   # bullet column when figure on right
RIGHT_L  = Inches(7.0)
RIGHT_W  = Inches(6.1)


# ═══════════════════════════════════════════════════════════
# ACT I — Setup & Recreation
# ═══════════════════════════════════════════════════════════

# Slide 1 — Title
slide, n = new_slide("Geospatial XAI: Adversarial Robustness + pEx-SBM Detection")
add_textbox(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(0.7),
            "Geospatial XAI: Adversarial Robustness + pEx-SBM Detection",
            font_size=Pt(30), bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.5), Inches(2.4), Inches(12.3), Inches(0.5),
            "Progress report for Peng Luo — Thursday, 23 April 2026",
            font_size=Pt(18), color=ACCENT_BLUE, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.5), Inches(3.1), Inches(12.3), Inches(0.4),
            "Robert Romani  |  UROP in Geospatial XAI",
            font_size=Pt(14), color=MID_GRAY, align=PP_ALIGN.CENTER)

acts = [
    "Act I: Setup & Slack/Vascotti Recreation (Exp 1–6)",
    "Act II: No-Location Ablation (Exp 4–4b)",
    "Act III: Moran's I Exploration (Exp 7–11)",
    "Act IV: pEx-SBM as Scaffold-Attack Detector (Exp 9)",
    "Act V: §2 Explanation Primitives + Block Characterization",
    "Act VI: Gaps, Asks & Next Steps",
]
for i, a in enumerate(acts):
    add_textbox(slide, Inches(1.5), Inches(3.8 + i * 0.45), Inches(10), Inches(0.42),
                f"  {i+1}.  {a}", font_size=Pt(13), color=NAVY)

# Slide 2 — Roadmap
slide, n = new_slide("Roadmap — Six Acts", "Overview")
bullet_items = [
    (0, "Six-act narrative covering all experiments run through 2026-04-22"),
    (0, "Act I  — Setup: dataset, explainer stack, scaffold recreation (Exp 1–6)"),
    (0, "Act II — No-location ablation: does removing UTM neutralise the attack? (Exp 4–4b)"),
    (0, "Act III — Moran's I thread: spatial autocorrelation as moderator? (Exp 7–11)"),
    (0, "Act IV — pEx-SBM detection: rung-3 fingerprint encoder, four-dataset replication (Exp 9)"),
    (0, "Act V  — §2 explanation outputs: Ψ, pairwise co-clustering, per-node uncertainty, block characterisation"),
    (0, "Act VI — Three theoretical gaps + three asks for today's discussion"),
    (0, "─────────────────────────────────────────────────────────────────────"),
    (0, "Headline:  scaffold attack hides coordinate-based redlining regardless of Moran's I;"),
    (1, "rung-3 + finite-K pEx is the first detector with non-null layer signal across four datasets;"),
    (1, "per-block structure reveals novel laundering mechanism (Block 1 fingerprint flip)."),
]
bullets(slide, LEFT_COL, BODY_TOP + Inches(0.1), FULL_W, BODY_H, bullet_items)

# Slide 3 — Dataset
slide, n = new_slide("Dataset: Seattle / King County Housing", "Act I — Setup")
left_items = [
    (0, "3,000 King County housing transactions, 80/20 train/test (random_state=42)"),
    (0, "10 baseline features: bathrooms, sqft_living, sqft_lot, grade, condition,"),
    (1, "waterfront, view, age, UTM_X, UTM_Y (EPSG:32610 — metric, so k-NN / radii meaningful)"),
    (0, "Demographics join: ACS 2011–2015 B03002 → pct_black, pct_latino, pct_black_latino"),
    (1, "398 KC tracts; 3,000 records matched to 334 tracts (spatial join, TIGER/Line 2015)"),
    (0, "Target: log_price (natural-log sale price)"),
    (0, "─────────────────────────────────────────────────────────────────────"),
    (0, "Baseline XGBoost (n_estimators=500, max_depth=6, lr=0.1):"),
    (1, "R² = 0.8895  |  RMSE = 0.0765  |  MAE = 0.0574  |  MSE = 0.00585"),
    (0, "UTM_Y + sqft_living dominate mean |SHAP|"),
    (0, "Demographic penalty trigger: pct_black_latino > 0.25 (14% of test set = 84 / 600 pts)"),
]
bullets(slide, LEFT_COL, BODY_TOP + Inches(0.05), LEFT_W, BODY_H, left_items)
embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.5),
             RESULTS / "figures" / "demographics_pct_black_latino.png",
             "[demographics_pct_black_latino.png]")

# Slide 4 — Explainer stack
slide, n = new_slide("Explainer Stack: KernelSHAP, TreeSHAP, GeoSHAP, LIME", "Act I — Setup")
bullet_items = [
    (0, "KernelSHAP — coalition-based Shapley values; background = shap.sample(X_train, 100);"),
    (1, "lambda wrapper prevents SHAP from reading XGBoost internal feature names"),
    (0, "TreeSHAP — exact path-dependent Shapley values in polynomial time (no background needed);"),
    (1, "default for robustness (Exp 5) and worst-case (Exp 6) due to batching speed"),
    (0, "GeoSHAP — novel spatial extension: UTM coords used only to select k=100 geographically"),
    (1, "nearest training points as per-point background; attribution answers 'vs geographic neighbors'"),
    (1, "Consequence: every coalition query lies on the real-data manifold → maximally vulnerable"),
    (0, "LIME — local linear surrogate; 500 Gaussian perturbations per point (σ=0.3 in feature space);"),
    (1, "perturbations straddle the manifold → source of partial resistance to scaffold"),
    (0, "─────────────────────────────────────────────────────────────────────"),
    (0, "All return shape (n_test, n_features) NumPy arrays"),
    (0, "GeoSHAP subsampled to n=20 in robustness experiment (~50× slower than TreeSHAP)"),
]
bullets(slide, LEFT_COL, BODY_TOP + Inches(0.05), FULL_W, BODY_H, bullet_items)

# Slide 5 — Exp 3 Architecture + OOD
slide, n = new_slide("Exp 3 — Adversarial Scaffold Architecture (Slack Recreation)", "Act I — Exp 3")
left_items = [
    (0, "Biased model:  f_biased(x) = f_honest(x) − 0.30 × 1{pct_black_latino > 0.25}"),
    (1, "−0.30 log-price ≈ −26% in linear price for minority-majority tracts"),
    (0, "Innocuous model: XGBoost on 8 safe features (no coords, no demographics); R² = 0.6414"),
    (0, "OOD detectors (Random Forest, n_estimators=100):"),
    (1, "LIME: real vs Gaussian copies (σ=0.3, 30×). Test accuracy = 100.0%"),
    (1, "SHAP: real vs k-means coalition substitutions (10 clusters, 20k samples). Accuracy = 99.5%"),
    (0, "Routing:  real query → f_biased  |  OOD query → ψ_innocuous"),
    (0, "─────────────────────────────────────────────────────────────────────"),
    (0, "Scaffold fidelity vs f_biased on real test data:"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(4.0), left_items)

headers = ["Scaffold", "Pearson r", "MSE", "% → biased"]
rows = [
    ["LIME scaffold",  "0.9996", "6.1e-05", "99.5%"],
    ["SHAP scaffold",  "0.9109", "1.26e-02", "72.3%"],
]
pptx_table(slide, LEFT_COL, Inches(4.3), LEFT_W - Inches(0.1), headers, rows,
           col_widths=[Inches(2.3), Inches(1.4), Inches(1.4), Inches(1.4)])

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "paper_figures" / "fig1_scaffold_attack.png",
             "[fig1_scaffold_attack.png]")

# Slide 6 — Exp 3 Attribution table
slide, n = new_slide("Exp 3 — Attribution Comparison + Spatial Correlation", "Act I — Exp 3")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Mean |attribution| over 200 test points (Honest = 11-feat XGBoost with trigger added)",
            font_size=Pt(11), italic=True, color=MID_GRAY)

headers = ["Feature", "Honest SHAP", "Adv SHAP", "Honest LIME", "Adv LIME", "Adv GeoSHAP"]
rows = [
    ["UTM_Y",            "0.106", "0.050 (−53%)", "0.110", "0.124",       "0.009 (−93%)"],
    ["pct_black_latino", "0.022", "0.035",         "0.026", "0.108",       "0.030"],
    ["sqft_living",      "0.066", "0.082",         "0.092", "0.096",       "0.074"],
    ["grade",            "0.048", "0.075",         "0.087", "0.087",       "0.060"],
    ["age",              "0.012", "0.052",         "0.046", "0.057 (+)",   "0.038"],
]
cws = [Inches(2.1), Inches(1.65), Inches(1.95), Inches(1.65), Inches(1.65), Inches(1.95)]
pptx_table(slide, LEFT_COL, Inches(1.4), FULL_W, headers, rows, col_widths=cws, font_size=Pt(10))

headers2 = ["Model", "Pearson r", "p-value", "Interpretation"]
rows2 = [
    ["Honest",  "−0.506", "2.6e-40",  "Moderate redlining signal"],
    ["Biased",  "−0.727", "9.6e-100", "+0.22 amplification vs honest"],
]
pptx_table(slide, LEFT_COL, Inches(3.7), Inches(8.5), headers2, rows2,
           col_widths=[Inches(1.6), Inches(1.4), Inches(1.8), Inches(3.7)])

left_items = [
    (0, "SHAP: UTM_Y −57%; signal launders into age / grade / bathrooms"),
    (0, "GeoSHAP: most compromised — geographic backgrounds trigger biased branch on every query"),
    (0, "LIME: partially resistant — Gaussian perts straddle manifold, leak through OOD detector"),
]
bullets(slide, LEFT_COL, Inches(4.8), FULL_W, Inches(1.7), left_items)

# Slide 7 — Exp 5 Robustness matrix
slide, n = new_slide("Exp 5 — Vascotti 3×5 Robustness Matrix (Spearman ρ)", "Act I — Exp 5")
left_items = [
    (0, "Three neighbourhood generators (50 neighbours per reference, n=50 test pts):"),
    (1, "Random: Gaussian noise (σ = 0.05 × train std) — off-manifold"),
    (1, "Medoid: 200 nearest real training pts in standardised feature space — on-manifold"),
    (1, "Spatial: α=0.5 blend of geo-distance + feature-distance — on-manifold + geographic"),
    (0, "Prediction filter: discard neighbours with |Δlog_price| > 0.15"),
    (0, "Robustness metric: mean Spearman ρ of feature rankings vs reference"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(3.0), left_items)

headers = ["Level \\ Explainer", "TreeSHAP", "KernelSHAP", "GeoSHAP*", "LIME", "Ensemble"]
rows = [
    ["Random",  "0.845", "0.847", "0.720", "0.523", "0.763"],
    ["Medoid",  "0.515", "0.531", "0.336", "0.473", "0.635"],
    ["Spatial", "0.520", "0.537", "0.329", "0.481", "0.633"],
]
cws = [Inches(2.2)]+[Inches(1.7)]*5
pptx_table(slide, LEFT_COL, Inches(3.9), LEFT_W - Inches(0.1), headers, rows, col_widths=cws)

add_textbox(slide, LEFT_COL, Inches(5.35), LEFT_W, Inches(0.3),
            "* GeoSHAP n=20 only. Ensemble: TreeSHAP + KernelSHAP + LIME (sign-disagreement penalty λ=0.15).",
            font_size=Pt(10), italic=True, color=MID_GRAY)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "robustness" / "figures" / "heatmap.png",
             "[heatmap.png]")

# Slide 8 — Exp 6 Worst-case x*
slide, n = new_slide("Exp 6 — Worst-Case x* Attack (No Adversarial Machinery)", "Act I — Exp 6")
left_items = [
    (0, "x* = argmax_{x ∈ N_ε(y)} ‖attr(x) − attr(y)‖₂"),
    (1, "N_ε(y) = real training points within ε_geo metres (UTM Euclidean)"),
    (1, "x* is always a real house — no OOD detector, no scaffold, no synthetic perts"),
    (0, "Implementation: batch TreeSHAP on reference + up to 200 candidates"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(2.3), left_items)

headers = ["ε_geo (m)", "Mean worst-case L2", "Std", "Mean candidates"]
rows = [
    ["500",    "0.165", "0.079", "3.6"],
    ["1,000",  "0.228", "0.083", "12.4"],
    ["2,000",  "0.275", "0.080", "39.3"],
    ["5,000",  "0.382", "0.083", "150.0"],
    ["10,000", "0.411", "0.072", "195.9"],
]
pptx_table(slide, LEFT_COL, Inches(3.05), LEFT_W - Inches(0.1), headers, rows,
           col_widths=[Inches(1.5), Inches(2.0), Inches(1.3), Inches(1.9)])

add_textbox(slide, LEFT_COL, Inches(5.1), LEFT_W, Inches(0.5),
            "At ε=5,000 m: worst-case L2 = 0.382 vs mean-neighbor L2 = 0.147 → ratio 2.60×\n"
            "grade and sqft_living account for ~75% of L2 norm (split-crossing at grade≥8)",
            font_size=Pt(12), color=ACCENT_RED)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "worst_case" / "figures" / "eps_sweep.png",
             "[eps_sweep.png]")

# Slide 9 — Trustworthiness
slide, n = new_slide("Trustworthiness: 1/50 Points Pass ρ ≥ 0.8 Threshold", "Act I — Exp 5")
bullet_items = [
    (0, "Threshold r_th = 0.8 (strict on-manifold standard)"),
    (0, "Spatial × TreeSHAP: 1 of 50 test points clears the threshold"),
    (0, "5-NN regressor trained on (TreeSHAP attribution, prediction) to predict ρ:"),
    (1, "Also finds essentially zero robust points — the k-NN predictor agrees"),
    (0, "Interpretation:"),
    (1, "Under a strict on-manifold standard virtually no explanation in this housing model is trustworthy"),
    (1, "Average ρ = 0.520 means 'somewhat consistent'; but >90% of points fail even a moderate threshold"),
    (0, "Complement with worst-case (Exp 6): average already bad; worst-case 2.6× worse"),
    (0, "Design implication: any fairness audit that relies on explanation consistency"),
    (1, "across comparable properties is operating on unreliable ground"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, BODY_H, bullet_items)
embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "robustness" / "figures" / "trustworthiness_dist.png",
             "[trustworthiness_dist.png]")


# ═══════════════════════════════════════════════════════════
# ACT II — No-Location Ablation
# ═══════════════════════════════════════════════════════════

# Slide 10 — Exp 4
slide, n = new_slide("Exp 4 — No-Location Ablation: R² 0.89 → 0.62", "Act II — Exp 4")
left_items = [
    (0, "Remove UTM_X + UTM_Y → train on 8 structural features"),
    (0, "Same XGBoost hyperparameters and 80/20 split"),
    (0, "R² drops 0.8895 → 0.6174 (−0.272): coordinates supply ~27 R²-points, ≈ ⅓ of explanatory power"),
    (0, "Feature importance shift:"),
    (1, "age rises to 3rd place across all SHAP variants (neighbourhood-vintage proxy for location)"),
    (1, "LIME upweights view / condition / waterfront — binary amenity features create large local swings"),
    (1, "grade and sqft_living remain top-2 overall"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(4.0), left_items)

headers = ["Feature", "TreeSHAP", "KernelSHAP", "LIME"]
rows = [
    ["grade",      "0.084", "0.082", "0.147"],
    ["sqft_living","0.077", "0.084", "0.144"],
    ["age",        "0.058", "0.063", "0.021"],
    ["sqft_lot",   "0.032", "0.034", "0.035"],
    ["view",       "0.013", "0.013", "0.092"],
    ["condition",  "0.011", "0.012", "0.051"],
    ["waterfront", "0.002", "0.002", "0.049"],
]
pptx_table(slide, LEFT_COL, Inches(4.4), LEFT_W - Inches(0.1), headers, rows,
           col_widths=[Inches(2.1), Inches(1.5), Inches(1.55), Inches(1.35)])

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "no_location" / "figures" / "ranking_comparison.png",
             "[ranking_comparison.png]")

# Slide 11 — Exp 4b
slide, n = new_slide("Exp 4b — Scaffold Without Coordinates Still Works", "Act II — Exp 4b")
left_items = [
    (0, "Scaffold rebuilt on 9 features (8 structural + pct_black_latino); UTM_X/Y removed from all models"),
    (0, "OOD detectors and penalty identical to Exp 3"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(1.3), left_items)

headers = ["Scaffold", "OOD accuracy", "Fidelity r", "% → biased"]
rows = [
    ["LIME", "100.0%", "1.0000", "99.8%"],
    ["SHAP", "99.3%",  "0.9834", "78.8%"],
]
pptx_table(slide, LEFT_COL, Inches(2.15), LEFT_W, headers, rows,
           col_widths=[Inches(1.6), Inches(1.65), Inches(1.65), Inches(1.65)])

left2_items = [
    (0, "Bottom line: OOD routing is the attack mechanism — NOT coordinate laundering"),
    (1, "Without coordinates, pct_black_latino becomes *more* visible (adv SHAP = 0.044 vs 0.035 in Exp 3)"),
    (1, "grade (0.090) and age (0.063) absorb the penalty variance instead of UTM_Y"),
    (0, "Spatial correlation amplification persists: biased r = −0.600 vs honest no-loc r = −0.289 (+0.31)"),
    (0, "LIME fidelity is perfect (r=1.000): Gaussian perturbations are trivially distinguishable regardless"),
]
bullets(slide, LEFT_COL, Inches(3.4), LEFT_W, Inches(3.0), left2_items)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "no_location" / "figures" / "baseline_vs_nolocation.png",
             "[baseline_vs_nolocation.png]")


# ═══════════════════════════════════════════════════════════
# ACT III — Moran's I Exploration
# ═══════════════════════════════════════════════════════════

# Slide 12 — Motivation
slide, n = new_slide("Moran's I Thread — Motivation (Peng's 2026-04-16 Ask)", "Act III — Motivation")
bullet_items = [
    (0, "All prior experiments used pct_black_latino as trigger (Moran's I ≈ 0.83 — strongly clumpy)"),
    (0, "Peng's question: does the scaffold's OOD detector rely on spatial clumpiness?"),
    (1, "If yes → attacks on weakly autocorrelated features should fail or leak more"),
    (0, "Three-stage investigation:"),
    (1, "Exp 7: survey Moran's I for all 11 non-coordinate features"),
    (1, "Exp 8 / 8b: sweep scaffold attack across I spectrum + normalise by base importance"),
    (1, "Exp 10–11: test GeoSHAP-reversal hypothesis with low-I binary triggers"),
    (0, "Global Moran's I:  I = (zᵀWz) / (zᵀz)  where W is row-standardised 8-NN spatial weights"),
    (1, "499-permutation pseudo-p-value; all 11 features significant"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 13 — Exp 7 Moran's I survey
slide, n = new_slide("Exp 7 — Moran's I Survey Across 11 Features", "Act III — Exp 7")
headers = ["Feature", "Moran's I", "Level"]
rows = [
    ["pct_black",        "0.850", "very high"],
    ["pct_black_latino", "0.834", "very high"],
    ["pct_latino",       "0.756", "high"],
    ["sqft_lot",         "0.469", "medium"],
    ["grade",            "0.365", "medium"],
    ["age",              "0.317", "medium"],
    ["sqft_living",      "0.291", "low-medium"],
    ["view",             "0.242", "low"],
    ["bathrooms",        "0.183", "low"],
    ["condition",        "0.090", "very low"],
    ["waterfront",       "0.051", "very low"],
]
pptx_table(slide, LEFT_COL, BODY_TOP, LEFT_W, headers, rows,
           col_widths=[Inches(2.5), Inches(1.6), Inches(2.4)])
add_textbox(slide, LEFT_COL, Inches(5.6), LEFT_W, Inches(0.4),
            "All p < 0.05 (permutation). Spectrum spans 0.051–0.850.",
            font_size=Pt(11), italic=True, color=MID_GRAY)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "morans_i" / "morans_i_barplot.png",
             "[morans_i_barplot.png]")

# Slide 14 — Exp 8 sweep
slide, n = new_slide("Exp 8 — Feature Sweep: OOD Accuracy Flat, Visibility Rises with I", "Act III — Exp 8")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Visibility = mean |adv_attr_biasfeature| / mean Σ|adv_attr|  (share of total attribution going to biased feature)",
            font_size=Pt(10.5), italic=True, color=MID_GRAY)

headers = ["Feature", "Role", "Moran's I", "SHAP vis", "LIME vis", "% penalised", "SHAP OOD", "LIME OOD"]
rows = [
    ["waterfront",       "structural",   "0.051", "0.014", "0.106", "0.8%",  "99.3%", "100%"],
    ["view",             "structural",   "0.242", "0.070", "0.160", "9.7%",  "99.3%", "100%"],
    ["sqft_lot",         "structural",   "0.469", "0.134", "0.175", "25.3%", "99.3%", "100%"],
    ["pct_latino",       "demographic",  "0.756", "0.136", "0.190", "24.7%", "99.4%", "100%"],
    ["pct_black",        "demographic",  "0.850", "0.140", "0.207", "23.2%", "99.6%", "100%"],
]
cws = [Inches(1.55), Inches(1.45), Inches(1.2), Inches(1.0), Inches(1.0), Inches(1.3), Inches(1.15), Inches(1.1)]
pptx_table(slide, LEFT_COL, Inches(1.4), FULL_W, headers, rows, col_widths=cws, font_size=Pt(10))

left_items = [
    (0, "Finding 1: OOD accuracy flat at 99.3–100% across full I spectrum — attack feasibility is I-invariant"),
    (0, "Finding 2: visibility increases monotonically with I — clumpy features leak MORE, not less"),
    (0, "Finding 3: two regimes — structural (in baseline, amplitude suppression) vs demographic (off-baseline, proxy laundering)"),
    (0, "Finding 4: waterfront under-penalised (P75=0, only 0.8% triggered)"),
    (0, "Finding 5: Spearman ρ(visibility, I) = +0.94 (raw) — see next slide for base-importance correction"),
]
bullets(slide, LEFT_COL, Inches(3.55), FULL_W, Inches(2.9), left_items)

# Slide 15 — Exp 8b base-importance normalisation
slide, n = new_slide("Exp 8b — Base-Importance Normalisation: I-Trend is an Artifact", "Act III — Exp 8b")
bullet_items = [
    (0, "A4 confound: biasing a high-importance feature inflates adversarial attribution regardless of I"),
    (1, "Exp 8's Spearman ρ(raw visibility, I) = +0.94 could just reflect importance headroom"),
    (0, "Exp 8b fix: normalise visibility by honest base-model importance (same feature, honest model)"),
    (1, "Normalised visibility = adv_attr / honest_attr — controls for 'how important was this feature already?'"),
    (0, "Result: Spearman ρ(normalised visibility, I) = −0.43  (p = 0.40)"),
    (1, "Not significant — the I-trend is entirely explained by base-model importance"),
    (1, "Structural features (grade, sqft_lot) have HIGH honest importance → numerically small normalised visibility"),
    (1, "Demographic features (pct_latino, pct_black) have ZERO honest importance → infinite normalised visibility"),
    (0, "Conclusion: Moran's I does NOT moderate attack feasibility or explainer foolability"),
    (1, "OOD routing is I-invariant; the apparent trend is a feature-role / importance artifact"),
]
bullets(slide, LEFT_COL, BODY_TOP + Inches(0.05), FULL_W, BODY_H, bullet_items)

# Slide 16 — 2026-04-16 proposal
slide, n = new_slide("2026-04-16 Proposal: Cross-Feature GeoSHAP Reversal Hypothesis", "Act III — Design")
bullet_items = [
    (0, "Observation: in Exp 3, GeoSHAP is the MOST compromised explainer (UTM_Y collapses −93%)"),
    (1, "Reason: geographic background always samples real-data manifold → hits biased branch"),
    (0, "Hypothesis: for a trigger with LOW Moran's I, GeoSHAP should REVERSE — become most RESISTANT"),
    (1, "Low-I trigger → geographically dispersed → geographic background does NOT cluster around trigger"),
    (1, "GeoSHAP would see trigger in background → anchor attribution exposed, not laundered"),
    (0, "Test plan: find a binary trigger feature with:"),
    (1, "Low Moran's I (to create dispersed spatial structure)"),
    (1, "Reasonable coverage (≥5% of test set triggered)"),
    (1, "Real price signal (honest model attributes non-zero weight to it)"),
    (0, "Candidates investigated: FEMA 100-yr flood zone (Exp 10), 500-yr extension (Exp 10b),"),
    (1, "KC parcel dataset 59k (Exp 10c), Geologic Hazard (Exp 10d), SPS school zone (Exp 11)"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 17 — Exp 11 SPS
slide, n = new_slide("Exp 11 — SPS School Zone: Moran's I = 0.80 (Untestable)", "Act III — Exp 11")
bullet_items = [
    (0, "Dataset: 1,231 housing records with Seattle Public Schools zone indicator (in_low_score_zone)"),
    (0, "Computed Moran's I for in_low_score_zone: 0.80"),
    (0, "Problem: zone boundaries are administratively contiguous → unavoidable spatial clustering"),
    (1, "Binary zone indicators at housing density cannot achieve low-I — the geometry prevents it"),
    (0, "Scaffold still works: SHAP 18× trigger inflation (0.002 → 0.036), LIME 14× (0.010 → 0.144)"),
    (0, "UTM_Y drops −64% under SHAP (consistent with 3k-set experiments)"),
    (0, "Verdict: school zone binaries are not a viable low-I probe"),
    (1, "Hypothesis is untestable via this route: need features that are genuinely spatially dispersed"),
    (1, "FEMA flood zones are the natural next candidate (elevation-driven, not administratively bounded)"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 18 — Exp 10 FEMA 100yr
slide, n = new_slide("Exp 10 — FEMA 100-yr Flood Zone: Coverage 0.2% (Untestable)", "Act III — Exp 10")
bullet_items = [
    (0, "Trigger: in_floodplain (FEMA 100-yr flood zone binary indicator)"),
    (0, "Moran's I ≈ −0.0015 — confirmed low (spatially dispersed/random)"),
    (0, "Coverage: 0.2% of 3k test set → only ~6 triggered test points"),
    (1, "Penalty of −0.30 log-price on 6 points provides essentially no detectable signal"),
    (0, "Honest SHAP attribution ≈ 0.000 (model ignores in_floodplain as predictor)"),
    (0, "Adversarial LIME raises it to 0.067, but statistical confidence is negligible"),
    (0, "Verdict: untestable — coverage too thin for any conclusive analysis"),
    (1, "The honest model cannot use in_floodplain: it has near-zero correlation with log_price in Seattle"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 19 — Exp 10b 500yr
slide, n = new_slide("Exp 10b — 500-yr Flood Extension: Coverage 0.5% (Still Untestable)", "Act III — Exp 10b")
bullet_items = [
    (0, "Extension of Exp 10 to 500-yr flood zone (wider buffer)"),
    (0, "Coverage: 0.5% of 3k test set — still too thin"),
    (0, "Honest SHAP ≈ 0.000 (same as 100-yr; model ignores)"),
    (0, "Adversarial LIME: 0.090 (nearly 3× honest LIME value of 0.033)"),
    (1, "Scaffold can manufacture LIME attribution even on null-signal features via OOD gating"),
    (0, "Conclusion: flood zones in Seattle 3k sample have no real price signal at any buffer"),
    (1, "ACS-matched census tracts don't partition flood risk well in King County"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 20 — Exp 10c KC parcel
slide, n = new_slide("Exp 10c — KC Parcel Dataset (59k Records): Coverage 0.47% at 20× Scale", "Act III — Exp 10c")
bullet_items = [
    (0, "Motivation: 3k sample underrepresents flood zones — try full KC parcel dataset (59,024 records)"),
    (0, "Baseline R² = 0.7625 (10-feat XGBoost on parcel data; slightly weaker feature quality)"),
    (0, "Coverage: 0.47% even at 20× scale — flood-zone parcels are systematically under-transacted"),
    (1, "Transaction data is biased toward non-flood properties (insurance/financing disincentives)"),
    (0, "Honest SHAP ≈ 0.000 (feature carries no price signal in parcel transactions)"),
    (0, "Adversarial LIME = 0.150 — LIME sees through the scaffold more in this dataset"),
    (0, "UTM_Y redistribution smaller: −0.027 vs −0.056 in 3k-set (parcel data has noisier coordinates)"),
    (0, "Verdict: transaction-based housing data structurally cannot capture flood-zone properties"),
    (1, "Need administrative or assessed value data to test this hypothesis"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 21 — Exp 10d Geologic Hazard
slide, n = new_slide("Exp 10d — Geologic Hazard: Coverage 5.1%, 13/200 Explained Points", "Act III — Exp 10d")
left_items = [
    (0, "Trigger: in_geologic_hazard (landslide/erosion/seismic risk binary, KC GIS)"),
    (0, "Moran's I = 0.25 — genuinely moderate (topography-driven, not administratively bounded)"),
    (0, "Coverage: 5.1% of 3k test set → 13 / 200 explained points triggered"),
    (0, "11-feat honest R² = 0.8875 (−0.0020 vs 10-feat baseline — minimal colinearity)"),
    (0, "Attribution results:"),
    (1, "Honest SHAP ≈ 0.000 (model ignores hazard — no price signal in Seattle data)"),
    (1, "Adv LIME = 0.186 vs honest LIME = 0.022 → factor of 8.5× (most dramatic LIME inflation)"),
    (1, "Adv SHAP = 0.013 (SHAP more resistant)"),
    (1, "UTM_Y drops −57% under SHAP — consistent with all 3k-set experiments"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(5.8), left_items)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             RESULTS / "scaffold_geologic_hazard" / "figures" / "fig_geoshap_reversal.png",
             "[fig_geoshap_reversal.png]")

# Slide 22 — GeoSHAP reversal verdict
slide, n = new_slide("GeoSHAP Reversal Verdict: DISCONFIRMED", "Act III — Verdict")
bullet_items = [
    (0, "Hypothesis: low-I trigger → GeoSHAP reverses from most-compromised to most-resistant"),
    (0, "Exp 10d result: GeoSHAP trigger attribution = 0.011 (still suppressed); UTM_Y collapses −92%"),
    (1, "GeoSHAP remains the MOST compromised explainer even with I=0.25 trigger"),
    (0, "Why the hypothesis fails:"),
    (1, "GeoSHAP's background is geographically local — anchored to 100 nearest-UTM training points"),
    (1, "The local neighbourhood contains BOTH triggered and non-triggered properties (I=0.25 is moderate, not zero)"),
    (1, "Even at I≈0, geologic hazard = 0 for ~95% of neighbours → anchor still dominated by untriggered properties"),
    (1, "The OOD routing mechanism operates in feature space, not geographic space — I is irrelevant"),
    (0, "Revised finding (see Slide 23): Moran's I does NOT moderate attack feasibility via any route"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 23 — Revised finding
slide, n = new_slide("Revised Finding: I-Invariant Attack + LIME as Honest Mirror", "Act III — Synthesis")
bullet_items = [
    (0, "Central result of the Moran's I thread:"),
    (1, "OOD routing is I-invariant (Exp 8): attack feasibility does not depend on spatial clustering"),
    (1, "Visibility trend (Exp 8) is a base-importance artifact, not a real I-effect (Exp 8b normalisation)"),
    (1, "GeoSHAP-reversal hypothesis fails (Exp 10d): geographic background does not flip resistance"),
    (0, "What DOES predict explainer foolability?"),
    (1, "Feature role (structural = in baseline / demographic = off baseline)"),
    (1, "OOD detector strictness vs perturbation scale (partial routing mixture required for detection)"),
    (0, "LIME's role re-framed: LIME is NOT resistant because it is cleverer —"),
    (1, "it shows the biased model's behavior on real inputs (anchor visible through its Gaussian perts)"),
    (1, "LIME's honest reflection of what the biased model does is the detection signal, not explainer quality"),
    (0, "Next step: use this signal systematically via pEx-SBM (Act IV)"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 24 — Honest vs Adv laundering comparison
slide, n = new_slide("Honest vs. Adversarial Laundering: 5–7× the Natural Redistribution", "Act III — Synthesis")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Honest UTM_Y shift when trigger added to honest model vs adversarial scaffold shift (SHAP)",
            font_size=Pt(11), italic=True, color=MID_GRAY)

headers = ["Exp", "Trigger", "Honest UTM_Y Δ (10→11 feat)", "Adv UTM_Y", "Adv−Honest11 Δ"]
rows = [
    ["Exp 3",  "pct_black_latino",  "−0.010",   "0.050", "−0.056"],
    ["Exp 10", "in_floodplain",     "≈ +0.001",  "0.050", "−0.067"],
    ["Exp 10b","in_flood_500yr",    "≈ +0.002",  "0.050", "−0.068"],
    ["Exp 10c","in_floodplain(59k)","≈ 0.000",   "0.227", "−0.027"],
    ["Exp 10d","in_geologic_hazard","≈ +0.001",  "0.050", "−0.067"],
    ["Exp 11", "in_low_score_zone", "≈ 0.000",   "0.027", "−0.049"],
]
pptx_table(slide, LEFT_COL, Inches(1.4), FULL_W, headers, rows,
           col_widths=[Inches(1.1), Inches(2.3), Inches(2.7), Inches(1.5), Inches(2.5)],
           font_size=Pt(10.5))

left_items = [
    (0, "Adding trigger to HONEST model: UTM_Y drops ≤0.010 for pct_black_latino, ≤0.002 for binary triggers"),
    (0, "Scaffold redistributes: 5–7× more than honest inclusion"),
    (0, "Confirms genuine laundering — scaffold mechanically suppresses UTM_Y via innocuous routing"),
]
bullets(slide, LEFT_COL, Inches(4.5), FULL_W, Inches(2.2), left_items)

embed_figure(slide, Inches(9.5), Inches(4.5), Inches(3.7), Inches(2.5),
             DOCS_FIGS / "fig_honest_vs_adv_off_baseline.png",
             "[fig_honest_vs_adv_off_baseline.png]")


# ═══════════════════════════════════════════════════════════
# ACT IV — pEx-SBM as Scaffold-Attack Detector
# ═══════════════════════════════════════════════════════════

# Slide 25 — Motivation
slide, n = new_slide("Motivation: Detection Signal Must Live in Scaffold's Routing Mixture", "Act IV — Motivation")
bullet_items = [
    (0, "Exps 1–8: established THAT scaffold attacks hide coordinate-based redlining"),
    (0, "Exps 7–11: feature-side moderators (Moran's I) do NOT predict foolability"),
    (0, "Complementary question: can we DETECT the scaffold without access to its internals?"),
    (1, "No access to biased model, OOD detector, or routing logic"),
    (1, "Only access: black-box scaffold predictions on real test points + arbitrary perturbations"),
    (0, "Key insight: a strict OOD detector (which scaffold REQUIRES to hide bias) creates"),
    (1, "a distinctive one-anchor-plus-K-refinements pattern in predictions"),
    (1, "Real input x_i → biased branch (anchor); perturbations δ → innocuous branch (refinements)"),
    (0, "If points sharing the same protected-attribute layer share similar anchor/refinement patterns,"),
    (1, "a partially exchangeable SBM can recover that layer from the fingerprint graph alone"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 26 — pEx-SBM primer
slide, n = new_slide("pEx-SBM Primer — §2 in Five Bullets", "Act IV — Method")
bullet_items = [
    (0, "Durant, Gaffi, Lijoi, Prünster (JASA 2025, arXiv:2410.10619) — 'Partially exchangeable SBMs'"),
    (1, "First SBM whose exchangeability group matches what fairness auditing needs:"),
    (1, "nodes exchangeable WITHIN a protected subpopulation (color/layer), not across"),
    (0, "Partition prior: partially exchangeable partition probability function (pEPPF)"),
    (1, "Hierarchical Dirichlet or Pitman-Yor — closed-form posteriors from Camerlenghi et al. (AoS 2019)"),
    (0, "Layers = audit attribute (e.g., pct_black_latino > 0.25)"),
    (1, "Block-probability matrix Ψ_{h,k} = P(node in layer h assigned to block k)"),
    (0, "Explanation outputs (§2): Ψ matrix, pairwise co-clustering posterior P(z_i = z_j | G, ℓ) with CI,"),
    (1, "per-node posterior entropy H_i over block assignment"),
    (0, "Our implementation: finite-K Dirichlet-Categorical pEx (K=4), Gibbs sampler (N=3000, burn=1000)"),
    (1, "H-DP pEx (Durante's original, K inferred ≈ 10) as extension — see §3 summary"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 27 — Encoder ladder
slide, n = new_slide("The Encoder Ladder: Rungs 1–2 Null, Rung 3 Fires", "Act IV — Method")
left_items = [
    (0, "Three encoder rungs tested; first two null on all configs:"),
    (0, "Rung 1: SHAP / LIME attribution vectors (original attempt)"),
    (1, "All Δ(pEx − plain) = 0 — attribution vectors are posterior-blind (regime b)"),
    (1, "Why: attribution vectors reflect BOTH branches mixed; no routing-mixture geometry"),
    (0, "Rung 2: XGBoost honest-booster leaf indices"),
    (1, "Δ = 0 again — honest-booster leaves don't reflect scaffold gate"),
    (0, "Rung 3: behavioral fingerprint under scaffolded model:"),
    (1, "e_i = [f_scaffold(x_i),  f_scaffold(x_i+δ_1), …, f_scaffold(x_i+δ_K)]"),
    (1, "σ = 0.1 × feat_std Gaussian perts; K=50; 8-NN cosine graph; binarised adjacency"),
    (0, "Why rung 3 fires: strict LIME OOD detector routes 100% of δ_k to innocuous branch"),
    (1, "e_i = [f_biased(x_i),  f_innocuous(x_i+δ_1), …, f_innocuous(x_i+δ_K)]"),
    (1, "One gate-exposed anchor + K smooth gate-hidden refinements → layer-correlated geometry"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, BODY_H, left_items)

# sanity check table right
headers = ["Config", "plain NMI", "pEx NMI", "Δ", "C-ratio"]
rows = [
    ["honest XGBoost",        "0.007", "0.009", "+0.002", "1.03"],
    ["BiasedModel only",      "0.004", "0.015", "+0.011", "1.04"],
    ["InnocuousModel only",   "0.007", "0.015", "+0.007", "1.06"],
    ["scaffold (LIME OOD) ★", "0.287", "0.334", "+0.048", "2.50"],
]
pptx_table(slide, RIGHT_L, BODY_TOP, RIGHT_W, headers, rows,
           col_widths=[Inches(2.1), Inches(1.0), Inches(0.95), Inches(0.85), Inches(1.0)],
           font_size=Pt(10))
add_textbox(slide, RIGHT_L, Inches(3.0), RIGHT_W, Inches(0.35),
            "★ Only scaffold routing mixture produces signal",
            font_size=Pt(10.5), italic=True, color=ACCENT_RED)

# Slide 28 — Exp 9 main result
slide, n = new_slide("Exp 9 — Housing Gauss+LIME: Δ NMI = +0.048, p = 1.8e-9", "Act IV — Exp 9")
left_items = [
    (0, "20-seed CI on Gauss+LIME fingerprint (housing, pct_black_latino layer, K=4):"),
    (1, "Δ = 0.043 ± 0.018 (SEM 0.004)  |  95% CI [+0.017, +0.076]"),
    (1, "t-test vs Δ=0:  t = 10.68,  p = 1.8 × 10⁻⁹"),
    (0, "Four encoder × scaffold cells:"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(2.5), left_items)

headers = ["Encoder", "Scaffold", "log BF", "plain NMI", "pEx NMI", "Δ", "C-ratio"]
rows = [
    ["Gaussian",  "LIME", "+145", "0.287", "0.334", "+0.048 ★", "2.51"],
    ["Gaussian",  "SHAP", "−2",   "0.004", "0.003", "−0.001",   "0.99"],
    ["Targeted",  "LIME", "+2",   "0.063", "0.063", "0.000",    "1.16"],
    ["Targeted",  "SHAP", "−3",   "0.012", "0.012", "0.000",    "1.01"],
]
pptx_table(slide, LEFT_COL, Inches(3.2), LEFT_W, headers, rows,
           col_widths=[Inches(1.3), Inches(1.2), Inches(1.0), Inches(1.2), Inches(1.2), Inches(1.35), Inches(1.05)],
           font_size=Pt(10.5))

add_textbox(slide, LEFT_COL, Inches(5.35), LEFT_W, Inches(0.65),
            "★ Only Gauss+LIME cell in the middle regime (C-ratio 2.51 — neither blind ~1 nor saturated ≥4)",
            font_size=Pt(11), italic=True, color=ACCENT_RED)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             SBM_DOCS / "k_scaling.png",
             "[k_scaling.png]")

# Slide 29 — Four-dataset replication
slide, n = new_slide("Four-Dataset Replication: Invariant Δ > 0 iff Partial Routing Mixture", "Act IV — Replication")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Best cell per dataset (anchor = the partial-routing cell; role reversal is detector-manifold dependent)",
            font_size=Pt(11), italic=True, color=MID_GRAY)

headers = ["Dataset", "Scaffold", "σ", "plain NMI", "pEx NMI", "Δ", "C-ratio", "routing"]
rows = [
    ["Housing (reg.)", "LIME", "0.1×std", "0.287", "0.334", "+0.048", "2.51", "0% real"],
    ["COMPAS (clf.)",  "SHAP", "0.1",     "0.303", "0.340", "+0.038", "1.94", "70% real"],
    ["German (clf.)",  "LIME", "1.0",     "0.115", "0.172", "+0.057", "1.36", "0.1% real"],
    ["German (clf.)",  "SHAP", "1.0",     "0.048", "0.119", "+0.071", "1.05", "17% real"],
    ["CC psi_two(reg)","LIME", "1.0",     "0.124", "0.699", "+0.575", "1.43", "0.0% real"],
]
cws = [Inches(1.7), Inches(1.1), Inches(1.0), Inches(1.15), Inches(1.05), Inches(1.05), Inches(1.15), Inches(1.55)]
pptx_table(slide, LEFT_COL, Inches(1.4), FULL_W, headers, rows, col_widths=cws, font_size=Pt(10))

bullet_items = [
    (0, "Role reversal: on Housing, LIME scaffold fires; on COMPAS, SHAP scaffold fires"),
    (1, "LIME OOD on COMPAS (post-StandardScaler) rejects 100% of σ=0.1 perts → all-innocuous → null"),
    (1, "SHAP OOD (k-means) is moderately permissive → partial mixture → signal"),
    (0, "Invariant: Δ > 0 iff OOD produces partial routing mixture (not all-biased, not all-innocuous)"),
    (0, "CC psi_two (XOR innocuous): Δ=+0.575 — XOR amplifies pEx signal (refinements more distinctive)"),
    (0, "Coverage: regression + 3 classifications; custom + Slack-original scaffolds; d=11 to d=102"),
]
bullets(slide, LEFT_COL, Inches(3.85), FULL_W, Inches(2.8), bullet_items)

# Slide 30 — Mechanism anchor-dominance erosion
slide, n = new_slide("Mechanism: Anchor-Dominance Erosion (K-Sweep Confirmation)", "Act IV — Mechanism")
left_items = [
    (0, "Cosine decomposition of fingerprint e_i:"),
    (1, "e_i · e_j / (‖e_i‖ ‖e_j‖) ≈ m_i m_j / (s_i s_j)  +  a_i a_j / (K s_i s_j)  +  O(1/√K)"),
    (1, "a_i = anchor value; m_i = mean of K refinements; s_i = ‖e_i‖"),
    (0, "At small K: anchor term O(1/K) is large → anchor dominates → plain SBM sees layer"),
    (0, "At large K: refinement mean-field m_i dominates at O(1) — layer-independent → plain collapses"),
    (0, "pEx holds via conditional prior even when plain fails"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(3.2), left_items)

headers = ["K", "plain NMI", "pEx NMI", "Δ", "C-ratio"]
rows = [
    ["5",  "0.333", "0.334", "+0.001", "21.4"],
    ["10", "0.187", "0.307", "+0.120", "3.24"],
    ["20", "0.082", "0.555", "+0.473", "1.25"],
    ["40", "0.050", "0.491", "+0.441", "1.18"],
    ["80", "0.020", "0.426", "+0.406", "1.05"],
]
add_textbox(slide, LEFT_COL, Inches(3.4), LEFT_W, Inches(0.3),
            "CC psi_two LIME σ=1.0 K-sweep:", font_size=Pt(11), bold=True, color=NAVY)
pptx_table(slide, LEFT_COL, Inches(3.8), LEFT_W - Inches(0.1), headers, rows,
           col_widths=[Inches(0.7), Inches(1.4), Inches(1.1), Inches(1.1), Inches(1.1)])

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(2.8),
             SBM_DOCS / "tanh_sweep.png",
             "[tanh_sweep.png]")
add_textbox(slide, RIGHT_L, Inches(3.8), RIGHT_W, Inches(0.3),
            "Tanh probe (2nd mechanism confirmation):", font_size=Pt(11), bold=True, color=NAVY)
add_textbox(slide, RIGHT_L, Inches(4.15), RIGHT_W, Inches(1.8),
            "β=0.1–2: saturated-plain regime (m_i nearly constant → anchor dominates → plain suffices)\n"
            "β=5: Δ=+0.071 (plain 0.582, pEx 0.653)\n"
            "β=10: Δ=+0.257 (plain 0.394, pEx stays 0.651)\n"
            "Plain monotonically degrades; pEx flat — two-regime prediction confirmed",
            font_size=Pt(11), color=NAVY)


# ═══════════════════════════════════════════════════════════
# ACT V — §2 Explanation Primitives + Block Characterisation
# ═══════════════════════════════════════════════════════════

# Slide 31 — §2 outputs overview
slide, n = new_slide("Act V — §2 Explanation Outputs: Three Promised Primitives", "Act V — §2")
bullet_items = [
    (0, "The pEx-SBM paper proposal §2 promises three concrete explanation outputs:"),
    (0, "Output 1: Block-probability matrix Ψ_{h,k}"),
    (1, "For layer h and block k: probability a node in layer h is assigned to block k"),
    (1, "Closed-form posterior from aligned Gibbs trace (label-switching via Hungarian assignment)"),
    (0, "Output 2: Pairwise co-clustering posterior P(z_i = z_j | G, ℓ) with Jeffreys credible intervals"),
    (1, "Bayesian analog of the NMI Δ = +0.048 — operates on pairs, not partitions"),
    (1, "Within-layer > across-layer → layer-disassortative structure"),
    (0, "Output 3: Per-node posterior entropy H_i over block assignment"),
    (1, "Identifies gate-boundary cases vs confidently-assigned nodes"),
    (1, "Inputs for H-DP refinement: the high-entropy nodes are the gate-boundary sub-population"),
    (0, "All three extracted from housing Gauss+LIME run (pct_black_latino layer, K=4, N=3000, burn=1000)"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 32 — Ψ matrix
slide, n = new_slide("Output 1 — Block-Probability Matrix Ψ (Layer-Disassortative Structure)", "Act V — §2")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Posterior mean Ψ_{h,k} (rows = layers, cols = blocks 0–3) from 2000 post-burn-in samples",
            font_size=Pt(11), italic=True, color=MID_GRAY)

headers = ["Layer", "Block 0", "Block 1", "Block 2", "Block 3"]
rows = [
    ["not_protected (pct_bl_lat ≤ 0.25)", "0.093", "0.124", "0.290", "0.493"],
    ["protected (pct_bl_lat > 0.25)",     "0.067", "0.209", "0.524", "0.201"],
]
pptx_table(slide, LEFT_COL, Inches(1.4), FULL_W, headers, rows,
           col_widths=[Inches(4.0), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.6)],
           font_size=Pt(11))

left2_items = [
    (0, "Protected layer concentrates in Block 2 (ψ=0.524)"),
    (0, "Not-protected concentrates in Block 3 (ψ=0.493)"),
    (0, "Layer-disassortative: dominant blocks DIFFER between layers"),
    (1, "Weaker than sqft_lot run (which had near-zero cross-layer mass) — consistent with pct_black_latino"),
    (1, "being a more diffuse signal than lot size in the fingerprint graph"),
    (0, "Credible intervals from 2000 post-burn-in samples (all in output files)"),
]
bullets(slide, LEFT_COL, Inches(3.0), LEFT_W, Inches(3.5), left2_items)
embed_figure(slide, RIGHT_L, Inches(2.8), RIGHT_W, Inches(4.5),
             SBM_FIGS / "fig_psi_heatmap.png",
             "[fig_psi_heatmap.png]")

# Slide 33 — Co-clustering posterior
slide, n = new_slide("Output 2 — Pairwise Co-Clustering Posterior (20-Node Subset)", "Act V — §2")
left_items = [
    (0, "20-node random subset (seed=42): 18 not-protected, 2 protected"),
    (0, "Computed from 2000 post-burn-in samples (label-switching safe — computed before alignment)"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, Inches(1.3), left_items)

headers = ["Pair type", "Mean P(z_i = z_j)", "Notes"]
rows = [
    ["Within-layer",   "0.312", "Same pct_black_latino class"],
    ["Across-layer",   "0.237", "Opposite classes"],
    ["Ratio within/across", "1.319", "~32% excess within-layer co-clustering"],
]
pptx_table(slide, LEFT_COL, Inches(2.3), LEFT_W, headers, rows,
           col_widths=[Inches(2.2), Inches(2.0), Inches(2.5)])

left2_items = [
    (0, "Direction consistent with Δ NMI = +0.048: pEx recovers layer-correlated partition"),
    (0, "Ratio 1.319 < sqft_lot run (1.50): pct_black_latino is more diffuse signal"),
    (1, "Expected: 20-node subset has only 2 protected nodes → within-protected has 1 pair vs 36 not-protected"),
    (0, "Sanity check passes: ratio > 1 with within > across"),
]
bullets(slide, LEFT_COL, Inches(4.05), LEFT_W, Inches(2.7), left2_items)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             SBM_FIGS / "fig_coclust_matrix.png",
             "[fig_coclust_matrix.png]")

# Slide 34 — Per-node uncertainty
slide, n = new_slide("Output 3 — Per-Node Posterior Entropy: 9/20 Confident, 4/20 Gate-Boundary", "Act V — §2")
left_items = [
    (0, "Entropy range: [~0.000, 0.672] nats  (max possible = ln(4) = 1.386 nats)"),
    (0, "Mean entropy: 0.117 nats"),
    (0, "9 / 20 nodes: effectively zero entropy (H < 0.001) — confidently assigned to single block"),
    (0, "4 / 20 nodes: H > 0.3 nats — genuine posterior uncertainty"),
    (1, "Highest-entropy node: H ≈ 0.672 (Block 0), oscillates between Block 0 and Block 2"),
    (0, "All 4 high-entropy nodes sit on the Block 2 periphery — H-DP gate-boundary candidates"),
    (0, "Posterior is well-mixed (not smeared by label-switching) — sampler quality confirmed"),
    (0, "Comparable to sqft_lot run ([0.000, 0.681], 11/20 confident)"),
]
bullets(slide, LEFT_COL, BODY_TOP, LEFT_W, BODY_H, left_items)
embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             SBM_FIGS / "fig_node_uncertainty.png",
             "[fig_node_uncertainty.png]")

# Slide 35 — Block geography + fingerprint shapes
slide, n = new_slide("Block Characterisation — Fingerprint Shapes + Geography", "Act V — Blocks")
embed_figure(slide, LEFT_COL, BODY_TOP, Inches(6.3), Inches(5.6),
             SBM_FIGS / "fig_block_fingerprint_shapes.png",
             "[fig_block_fingerprint_shapes.png]")
embed_figure(slide, Inches(6.7), BODY_TOP, Inches(6.4), Inches(5.6),
             SBM_FIGS / "fig_block_geography.png",
             "[fig_block_geography.png]")

# Slide 36 — Block 1 fingerprint flip
slide, n = new_slide("Novel Finding: Block 1 Fingerprint Flip — A Laundering Mechanism", "Act V — Block 1")
headers = ["Block", "n", "% prot.", "Anchor", "Tail mean", "Δ anchor→tail", "Semantic label"]
rows = [
    ["0", "20",  "10%", "5.677", "5.599", "−0.08 (anchor > tail)", "Young central, gate-boundary"],
    ["1", "27",  "19%", "5.303", "5.643", "+0.34 (tail > anchor) ★","Suburban large-lot, FLIP"],
    ["2", "64",  "19%", "5.527", "5.617", "+0.09 (moderate)",      "Protected-dominant western"],
    ["3", "89",   "6%", "5.750", "5.608", "−0.14 (anchor > tail)", "Northern older, safe zone"],
]
pptx_table(slide, LEFT_COL, BODY_TOP, FULL_W, headers, rows,
           col_widths=[Inches(0.8), Inches(0.7), Inches(0.9), Inches(1.0), Inches(1.05), Inches(2.05), Inches(3.35)],
           font_size=Pt(10))
add_textbox(slide, LEFT_COL, Inches(2.75), FULL_W, Inches(0.3),
            "★ Block 1 is the only block where tail mean > anchor by +0.34 log_price",
            font_size=Pt(11), italic=True, color=ACCENT_RED)

left_items = [
    (0, "Block 1 mechanism: real inputs trigger biased-model route (low prediction = anchor);"),
    (1, "Gaussian perturbations escape to innocuous-model route (higher prediction = tail)"),
    (1, "EXPLAINER sees a 'healthy' perturbation cloud above the real input → attributes to structural features"),
    (1, "Scaffold is actively hiding these nodes' penalised status through the OOD detector asymmetry"),
    (0, "This laundering mechanism was NOT visible in Exp 3 aggregate attribution analysis"),
    (1, "pEx block structure surfaced it as a distinct behavioural regime (n=27, 5 protected)"),
    (0, "Block 2 is the policy-relevant block: 12/24 protected nodes, western-corridor geography"),
    (0, "Block 3: 'safe zone' — northern, old homes; 6% protected; anchor > tail (no penalty fired often)"),
]
bullets(slide, LEFT_COL, Inches(3.1), FULL_W, Inches(3.8), left_items)

# Slide 37 — Exp 12 spectrum sweep
slide, n = new_slide("Exp 12 — pEx-SBM Across Moran's I Spectrum (§6.2 Validation)", "Act V — Exp 12")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "Correction: pct_black_latino row was affected by a data-lookup bug (fixed). Corrected: plain=0.287, pEx=0.334, Δ=+0.048.",
            font_size=Pt(10), italic=True, color=ACCENT_RED)

headers = ["Feature", "I", "Plain NMI", "pEx NMI", "Δ NMI", "C-ratio", "Layer-disass.?"]
rows = [
    ["condition",        "0.090", "0.065", "0.064", "−0.000", "0.869", "No"],
    ["view",             "0.242", "0.877", "0.877", "0.000",  "sat.",  "Yes (sat.)"],
    ["sqft_living",      "0.291", "0.014", "0.007", "−0.007", "0.984", "No"],
    ["grade",            "0.365", "0.014", "0.020", "+0.006", "1.001", "Yes"],
    ["sqft_lot",         "0.469", "0.027", "0.041", "+0.014", "1.204", "Yes"],
    ["pct_latino",       "0.756", "0.011", "0.012", "+0.000", "0.940", "No"],
    ["pct_black_latino", "0.834", "0.287", "0.334", "+0.048", "1.319", "Yes"],
    ["pct_black",        "0.850", "0.013", "0.014", "+0.002", "0.891", "No"],
]
cws = [Inches(2.0), Inches(0.7), Inches(1.1), Inches(1.0), Inches(1.0), Inches(1.0), Inches(2.0)]
pptx_table(slide, LEFT_COL, Inches(1.5), LEFT_W, headers, rows, col_widths=cws, font_size=Pt(10))
add_textbox(slide, LEFT_COL, Inches(5.55), LEFT_W, Inches(0.35),
            "5/8 features Δ > 0. Middle regime (C-ratio 1–3): grade, sqft_lot, pct_black_latino.",
            font_size=Pt(11), italic=True, color=NAVY)

embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             SBM_SWP / "fig_delta_nmi_vs_morans_i.png",
             "[fig_delta_nmi_vs_morans_i.png]")

# Slide 38 — §6.2 reframing
slide, n = new_slide("§6.2 Reframing: Middle Regime (C-ratio 1–3) as the Actionable Condition", "Act V — §6.2")
bullet_items = [
    (0, "Strong §6.2 claim ('recovers across full I range') is NOT fully validated"),
    (0, "Three regimes in the master table:"),
    (1, "Saturation (view, I=0.242): plain NMI=0.877 — scaffold perfectly separates feature by structural means alone;"),
    (1, "  pEx has no room to add. Not a failure — demonstrates scaffold works."),
    (1, "Middle regime (grade I=0.365, sqft_lot I=0.469, pct_bl_lat I=0.834): C-ratio 1.0–1.3, pEx fires"),
    (1, "Collapse/diffuse (condition, sqft_living, pct_latino, pct_black): C-ratio < 1 or near 1;"),
    (1, "  scaffold penalty too weak OR demographic feature absent from baseline → null signal"),
    (0, "Revised §6.2 language: 'pEx-SBM recovers block structure in the middle behavioral regime"),
    (1, "(C-ratio ∈ [1, 3]), which spans Moran's I ≈ 0.35–0.83 in King County housing data'"),
    (0, "C-ratio from plain SBM is an auditor-actionable pre-diagnostic:"),
    (1, "If C-ratio ∈ [1, 3]: run pEx — likely to detect audit-relevant structure"),
    (1, "If C-ratio < 1: scaffold penalty is too diffuse; if C-ratio >> 3: plain SBM already suffices"),
]
bullets(slide, LEFT_COL, BODY_TOP + Inches(0.05), LEFT_W, BODY_H, bullet_items)
embed_figure(slide, RIGHT_L, BODY_TOP, RIGHT_W, Inches(5.6),
             SBM_SWP / "fig_regime_diagnostic.png",
             "[fig_regime_diagnostic.png]")


# ═══════════════════════════════════════════════════════════
# ACT VI — Gaps, Asks, Next Steps
# ═══════════════════════════════════════════════════════════

# Slide 39 — Three theoretical gaps
slide, n = new_slide("Three Theoretical Gaps from Literature Foundations", "Act VI — Gaps")
bullet_items = [
    (0, "Gap 1: pEPPF → explanation-functional CIs (Claim a)"),
    (1, "Camerlenghi, Lijoi, Orbanz, Prünster (AoS 2019) gives partition distribution in closed form"),
    (1, "But translating to credible sets over EXPLANATION FUNCTIONALS (e.g. Shapley-within-block)"),
    (1, "requires an additional lemma — not yet derived. Concrete work item."),
    (0, "Gap 2: Projectivity must be stated for hDP/hPY specifically (Claim b)"),
    (1, "Orbanz (2017) Thm 3 gives the stability template for projective models"),
    (1, "Lauritzen, Rinaldo, Sadeghi (JRSSB 2018): projectivity is NOT a generic consequence of exchangeability"),
    (1, "Claim (b) must be proved for the specific hierarchical-process prior, not assumed"),
    (1, "Our finite-K variant is used empirically; projectivity for it is not automatic"),
    (0, "Gap 3: DMMS SNR threshold is a hard ceiling (Claim c)"),
    (1, "Deshpande, Montanari, Mossel, Sen (NeurIPS 2018) + Lu-Sen (2020): tight threshold λ²+μ²/γ > 1"),
    (1, "pEx-SBM does NOT beat this bound; it pools covariate information within each stratum"),
    (1, "Recovery theorem must explicitly scope to the SNR regime — cannot claim universal recovery"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 40 — Empirical gaps
slide, n = new_slide("Empirical Gaps and Open Questions", "Act VI — Gaps")
bullet_items = [
    (0, "H-DP refinement has K̂ ≈ 10 vs binary layer (arity mismatch)"),
    (1, "H-DP consistently infers K̂≈10 across θ₀ ∈ [0.25, 2.0] — not a prior artifact"),
    (1, "Mechanism: K=10 refines the binary layer + identifies ~20% gate-boundary sub-population"),
    (1, "But NMI is mechanically penalised by the arity mismatch vs binary truth"),
    (1, "Open: is there a binary-layer-compatible H-DP evaluation metric?"),
    (0, "Penalty-magnitude robustness sweep still pending"),
    (1, "All Exp 9 results at penalty = −0.30; need sweep at 0.10 / 0.30 / 0.50 to confirm Δ robustness"),
    (0, "No predictive test on H-DP gate-boundary clusters"),
    (1, "Do high-entropy / gate-boundary nodes (H > 0.3) have higher adversarial sensitivity on held-out?"),
    (1, "If yes: a practical test for 'which houses are most vulnerable to explanation hiding'"),
    (0, "Lipschitz (β) sweep downgraded to robustness result (not mechanism confirmation)"),
    (1, "3-seed CIs overlap; n=20 re-run at β∈{2,5,10} recommended if NeurIPS track chosen"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 41 — Three asks
slide, n = new_slide("Three Asks for Today's Discussion", "Act VI — Asks")
add_textbox(slide, LEFT_COL, BODY_TOP, FULL_W, Inches(0.3),
            "These are the three open decisions gating the write-up direction",
            font_size=Pt(12), italic=True, color=MID_GRAY)

asks = [
    ("Ask (a): Lead story — finite-K Δ or H-DP refinement?",
     "Finite-K: clean NMI headline (Δ=+0.048, p=1.8e-9), four-dataset replication, K-sweep mechanism.\n"
     "H-DP: richer story — three LIME attribution regimes, gate-boundary sub-population, K̂≈10.\n"
     "Tradeoff: NMI + K-sweep vs refinement narrative. Can structure as finite-K main + H-DP appendix."),
    ("Ask (b): Workshop vs NeurIPS-main?",
     "Workshop (NeurIPS XAI, July deadline): four-dataset + K-sweep + empirical regularity is sufficient.\n"
     "NeurIPS main: needs (a) formal plain-SBM recovery-probability bound as fn of (K, m-variability, layer-alignment),\n"
     "OR (b) H-DP refinement/LIME-regime story as second contribution. Either requires substantial additional work."),
    ("Ask (c): SNR-regime analysis or projectivity first?",
     "SNR regime: scopes the recovery theorem — 'pEx recovers in middle regime (C-ratio ∈ [1,3])' is \n"
     "empirically grounded and tightens the Claim (c) statement. More tractable.\n"
     "Projectivity: stronger theoretical contribution but requires proving the specific hDP/hPY case \n"
     "(Orbanz 2017 Thm 3 as template). More impactful if successful."),
]
for i, (q, detail) in enumerate(asks):
    top = BODY_TOP + Inches(0.5) + i * Inches(1.8)
    add_rect(slide, LEFT_COL, top, FULL_W, Inches(1.65), fill_rgb=TABLE_EVEN)
    add_textbox(slide, LEFT_COL + Inches(0.1), top + Inches(0.05), FULL_W - Inches(0.2), Inches(0.35),
                q, font_size=Pt(13), bold=True, color=NAVY)
    add_textbox(slide, LEFT_COL + Inches(0.15), top + Inches(0.38), FULL_W - Inches(0.3), Inches(1.2),
                detail, font_size=Pt(11), color=MID_GRAY)

# Slide 42 — Summary of what's established
slide, n = new_slide("Summary: What Is Established", "Act VI — Summary")
bullet_items = [
    (0, "I. Scaffold attack works regardless of Moran's I (Exps 3, 4b, 8, 10–11)"),
    (1, "OOD routing is I-invariant; coordinate laundering is not the mechanism"),
    (1, "Attack feasibility predicted by feature role (structural vs demographic) + OOD calibration"),
    (0, "II. Rung-3 + finite-K pEx is the first scaffold-attack detector with non-null layer signal (Exp 9)"),
    (1, "Δ NMI = +0.048, 95% CI [+0.017, +0.076], p = 1.8e-9 (20-seed)"),
    (1, "Replicates on four datasets across regression/classification, custom + Slack-original scaffolds"),
    (1, "Single-model sanity checks (honest / biased / innocuous) all null — routing mixture is the signal"),
    (0, "III. Per-block structure reveals novel attack mechanisms not visible in aggregate (Act V)"),
    (1, "Block 1 fingerprint flip (+0.34 log-price): real inputs penalised, perts escape to innocuous"),
    (1, "Block 2 western-corridor protected cluster: where scaffold attack is most consequential"),
    (0, "IV. All three §2 primitives computed, matched figures in place (Ψ, co-clustering, entropy)"),
    (1, "Consistent with Exp 9 direction; attenuated relative to sqft_lot run as expected"),
    (0, "V. §6.2 partially validated — middle regime (C-ratio 1–3) as auditor-actionable pre-diagnostic"),
]
bullets(slide, LEFT_COL, BODY_TOP, FULL_W, BODY_H, bullet_items)

# Slide 43 — Acknowledgments / Q&A
slide, n = new_slide("Acknowledgments / Q&A", "Act VI")
add_textbox(slide, Inches(0.5), Inches(1.8), Inches(12.3), Inches(0.5),
            "Questions, corrections, and course-corrections welcome",
            font_size=Pt(20), color=NAVY, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.5), Inches(2.6), Inches(12.3), Inches(0.35),
            "Robert Romani — rpromani251@gmail.com",
            font_size=Pt(15), color=MID_GRAY, align=PP_ALIGN.CENTER)

thanks = [
    "Peng Luo — advising the geospatial XAI direction and the four open questions",
    "Durante, Gaffi, Lijoi, Prünster (JASA 2025) — pEx-SBM theoretical foundation",
    "Slack, Hilgard, Singh, Lakkaraju (2020) — fooling LIME and SHAP, original scaffold code",
    "Vascotti et al. (2025) — on-manifold robustness framework",
]
for i, t in enumerate(thanks):
    add_textbox(slide, Inches(1.5), Inches(3.3 + i * 0.55), Inches(10.3), Inches(0.45),
                "  • " + t, font_size=Pt(13), color=NAVY)

add_textbox(slide, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.4),
            "Repo: geospatial-xai-attacks/  •  Entry point: docs/sbm_explainer/README.md",
            font_size=Pt(12), color=MID_GRAY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════
# Save
# ══════════════════════════════════
prs.save(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
print(f"Slide count: {SLIDE_COUNT}")
print(f"File size: {OUT_PATH.stat().st_size / 1024:.1f} KB")
