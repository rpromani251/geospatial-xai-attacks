# Agent plan: matched-scaffold experiments and the draft for Peng

Written 12 November 2026 (revised the same evening) for a Claude Code agent (Opus) working in this repository on Robert's Mac.
Goal: by the end of the week, a seeded, editable draft of the geospatial fooling paper that Robert can send to Peng Luo.

**The paper's central claim** (Robert, 12 November): how much bias a scaffold can hide from an explainer is set by the
explainer's *estimator* — where it queries the model — not by whether the explainer knows about geography. LIME can be
fooled for free; KernelSHAP and GeoShapley can be fooled only by giving up fidelity, because Shapley values satisfy
efficiency and evaluate the explained home itself. Every section of the draft serves this claim. The draft covers
King County with three model families and an attacker who does not know the audit settings; a second metro (Cook
County) is added later (Phase 6, only when Robert starts it).

Read this whole file before doing anything. Work through the phases in order. Stop and report at every gate marked **STOP IF**.

---

## 0. What this is about (read first)

The paper asks whether the Slack et al. (2020) scaffold attack, which hides a biased model from LIME and SHAP, also hides it from **GeoShapley**, on King County housing data. The biased model subtracts 0.30 from log price for homes in census tracts where `pct_black_latino > 0.25`. A scaffold serves the biased model to inputs a detector calls real and an innocuous model (8 structural features, no location, no demographics) to inputs it calls synthetic.

Read, in this order:

1. `docs/fooling_geoxai_summary.tex` — Robert's October summary (real data, LIME matched detector, 20 seeds).
2. `docs/paper/draft_v0_manuscript.md` — the earlier manuscript draft (structure, introduction, background, data and methods prose). Its results tables came from a synthetic stand-in and **all of them must be replaced**. If this file is missing, ask Robert to export the doc "Fooling Geospatial XAI: Scaffold Attacks on SHAP, LIME, and GeoShapley" as Markdown to that path, and continue with the other phases meanwhile.
3. `docs/journal_paper_plan.md` — earlier plan. Keep its statistical design (seed as unit, effect sizes with intervals). Its claim C2 ("LIME resists; ordering inverts") and contribution 3 ("GeoShapley is the most compromised") are **withdrawn**.
4. `scripts/run_matched_attack.py` (docstring and `--help`), `scripts/summarize_matched_attack.py`, `scripts/run_residual_diagnostic.py`, `scripts/mechanism_routing.py` (docstring: the mechanism argument), `scripts/run_matched_queue.sh`. All writes in these scripts are atomic and every job is resumable: if anything is interrupted, just rerun the same command or queue.
5. `results/q1_attack_settings*/summary.csv` and `scripts/q1_attack_settings.py` — Robert's 16 October single-seed results; reconcile with them in the report appendix.

### What each deployment is

| Deployment (cell name) | Meaning |
| --- | --- |
| `honest` | honest model h |
| `biased` | biased model f served directly; **the reference the attack must hide** |
| `gen` | scaffold with the generic (Slack-recipe) detector, threshold 0.5 |
| `mat` | scaffold with a detector trained on the explainer's own queries; threshold set so ≥ 90% of held-out real homes still get f |
| `mat75`, `mat50` | same detector, thresholds giving 75% and 50% fidelity (the frontier) |

Designs: `feature` (11 features, trigger is the `pct_black_latino` column), `location` (10 real features; the penalty is decided inside the model from the home's coordinates via the real tract polygons), `nocoord` (no coordinates; LIME and KernelSHAP only).

Robustness options (each gets its own tag in the file name, so results never mix):

| Option | Tag | Meaning |
| --- | --- | --- |
| `--model rf` / `--model mlp` | `_m-rf`, `_m-mlp` | model family of h and ψ (default XGBoost). Test R² about 0.87 (rf) and 0.81 (mlp) against 0.89 |
| `--audit-bg N`, `--audit-bg-kind kmeans` | `_aud-bg25`, `_aud-bg100kmeans`, … | the auditor uses a Shapley background the attacker did not train for |
| `--audit-lime-samples N`, `--audit-lime-sampler gaussian` | `_aud-lime5000`, `_aud-lime500gauss` | the auditor changes LIME's settings |
| `--attacker rf_leaf5` / `xgb`, `--harvest-scale 4` | `_att-…`, `_hs4` | stronger attacker (optional) |

Mismatch runs compute only the `biased` and `mat` cells of the affected explainers and reuse the matched run's
detectors (they depend only on what the attacker assumed). The harvested Shapley detectors already use background draws
different from the auditor's, so "a fresh background sample" is the default, not a mismatch.

### What the first three seeds already show (preliminary, do not quote; for sanity checks)

Feature design, seeds 0–2, 200 homes:

- LIME: matched detector separates LIME's queries perfectly (AUC 1.000); trigger in top three 0.1%; penalty share kept 2%; fidelity 96%. Generic detector: 96% of LIME's queries reach f, nothing hidden.
- KernelSHAP and GeoShapley at 90% fidelity: penalty share kept 0.69 and 0.74; detector AUC 0.77 and 0.72. At the generic detector (fidelity ~74%): kept ~0.50.
- So: LIME is fully fooled; the Shapley explainers are only partly fooled, and hiding more costs the attacker fidelity. GeoShapley's queries look more real than KernelSHAP's (lower AUC); whether that makes it measurably harder to fool is the C2 question below.

If your final numbers differ from these by much more than the intervals, find out why before writing. Two known
differences from the runs that produced them: GeoShapley's coalition sampling is now seeded (its numbers move by
Monte Carlo noise only), and KernelSHAP now uses `l1_reg=False` (checked at seed 42: no change at three decimals).

---

## 1. Ground rules

- **Do not modify or delete anything that already exists** except the files this plan names. New outputs go only under `results/matched_attack/`, `results/matched_attack_check/`, `results/residual_diagnostic/`, `results/mechanism/`, `docs/paper/`. The one permitted move of existing files is the sandbox-run set in Phase 0, step 4.
- **Disk.** About 150 GB free on 12 November evening. Every job runs with `--clean-cache` (the queue script adds it), which deletes each seed's cached explanation chunks once that seed finishes but keeps its detectors (about 130 MB per seed; about 10 GB over all queues; `mechanism_routing.py` and the mismatch runs reuse them). The queue skips jobs if less than 1.5 GB is free. **STOP IF** free space is under 5 GB before starting; tell Robert.
- **Never type a number into the paper by hand.** Every number in a table comes from a CSV through `scripts/make_paper_tables.py` (Phase 4); every number in prose must be copied from those generated tables or CSVs, and the review in Phase 3 will check this.
- Label claims as observed (measured here), from earlier work (cite the file), or conjectured.
- Do not commit raw data or caches. `results/` and `data/` are git-ignored: copy the summary CSVs you cite into `docs/paper/data/` and commit those. Stage files by explicit path only (never `git add -A` or `git add .`); Robert has other untracked work in this repo that must not be committed.
- Each Bash call may start a fresh shell, so never rely on `source …/activate`: call `.venv/bin/python` explicitly, and run the queue as `PY=.venv/bin/python bash scripts/run_matched_queue.sh …`.
- Keep the Mac awake for long runs: prefix with `caffeinate -i`.
- Run long jobs in the background (Bash `run_in_background`, or `nohup … &`) and poll the queue log; do not hold a foreground call for an hour.

---

## 2. Phase 0: environment and acceptance check (about 20 minutes)

1. Create a venv with the exact versions the first seeds used (Linux aarch64, Python 3.10):

   ```
   uv venv .venv --python 3.10
   uv pip install --python .venv/bin/python numpy==2.2.6 scipy==1.15.3 pandas==2.3.3 scikit-learn==1.7.2 xgboost==3.2.0 \
       shap==0.49.1 lime==0.2.0.1 geoshapley==0.2.1.1 shapely==2.1.2 pyproj==3.7.1 \
       matplotlib==3.10.9 pyyaml geopandas
   ```

   If Python 3.10 is not available, use 3.11 and note it.
2. Confirm the inputs exist: `data/processed/seattle_housing_with_demographics.csv`, `data/processed/king_county_tracts_demographics.geojson`, `results/matched_attack/feature_seed{0,1,2,42}.json`.
3. Acceptance run (seed 42, a subset of cells, written to a separate folder):

   ```
   .venv/bin/python scripts/run_matched_attack.py --design feature --seed 42 --out results/matched_attack_check \
       --cells lime:biased,lime:gen,lime:mat,kernelshap:biased --jobs 4
   ```

   Compare `results/matched_attack_check/feature_seed42.json` with `results/matched_attack/feature_seed42.json`:

   | Quantity | Expected |
   | --- | --- |
   | `honest_r2` | 0.8895 (±0.0005) |
   | `psi_r2` | 0.6414 (±0.002) |
   | `lime/gen` → `queries_routed_to_f` | 0.962 (±0.01) |
   | `lime/mat` → `queries_routed_to_f` | below 0.01 |
   | `kernelshap/biased` → `trig_signed_pen` | −0.261 (±0.01) |

   Then a smoke test of the robustness options (about 2 minutes; checks they run on this machine, not their numbers):

   ```
   .venv/bin/python scripts/run_matched_attack.py --design feature --seed 42 --n-explain 10 --chunk 5 \
       --model mlp --out results/matched_attack_check --cells lime:mat,kernelshap:mat --jobs 4
   .venv/bin/python scripts/run_matched_attack.py --design feature --seed 42 --n-explain 10 --chunk 5 \
       --audit-bg-kind kmeans --out results/matched_attack_check --cells kernelshap:biased,kernelshap:mat --jobs 4
   ```

   Both should end with `PARTIAL 2/2 cells` (an explicit `--cells` list never marks a run complete) and no traceback.

   Afterwards delete `results/matched_attack_check/_cache/`.

   - All match to these tolerances: continue.
   - R² matches but attribution numbers drift slightly (platform floating point): continue, and record the drift in the report's reproducibility appendix.
   - **STOP IF** `honest_r2` or the LIME routing numbers are off: report to Robert; do not run Phase 1.
4. Move the sandbox-run seeds aside so that every seed in the paper comes from this Mac. `feature_seed{0,1,2,3}` were computed in a Linux sandbox whose detector and chunk caches are not in this repository. Left in place, the queue would add frontier cells and seed 3's missing cells using detectors rebuilt here, mixed with cells computed there. So:

   ```
   mkdir -p results/matched_attack/_vm_runs
   mv results/matched_attack/feature_seed[0-3].json results/matched_attack/feature_seed[0-3]__*.npz \
      results/matched_attack/per_seed_feature.csv results/matched_attack/summary_feature.csv results/matched_attack/_vm_runs/
   ```

   Keep `feature_seed42.json` where it is (the acceptance reference; excluded from every table). The main queue then
   runs seeds 0–9 from scratch. Afterwards, compare the Mac's seeds 0–3 with `_vm_runs/` and report the largest
   difference in penalty share kept in the reproducibility appendix.

---

## 3. Phase 1: runs

Check cores with `sysctl -n hw.ncpu`; use `WORKERS = ncpu / 2`, 2 threads each. Queue files are in `docs/agent_plan/`.

1. **Main queue** (needed for the draft):

   ```
   PY=.venv/bin/python caffeinate -i bash scripts/run_matched_queue.sh docs/agent_plan/queue_main.txt
   ```

   20 jobs: feature seeds 0–9 and location seeds 0–9, all run on this Mac. Expect roughly 15–30 minutes per seed per worker; on 8 cores (4 workers) about two to three hours. Poll `results/matched_attack/_logs/queue.log`. A job printing `PARTIAL` means it stopped early: rerun the queue (finished jobs exit at once).
2. **Residual diagnostic** (E7; about 5 minutes):

   ```
   .venv/bin/python scripts/run_residual_diagnostic.py --seeds 10
   ```
3. **Mechanism** (after the main queue; a few minutes per seed):

   ```
   .venv/bin/python scripts/mechanism_routing.py --design feature --explainer kernelshap --seeds 0-4
   .venv/bin/python scripts/mechanism_routing.py --design feature --explainer geoshapley --seeds 0-4
   .venv/bin/python scripts/mechanism_routing.py --design location --explainer geoshapley --seeds 0-4
   ```

   Writes `results/mechanism/routing_by_size_*.csv` (routing rate and attribution contribution by coalition size, for
   each attacker threshold) and `predicted_vs_measured_*.csv` (Monte Carlo prediction of the penalty share kept versus
   the measured one from the attack JSON, at 90%, 75% and 50% fidelity).
4. **Robust queue** (needed for the draft; start it as soon as the main queue is done):

   ```
   PY=.venv/bin/python caffeinate -i bash scripts/run_matched_queue.sh docs/agent_plan/queue_robust.txt
   ```

   45 jobs: R1 model families (random forest, neural net; feature and location, seeds 0–4) and R2 audit-settings
   mismatch (background 25, 200, k-means; LIME 5,000 samples; LIME Gaussian sampler; feature, seeds 0–4). Mismatch jobs
   are short (two cells per explainer; background 200 doubles the Shapley cost). Neural-net jobs cost about as much as a
   main-queue seed; random-forest jobs about three times as much (the forest predicts about 3.5 times slower than
   XGBoost). Expect four to six hours on four workers.
5. **Extra queue** (run while you write, if time allows; each block is useful on its own):

   ```
   PY=.venv/bin/python caffeinate -i bash scripts/run_matched_queue.sh docs/agent_plan/queue_extra.txt
   ```

   Autocorrelation sweep (`--shuffle 0.5`, `0.0`), smaller penalty (`--penalty 0.10`), no coordinates (`nocoord`), seeds 0–4 each.
6. **Attacker queue** (optional, last): `docs/agent_plan/queue_attacker.txt` — stronger detector classifiers and four times the harvested queries.

Order of priority if time runs short: main → mechanism → E7 → robust → extra → attacker. The draft needs the first four.

**STOP IF** any job ends in a Python traceback (check its log): fix only obvious environment problems; for anything in the experiment logic, stop and report.

---

## 4. Phase 2: summaries and figures

1. Summaries:

   ```
   .venv/bin/python scripts/summarize_matched_attack.py --design feature
   .venv/bin/python scripts/summarize_matched_attack.py --design location
   .venv/bin/python scripts/summarize_matched_attack.py --design feature --shuffle 0.5     # and 0.0, if run
   .venv/bin/python scripts/summarize_matched_attack.py --design feature --penalty 0.1     # if run
   .venv/bin/python scripts/summarize_matched_attack.py --design nocoord                   # if run
   # robustness: one call per configuration
   .venv/bin/python scripts/summarize_matched_attack.py --design feature --model rf          # also mlp; also --design location
   .venv/bin/python scripts/summarize_matched_attack.py --design feature --audit bg25        # bg200, bg100kmeans, lime5000, lime500gauss
   .venv/bin/python scripts/summarize_matched_attack.py --design feature --attacker xgb      # if run
   ```

   The summaries also report `gap_kept_trigger`, `gap_relocated` and `gap_total`: the penalty bookkeeping on the gap
   "penalised minus unpenalised homes" relative to the biased reference, with `top_target_mode` naming the feature the
   penalty most often moves to. These are descriptive and noisy: switching a home between f and ψ also moves
   attribution on non-trigger features (ψ ≠ h), so `gap_relocated` mixes moved penalty with model-switching noise. Use
   them to say where the penalty goes, not as a test; the test of the mechanism is rule M with `mechanism_routing.py`.
   Runs made before this field existed are read from their saved `.npz` files.

   Each writes `results/matched_attack/summary_<design>.csv` and `per_seed_<design>.csv`. Seed 42 is excluded by default (calibration seed). Interval: mean ± t⁻¹ with S−1 degrees of freedom at 0.975, times sd/√S, over S seeds (the t quantile, because with 10 seeds the normal quantile Φ⁻¹(0.975) gives intervals about 13% too narrow).
2. Figures (write `scripts/make_paper_figures_matched.py`, save PDF and PNG to `docs/paper/figures/`):
   - **F1 study area.** Tracts coloured by `pct_black_latino`, the 0.25 threshold outlined, sale locations as points.
   - **F2 main result.** For each explainer, penalty share kept (signed attribution on penalised homes ÷ the same under `biased`) for biased, gen, mat, with 95% intervals. Feature design.
   - **F3 the frontier (the key figure).** Penalty share kept against fidelity on real homes, KernelSHAP and GeoShapley, points for mat (90%), mat75, mat50 and gen, with intervals in both directions; LIME as a single point at ~96% fidelity and ~0 kept. One panel per design.
   - **F4 GeoShapley decomposition.** GEO effect on penalised homes and summed absolute interaction effect, by deployment, both designs.
   - **F5 outcome diagnostic.** AUC of the local residual statistic against penalty (0.02–0.30), biased versus scaffold, honest model as the reference line.
   - **F6 autocorrelation** (if run). Query leak and penalty share kept against trigger Moran's I.
   - **F7 mechanism.** Left: routing rate to f by coalition size (k = 1 … M, the last point is the explained home itself), for the three attacker thresholds, KernelSHAP and GeoShapley. Right: predicted against measured penalty share kept, one point per seed and threshold, with the identity line.
   - **F8 robustness.** A forest plot of penalty share kept under the matched scaffold (with interval), one row per configuration: main, rf, mlp, each audit mismatch, each attacker variant; one colour per explainer; a vertical line at the C1' threshold 0.10.
   Use colour-blind-safe colours, the same colour per explainer in every figure, labelled axes with units.

---

## 5. Decision rules (fixed now; apply them, do not reinterpret after seeing results)

- **C1 (attack works on each explainer).** Paired difference, biased minus mat, in penalty share kept, across seeds; Holm-correct over the three explainers. Claim "fooled" only where the interval excludes 0.
- **C1' (completeness).** Call an explainer "fully fooled" only if, under mat, trigger-in-top-three ≤ 5% and penalty share kept ≤ 0.10 with fidelity ≥ 0.90. Otherwise "partly fooled" and report the frontier.
- **C2 (does GeoShapley's design protect it?).** Paired difference GeoShapley minus KernelSHAP, under mat, in penalty share kept (`summarize` prints it). Interval above 0: "slightly harder to fool". Interval containing 0: "no detectable difference at this sample size". Report the detector AUC difference alongside as the mechanism.
- **C3 (decomposition).** Report GEO on penalised homes and interaction sums for biased versus mat with intervals; claim a change only if the paired interval excludes 0.
- **C5 (autocorrelation).** Slope of mat leak on Moran's I across the sweep with its interval. No claim if fewer than three levels ran.
- **M (mechanism).** Claim that routing rates predict the share kept only if, across seeds and the three thresholds, the mean absolute difference between predicted and measured share kept is no larger than the seed-to-seed standard deviation of the measured share. Otherwise state only the efficiency identity (the gap cannot vanish while the explained home is served by f) and report the miss.
- **R (robustness).** The finding "holds" in a configuration if each explainer's C1' classification (fully versus partly fooled) is the same as in the main run. Report every configuration, including any that breaks the pattern; do not drop one.
- **E7 go/no-go.** Include the outcome-diagnostic section only if, at penalty 0.10, the scaffold's `auc_lag` is ≥ 0.80 and the honest model's LISA flag rate on unpenalised homes is ≤ 0.06. Otherwise one Discussion paragraph and an appendix table. Preliminary (3 seeds): biased 0.85, scaffold 0.78 at 0.10, so this is borderline; report whatever the 10 seeds give.

---

## 6. Phase 3: independent review (before writing prose)

Start a fresh subagent with no context except: this plan, `scripts/run_matched_attack.py`, `scripts/summarize_matched_attack.py`, `scripts/mechanism_routing.py`, the summary CSVs. Ask it to check, and report findings as a list:

1. The script does what Section 0 says: each deployment, thresholds (`tau05`, `taustar`, `tau_fidNN`), fidelity measured on held-out real homes not used to set τ, matched detectors trained on training homes only, explained homes never used to train detectors.
2. Penalty share kept is computed as described and the biased reference comes from the same seed.
3. The location raster agrees with the CSV tract assignment (`raster_agrees_with_csv_tract` in each location JSON should be ≥ 0.995; one or two of 600 homes on a tract edge can disagree).
4. Intervals use seeds as the unit; seed 42 is excluded.
5. Any seed whose numbers are outliers (more than 3 sd from the others) — list them, do not drop them.
6. Robustness runs: in a mismatch run both the `biased` reference and the `mat` cell use the auditor's changed settings, while the detectors are the matched run's (the `det_tag` cache); in a model-family run both h and ψ use the stated family.
7. The mechanism script rebuilds the deployment exactly as the attack script does (same split, background rows, thresholds and penalised homes), and its "measured" column is read from the same seed's JSON.

Fix genuine bugs only after reporting them to Robert; if a bug changes results, rerun the affected jobs.

---

## 7. Phase 4: the report

Deliverables, all under `docs/paper/`:

| File | What |
| --- | --- |
| `fooling_geoxai_draft_v1.tex` and `.pdf` | the full draft, compiled |
| `tables/*.tex` | table fragments generated by `scripts/make_paper_tables.py` from the CSVs |
| `figures/` | F1–F8 |
| `note_to_peng.md` | one page: what changed since October, the headline results, the three decisions needed from him |
| `CHANGELOG.md` | every claim that changed from earlier repo documents, old value → new value, with the source file |

Write `scripts/make_paper_tables.py` first. It reads the summary and per-seed CSVs and writes one `.tex` fragment per table, numbers formatted as `mean ± half-width` with 3 decimals (percentages with 1 decimal), seeds count in the caption.

### Structure of the draft

Use `docs/fooling_geoxai_summary.tex` for LaTeX style and macros. Reuse prose from `docs/paper/draft_v0_manuscript.md` for Sections 1–4 after checking every sentence against the new results; rewrite Sections 5–8.

1. **Abstract** (≤ 250 words). Written last, from the final tables.
2. **Introduction.** Valuation models, location as a proxy for race, explainers used for bias audits, GeoShapley, the Slack attack, the three questions, a one-paragraph summary of findings.
3. **Background.** LIME, KernelSHAP and GeoShapley estimators, emphasising *where each queries the model* (that is the attack surface); the scaffold attack; related work (list the references to add; do not invent citations — mark unknown ones `[CITE]`).
4. **Data.** Housing sample, ACS tract demographics, the spatial join, the split per seed, Moran's I of the trigger, share penalised.
5. **Methods.** Models (with R² from the JSONs); the scaffold; the three detectors and the threshold calibration (τ* rule, held-out calibration homes vs explained homes); the explainers and their settings (LIME 500 samples, default quartile sampler; KernelSHAP 100-row background, every coalition enumerated and `l1_reg=False`, i.e. exact Shapley values — say that shap's default `l1_reg="num_features(10)"` would zero one of 11 features per home and that at seed 42 it changed the trigger's attribution by less than 0.001; GeoShapley 100-row background, coordinates as the joint player, exact in the location design (8 non-location features) and the package's default coalition sampling in the feature design (9 non-location features: about 600 of 1,024 coalitions, seeded per run) — see the docstring of `run_matched_attack.py`); every metric defined with a formula (penalty share kept, trigger rank, top-three share, query leak, fidelity, detector AUC from the score histograms); statistics (seeds, intervals, paired contrasts, Holm); the decision rules of Section 5 stated as pre-specified.
6. **Results.**
   - 6.1 *Matching decides LIME's fate.* Generic vs matched detector for LIME; routing, fidelity, rank.
   - 6.2 *Shapley explainers can be fooled only by giving up fidelity.* Main table plus F2 and F3.
   - 6.3 *Does GeoShapley's design help?* C2 contrast, AUC difference, interpretation.
   - 6.4 *What the attack does to GeoShapley's decomposition.* C3, F4.
   - 6.5 *Bias keyed to location alone.* Location design table, with the honest model's location attribution so the legitimate location effect is separated from the penalty (use `penalty_share_kept_excess`).
   - 6.6 *Spatial autocorrelation and penalty size* (only what ran).
   - 6.7 *Why: efficiency and routing by coalition size.* State the identity: for a Shapley explainer, attributions on a home sum to the served prediction there minus the mean on the background, so while the explained home is served by f the penalty can only move between features or be withheld from real homes. Then the decomposition φ_j = (1/M) Σ_k E[e(x_{S∪j}) − e(x_S)] with |S| = k, the routing rate by coalition size, and the prediction (F7, rule M). Explain why LIME escapes: no efficiency constraint and samples far from the instance.
   - 6.8 *Robustness* (F8): model families, audit-settings mismatch, attacker strength (if run), with rule R.
   - 6.9 *An outcome-based check* (only if E7 passes; otherwise a paragraph in the Discussion).
7. **Discussion.** The estimator, not the geography, is the attack surface; why Shapley coalitions resist detection; what is specific to the spatial case; the stealth–fidelity trade-off; implications for auditing spatial models.
8. **Limitations.** One metropolitan area in this draft (Cook County to be added: write it as `\todo{}`), synthetic penalty, attacker knows which explainer is used (though not necessarily its settings), static attacker, seeds and sample sizes as run.
9. **Conclusion.**
10. **Appendices.** A per-seed tables; B detector diagnostics (AUC, τ, fidelity on explained and calibration homes, Pearson r); C reproducibility (exact commands, package versions from the JSONs, runtimes, platform note from Phase 0); D reconciliation with earlier results (the −93% GeoShapley figure, LIME "resistance", Robert's 16 October Q1 table — say what each was measuring and how it relates to the new numbers).

### Writing rules

- Each section and paragraph opens with its point and its number.
- Every number traceable to a generated table or CSV; give the table or figure reference.
- State intervals as "0.69 (95% CI 0.63 to 0.75), 10 seeds", and say once in Methods that intervals use the t quantile over seeds.
- Do not use the preliminary numbers from Section 0 of this plan anywhere in the draft.
- Mark open items with `\todo{…}` so Robert can find them; keep a list at the end of `note_to_peng.md`.

### Final check before telling Robert it is done

1. `pdflatex` (twice) compiles with no undefined references.
2. A second fresh subagent reads the PDF and the CSVs and lists every number in the text that it cannot find in a table, figure or CSV. Fix all of them.
3. On a new branch `paper-draft-v1`, stage by explicit path only: the scripts this plan uses or creates (`run_matched_attack.py`, `summarize_matched_attack.py`, `run_residual_diagnostic.py`, `mechanism_routing.py`, `run_matched_queue.sh`, `make_paper_tables.py`, `make_paper_figures_matched.py`), `docs/agent_plan/`, and `docs/paper/`. Check `git diff --cached --stat` lists nothing else, then commit (no data, no caches, no venv).
4. Report to Robert: what ran, what did not, the C1/C2/C3/C5/M/R/E7 outcomes in one line each, and the path to the PDF.

---

## 8. Phase 6: Cook County (only when Robert starts it; not part of this week's draft)

Goal: rerun E1, E2 and R1 on a second metro with stronger segregation. Do not start without Robert's go-ahead.

1. Data. Residential sales with coordinates from the Cook County Assessor's open-data portal (datacatalog.cookcountyil.gov; parcel sales joined to parcel universe for latitude/longitude and characteristics). ACS 5-year table B03002 by tract and TIGER/Line tract polygons from the Census Bureau, the same vintage logic as King County. Check the licence and the fields before downloading; record URLs, dates and row counts in `data/raw/cook_county/README.md`.
2. Sample: one recent sale year, arm's-length single-family sales, about 3,000 homes drawn at random (fixed seed) so runtimes match King County. Features matching the King County eight as closely as the assessor data allows; write the mapping table into the README. Project coordinates to UTM zone 16N (EPSG:32616).
3. Code: add `--metro king|cook` to `run_matched_attack.py` (and the summariser, mechanism and residual scripts) through one data-loading function that returns the feature table, feature names, the tract GeoJSON path and the CRS. King County must give bit-identical results after the refactor: rerun the Phase 0 acceptance check.
4. Runs: E1 and E2 at seeds 0–9 with `--frontier`, R1 at seeds 0–4. Report the trigger's Moran's I and share penalised next to King County's.
