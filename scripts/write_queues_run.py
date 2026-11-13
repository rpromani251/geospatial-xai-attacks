"""Write docs/paper/data/queues_run.txt: every job queued for the paper, whether it completed, and the deviations from
the agent plan (docs/agent_plan/AGENT_PLAN.md). Job lists come from the runner's queue files.

  python scripts/write_queues_run.py --run-dir /path/to/run   (folder with q_*.txt and done.txt)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVIATIONS = """Deviations from the agent plan
- Main results use seeds 0-4. The plan queued seeds 5-9 as well (q_main_late); they were dropped for time after a
  container restart cost about four hours, so every attack table uses the same five seeds. To add them later run the
  q_main_late jobs, then scripts/mechanism_routing.py for the three design/explainer pairs, make_paper_tables.py,
  make_paper_figures_matched.py and pdflatex; every number in the draft updates.
- Attacker strength (q_attacker: forest with leaf size 5, XGBoost detector, 4x harvested queries) was optional in the
  plan and was not run; the draft states this as a limitation.
- Run order: robustness and extra runs ran before main seeds 5-9 (the plan put all main seeds first).
- Random forest models use 100 trees (plan: 300) for run time; model families were run in the feature design only.
- All runs on one Linux x86_64 cloud machine, Python 3.10 (numpy 2.2.6, scikit-learn 1.7.2, xgboost 3.2.0,
  shap 0.49.1, lime 0.2.0.1, geoshapley 0.2.1.1); the earlier aarch64 seed-42 reference was reproduced first.
- The outcome check (E7, run_residual_diagnostic.py --seeds 10) uses seeds 0-9.
- After the independent review: top three and fidelity are also reported on penalised homes, only complete runs enter
  tables and rules, and a robustness configuration needs at least three complete seeds (see CHANGELOG.md).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    done = {l.strip() for l in (rd / "done.txt").read_text().splitlines() if l.strip()}
    res = ROOT / "results/matched_attack"
    out = ["Jobs queued for draft v1 (scripts/run_matched_attack.py arguments), with status", ""]
    names = {"q_main.txt": "main (seeds 0-4)", "q_robust.txt": "robustness: audit settings and model families",
             "q_extra.txt": "extra: autocorrelation (C5), penalty 0.10, no-coordinate design",
             "q_main_late.txt": "main seeds 5-9 (dropped for time)",
             "q_attacker.txt": "attacker strength (optional, not run)"}
    for q, title in names.items():
        p = rd / q
        if not p.exists():
            continue
        jobs = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        n = sum(j in done for j in jobs)
        out += [f"## {q}: {title} -- {n}/{len(jobs)} completed"]
        out += [f"  [{'x' if j in done else ' '}] {j}" for j in jobs]
        out.append("")
    out.append("Outcome check: scripts/run_residual_diagnostic.py --seeds 10 (completed; results/residual_diagnostic/)")
    out.append("Mechanism: scripts/mechanism_routing.py --seeds 0-4 for feature/kernelshap, feature/geoshapley, "
               "location/geoshapley (completed; results/mechanism/)")
    out.append("")
    out.append(DEVIATIONS)
    dst = ROOT / "docs/paper/data/queues_run.txt"
    dst.write_text("\n".join(out))
    print("wrote", dst)


if __name__ == "__main__":
    main()
