"""Run queues of run_matched_attack.py jobs with a hard wall-clock deadline (for 2-hour background slots).

  python runner.py --deadline 6900 --workers 2 --threads 1 q_main.txt q_robust.txt ...

Every job gets --budget so it stops cleanly before the deadline; unfinished jobs resume on the next launch
(the attack script caches every detector and chunk). Finished jobs are recorded in done.txt.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/claude/rpromani251/geospatial-xai-attacks")
PY = "/home/claude/venv310/bin/python"
HERE = Path(__file__).resolve().parent


def name(args):
    return re.sub(r"[^A-Za-z0-9.]+", "_", args.replace("--", ""))[:80].strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queues", nargs="+")
    ap.add_argument("--deadline", type=float, default=6900)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    done_f = HERE / "done.txt"
    done = set(done_f.read_text().splitlines()) if done_f.exists() else set()
    jobs = []
    for q in a.queues:
        for line in Path(q).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line not in done and line not in jobs:
                jobs.append(line)
    logd = HERE / "logs"
    logd.mkdir(exist_ok=True)
    log = open(HERE / "runner.log", "a")

    def say(msg):
        log.write(f"{time.strftime('%F %T')} {msg}\n")
        log.flush()

    say(f"start: {len(jobs)} jobs left, deadline {a.deadline:.0f}s")
    running = {}
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    while jobs or running:
        for p, (args, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                last = [l for l in (logd / f"{name(args)}.log").read_text().splitlines() if l.strip()]
                last = last[-1] if last else ""
                if last.startswith("ALL DONE"):
                    done.add(args)
                    with open(done_f, "a") as f:
                        f.write(args + "\n")
                say(f"exit {p.returncode}: {last[:60]} | {args}")
                del running[p]
        left = a.deadline - (time.time() - t0)
        while jobs and len(running) < a.workers and left > 420:
            args = jobs.pop(0)
            speed = 3.0 if "--model rf" in args else 1.0
            fh = open(logd / f"{name(args)}.log", "a")
            cmd = [PY, "scripts/run_matched_attack.py", *args.split(), "--jobs", str(a.threads),
                   "--budget", str(int(left - 240)), "--speed", str(speed), "--clean-cache"]
            p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[p] = (args, fh)
            say(f"launch (budget {int(left - 240)}s) | {args}")
        if not running:
            break
        time.sleep(5)
    say(f"stop: {len(done)} done in total; {len(jobs)} not started this slot")


if __name__ == "__main__":
    main()
