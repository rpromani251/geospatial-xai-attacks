#!/usr/bin/env bash
# Run a queue of run_matched_attack.py jobs with a fixed number of parallel workers.
# Resumable: finished jobs print ALL DONE and exit at once, so rerunning the queue is safe.
#
#   bash scripts/run_matched_queue.sh QUEUE_FILE [WORKERS] [THREADS_PER_WORKER]
#
# QUEUE_FILE has one job per line: the arguments for run_matched_attack.py, e.g.
#   --design feature --seed 4 --skip-honest-shapley --frontier
# Lines starting with # are ignored. --clean-cache is added to every job.
# Logs: results/matched_attack/_logs/<design>_<seed>_<n>.log ; progress: results/matched_attack/_logs/queue.log
set -u
cd "$(dirname "$0")/.."
QUEUE=${1:?queue file}
NCPU=$( (sysctl -n hw.ncpu 2>/dev/null || nproc) )
WORKERS=${2:-$(( NCPU / 2 > 0 ? NCPU / 2 : 1 ))}
THREADS=${3:-2}
PY=${PY:-python}
LOGDIR=results/matched_attack/_logs
mkdir -p "$LOGDIR"
echo "$(date '+%F %T') start queue=$QUEUE workers=$WORKERS threads=$THREADS" | tee -a "$LOGDIR/queue.log"

run_job() {
  local args="$1" n="$2"
  local name; name=$(echo "$args" | sed -E 's/--//g; s/[^A-Za-z0-9.]+/_/g' | cut -c1-80)
  local free_kb; free_kb=$(df -k . | awk 'NR==2 {print $4}')
  if [ "$free_kb" -lt 1500000 ]; then
    echo "$(date '+%F %T') SKIP (less than 1.5 GB free) $args" | tee -a "$LOGDIR/queue.log"; return
  fi
  OMP_NUM_THREADS=$THREADS $PY scripts/run_matched_attack.py $args --jobs "$THREADS" --clean-cache \
      > "$LOGDIR/${name}.log" 2>&1
  echo "$(date '+%F %T') $(tail -1 "$LOGDIR/${name}.log") | $args" | tee -a "$LOGDIR/queue.log"
}

n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue;; esac
  n=$((n + 1))
  run_job "$line" "$n" &
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
done < "$QUEUE"
wait
echo "$(date '+%F %T') queue finished" | tee -a "$LOGDIR/queue.log"
