#!/usr/bin/env bash
# Step-4 lane: runs a queue of s4_run.sh jobs one after another on one CPU half of a pool GPU. Idempotent: relaunching
# after a kill skips finished runs (benchmark_best.json), resumes partial ones (last.pt with state), benchmarks trained ones.
#   bash tools/tn/s4_lane.sh <cores> <queue file>     queue line: <run_name> <config> <seed> <epochs> <lr> [--set k=v ...]
set -u
CORES=$1 Q=$2
PY="taskset -c $CORES /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
FX="--set lr_schedule=plateau --set warmup_epochs=1 --set lr_patience=100000 --set patience=100000 --set min_delta=0.0 --set select_metric=mrr_mean --set save_last=true --set save_best=true --set save_state=true --set tn_stats=true"
while read -r R C S E LR EXTRA; do
  EXTRA=${EXTRA:-}
  [ -z "${R:-}" ] && continue; case $R in \#*) continue;; esac
  D=checkpoints/mscoco/tn/$R
  eval "XA=($EXTRA)"   # extras may hold quoted YAML values, e.g. --set 'text.constraint={norm: all}'
  if [ -f $D/benchmark_best.json ]; then echo "[lane] $R done, skip"; continue; fi
  done_ep=$(python3 -c "import csv,sys;r=list(csv.DictReader(open('$D/steps.csv')));print(int(r[-1]['epoch'])+1)" 2>/dev/null || echo 0)
  if [ -f $D/last.pt ] && [ "$done_ep" -lt "$E" ] && [ ! -f $D/diverged.txt ]; then
    echo "[lane] $R resume after $done_ep epochs $(date -Is)"
    $PY scripts/train.py --resume $D/last.pt --epochs $E; rc=$?
  elif [ ! -d $D ]; then
    echo "[lane] $R start $(date -Is)"
    $PY scripts/train.py -cfg configs/tn/$C --seed $S --run_name $R --epochs $E --set lr=$LR --set mlflow_db=mlf_dbs/$R.db $FX "${XA[@]}"; rc=$?
  else
    echo "[lane] $R exists (epochs $done_ep, diverged: $(cat $D/diverged.txt 2>/dev/null || echo no)): no training"; rc=0
  fi
  echo "[lane] $R train exit=$rc $(date -Is)"
  [ -f $D/best.pt ] && $PY scripts/benchmark.py -cp $D/best.pt; echo "[lane] $R benchmark exit=$? $(date -Is)"
done < $Q
echo "[lane] queue $Q finished $(date -Is)"
