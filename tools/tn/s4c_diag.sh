#!/usr/bin/env bash
# Step-4c diagnostics (word influence, embedding geometry / recall by length, paraphrase agreement) on checkpoints given as args.
#   bash tools/tn/s4c_diag.sh <cores> <run>...   (outputs results/tn/step4c/diag/<tool>_<run>.json; existing files are skipped)
set -u
CORES=$1; shift
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
PY="taskset -c $CORES /cs/research/pplv/comp_bridge/.venv/bin/python"; O=results/tn/step4c/diag; mkdir -p $O
for r in "$@"; do
  for t in influence embed paraphrase; do
    [ -f $O/${t}_$r.json ] && { echo "[diag] $t $r exists"; continue; }
    $PY tools/tn/s4_$t.py checkpoints/mscoco/tn/$r/best.pt $O/${t}_$r.json 2>&1 | grep -vE 'it/s\]$|s/it\]$' | tail -5
    echo "[diag] $t $r exit=$? $(date -Is)"
  done
done
