#!/usr/bin/env bash
# Run several step-1 reference trainings one after another on the same GPU, each with its own log in comp_bridge/logs.
#   bash tools/tn/seq.sh <config> <tag> <seed>...   (run names tn_s1_<tag>_s<seed>)
set -u
cfg=$1 tag=$2; shift 2
for s in "$@"; do
  run=tn_s1_${tag}_s$s
  [ -e ../logs/$run.log ] && { echo "$run exists, skipped"; continue; }
  PYTHONPATH=. /cs/research/pplv/comp_bridge/.venv/bin/python scripts/train.py -cfg $cfg --seed $s --run_name $run \
      --set mlflow_db=mlf_dbs/$run.db > ../logs/$run.log 2>&1
  echo "$run exit=$?"
done
