#!/usr/bin/env bash
# Step-2 runs: train configs/tn/ref_clip.yaml for each seed, then benchmark last.pt; one GPU, sequential, P-cores.
#   bash tools/tn/seq2.sh <tag> <seed>... [-- --set k=v ...]   (run names tn_s2_<tag>_s<seed>; logs in comp_bridge/logs)
#   CFG=configs/tn/dec_t3.yaml PREFIX=tn_s3 bash tools/tn/seq2.sh ...   (other config / run-name prefix)
set -u
tag=$1; shift
seeds=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do seeds+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
LOGS=/cs/research/pplv/comp_bridge/logs
PY="taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1
for s in "${seeds[@]}"; do
  run=${PREFIX:-tn_s2}_${tag}_s$s
  [ -e $LOGS/$run.log ] && { echo "$run exists, skipped"; continue; }
  $PY scripts/train.py -cfg ${CFG:-configs/tn/ref_clip.yaml} --seed $s --run_name $run --set mlflow_db=mlf_dbs/$run.db "$@" > $LOGS/$run.log 2>&1
  echo "$run train exit=$?"
  $PY scripts/benchmark.py -cp checkpoints/mscoco/tn/$run/last.pt "$@" > $LOGS/${run}_bench.log 2>&1
  echo "$run benchmark exit=$?"
done
