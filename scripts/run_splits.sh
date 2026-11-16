#!/bin/bash
# qclip v1 five-split driver: for each split seed k, trains <config> with training seed k on split k, then benchmarks
# best.pt on every test set of the config into results/v1/<tag>/k<k>.json (existing outputs are skipped).
#   bash scripts/run_splits.sh <tag> <config> "<split seeds>" [--set key=value ...]
#   e.g. bash scripts/run_splits.sh svo_ps configs/v1/svo-probes/ps.yaml "0 1 2 3 4"
# PY (default: python) and X (a prefix, e.g. a container exec) can be set in the environment.
set -uo pipefail
tag=$1 cfg=$2 ks=$3; shift 3
PY=${PY:-python} X=${X:-}
export PYTHONPATH=${PYTHONPATH:-.}
for k in $ks; do
  run=v1_${tag}_k${k}; out=results/v1/$tag/k${k}.json
  echo "=== $run | $(date -Is) | $(hostname -s)"
  [[ -e $out ]] && { echo "skip: $out exists"; continue; }
  mkdir -p results/v1/$tag
  $X $PY scripts/train.py -cfg $cfg --seed $k --set run_name=$run --set mlflow_db=mlf_dbs/v1_${tag}_k${k}.db \
      --set dataset.split.seed=$k "$@" || { echo "FAILED train $run"; continue; }
  ck=$(ls -d checkpoints/*/*/$run 2>/dev/null | head -1)/best.pt
  $X $PY scripts/benchmark.py -cp $ck -o $out || echo "FAILED benchmark $run"
done
echo "=== all done $(date -Is)"
