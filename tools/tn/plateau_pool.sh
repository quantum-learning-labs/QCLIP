#!/usr/bin/env bash
# Step-3 plateau run on a pool GPU (same settings as the cluster jobs qclip/jobs/tn_s3_plateau/): train then benchmark best.pt.
#   bash tools/tn/plateau_pool.sh <run_name> <config> <seed> <cores> [--set ...]
set -u
R=$1 C=$2 S=$3 CORES=$4; shift 4
PL="--set lr_schedule=plateau --set warmup_epochs=1 --set lr_patience=4 --set patience=10 --set min_delta=0.0001 --set select_metric=mrr_mean --set epochs=150 --set save_last=false --set save_best=true"
PY="taskset -c $CORES /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
$PY scripts/train.py -cfg configs/tn/$C --seed $S --run_name $R --set mlflow_db=mlf_dbs/$R.db $PL "$@"
echo "[pool] $R train exit=$? $(date -Is)"
$PY scripts/benchmark.py -cp checkpoints/mscoco/tn/$R/best.pt "$@"
echo "[pool] $R benchmark exit=$? $(date -Is)"
