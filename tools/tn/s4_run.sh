#!/usr/bin/env bash
# Step-4 screen/stress run on a pool GPU: constant LR after a 1-epoch warmup (plateau rule disabled), fixed epoch budget,
# per-step stats (tn_stats), resumable last.pt (save_state), best.pt by val mrr_mean; then benchmark best.pt.
#   bash tools/tn/s4_run.sh <run_name> <config> <seed> <cores> <epochs> <lr> [--set ...]
#   resume after a kill: RESUME=1 bash tools/tn/s4_run.sh <same args>
set -u
R=$1 C=$2 S=$3 CORES=$4 E=$5 LR=$6; shift 6
FX="--set lr_schedule=plateau --set warmup_epochs=1 --set lr_patience=100000 --set patience=100000 --set min_delta=0.0 --set select_metric=mrr_mean --set save_last=true --set save_best=true --set save_state=true --set tn_stats=true --set lr=$LR"
PY="taskset -c $CORES /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
if [ "${RESUME:-0}" = 1 ]; then
  $PY scripts/train.py --resume checkpoints/mscoco/tn/$R/last.pt
else
  $PY scripts/train.py -cfg configs/tn/$C --seed $S --run_name $R --epochs $E --set mlflow_db=mlf_dbs/$R.db $FX "$@"
fi
echo "[pool] $R train exit=$? $(date -Is)"
$PY scripts/benchmark.py -cp checkpoints/mscoco/tn/$R/best.pt
echo "[pool] $R benchmark exit=$? $(date -Is)"
