#!/usr/bin/env bash
# Step-4 check 3: text.scale_free (2 epochs, stats on) vs tn_s4_c1_st (same settings without it): losses should agree up to rounding
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
PY="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
$PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 --seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set select_metric=mrr_mean --set save_best=false --set tn_stats=true --set text.scale_free=true --run_name tn_s4_c1_sf --set mlflow_db=mlf_dbs/tn_s4_c1_sf.db; echo "sf exit=$?"
