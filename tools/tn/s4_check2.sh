#!/usr/bin/env bash
# Step-4 checks, part 2: resume tn_s4_c1_res (1 epoch, save_state) to 2 epochs; defaults regression of the phase-1 code
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
PY="taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python"
$PY scripts/train.py --resume checkpoints/mscoco/tn/tn_s4_c1_res/last.pt --epochs 2; echo "res2 exit=$?"
$PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 --seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set select_metric=mrr_mean --set save_best=false --run_name tn_s4_c1_new2 --set mlflow_db=mlf_dbs/tn_s4_c1_new2.db; echo "new2 exit=$?"
