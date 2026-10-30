#!/usr/bin/env bash
# Step-4 phase-0 checks (one pool GPU, 2 lanes): A = old code (qclip_wt/step3) vs new code, defaults; B = new code with
# tn_stats + save_state, straight 2 epochs vs 1 epoch + --resume to 2. Compare with tools/tn/s4_cmp.py.
#   bash tools/tn/s4_check.sh <A|B> <tag>
set -u
lane=$1 tag=$2
NEW=/cs/research/pplv/comp_bridge/qclip_wt/step4 OLD=/cs/research/pplv/comp_bridge/qclip_wt/step3
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
FL="--seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set select_metric=mrr_mean --set save_best=false"
if [ $lane = A ]; then
  PY="taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python"
  (cd $OLD && $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL --run_name ${tag}_old --set mlflow_db=mlf_dbs/${tag}_old.db); echo "old exit=$?"
  (cd $NEW && $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL --run_name ${tag}_new --set mlflow_db=mlf_dbs/${tag}_new.db); echo "new exit=$?"
else
  PY="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
  ST="--set tn_stats=true --set save_state=true"
  cd $NEW
  $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL $ST --run_name ${tag}_st --set mlflow_db=mlf_dbs/${tag}_st.db; echo "st exit=$?"
  $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 1 $FL $ST --run_name ${tag}_res --set mlflow_db=mlf_dbs/${tag}_res.db; echo "res1 exit=$?"
  $PY scripts/train.py --resume checkpoints/mscoco/tn/${tag}_res/last.pt --epochs 2; echo "res2 exit=$?"
fi
