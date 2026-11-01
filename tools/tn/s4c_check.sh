#!/usr/bin/env bash
# Step-4c checks (one pool GPU, 2 lanes). A: old code (qclip_wt/step4) vs new code with the step-4 recommended setup, 2 epochs,
# same seed (steps.csv losses must be bit-identical; compare with tools/tn/s4_cmp.py). B: tools/tn/s4c_test.py on a step-4
# checkpoint, then 2-epoch smokes of V1 (hybrid), V2 (word_base 0.3), V3 (+ centre) for speed and memory.
#   bash tools/tn/s4c_check.sh <A|B> <tag>
set -u
lane=$1 tag=$2
NEW=/cs/research/pplv/comp_bridge/qclip_wt/step4c OLD=/cs/research/pplv/comp_bridge/qclip_wt/step4
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
FL="--seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set select_metric=mrr_mean --set save_best=false --set save_last=false"
REC=(--set text.scale_free=true --set 'text.constraint={init_orth: true}' --set text.head=no_bias)
if [ $lane = A ]; then
  PY="taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python"
  (cd $OLD && $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL "${REC[@]}" --run_name ${tag}_old --set mlflow_db=mlf_dbs/${tag}_old.db); echo "old exit=$?"
  (cd $NEW && $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL "${REC[@]}" --run_name ${tag}_new --set mlflow_db=mlf_dbs/${tag}_new.db); echo "new exit=$?"
else
  PY="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
  cd $NEW
  $PY tools/tn/s4c_test.py checkpoints/mscoco/tn/tn_s4b_hb_c10nobias_s0/best.pt 1024; echo "test exit=$?"
  for v in "v1 --set text.compose=hybrid" "v2 --set text.word_base=0.3" "v3 --set text.word_base=0.3 --set text.centre=true"; do
    set -- $v; n=$1; shift
    $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 $FL "${REC[@]}" "$@" --run_name ${tag}_$n --set mlflow_db=mlf_dbs/${tag}_$n.db; echo "$n exit=$?"
  done
fi
