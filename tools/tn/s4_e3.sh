#!/usr/bin/env bash
# Step-4 3-epoch budget (as step 3: dec_t3 defaults = cosine, warmup 1, 3 epochs; benchmark last.pt) for C10 and C11, seeds 0-2.
# Skips runs that already have benchmark_last.json.   bash tools/tn/s4_e3.sh <cores>
set -u
PY="taskset -c $1 /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
for c in c10 c11; do for s in 0 1 2; do
  R=tn_s4e3_${c}_s$s; D=checkpoints/mscoco/tn/$R
  [ -f $D/benchmark_last.json ] && continue
  X=(--set text.scale_free=true --set 'text.constraint={init_orth: true}'); [ $c = c11 ] && X+=(--set grad_clip=1.0)
  [ -d $D ] || $PY scripts/train.py -cfg configs/tn/dec_t3.yaml --seed $s --run_name $R --set mlflow_db=mlf_dbs/$R.db --set tn_stats=true "${X[@]}"
  echo "[e3] $R train exit=$? $(date -Is)"
  $PY scripts/benchmark.py -cp $D/last.pt; echo "[e3] $R benchmark exit=$? $(date -Is)"
done; done
