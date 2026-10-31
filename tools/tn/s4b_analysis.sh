#!/usr/bin/env bash
# Step-4b: regression of the 4b code (defaults, 2 epochs vs tn_s4_c1_old), CLIP ceiling, embedding analysis of 4 checkpoints
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
PY="taskset -c 16-23 /cs/research/pplv/comp_bridge/.venv/bin/python"; C=checkpoints/mscoco/tn; O=results/tn/step4/recall
$PY scripts/train.py -cfg configs/tn/dec_t3.yaml --epochs 2 --seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set select_metric=mrr_mean --set save_best=false --run_name tn_s4_c1_new4 --set mlflow_db=mlf_dbs/tn_s4_c1_new4.db; echo "new4 exit=$?"
python3 tools/tn/s4_cmp.py $C/tn_s4_c1_old $C/tn_s4_c1_new4
$PY tools/tn/s4_clip_ceiling.py $O/clip_ceiling.json; echo "ceiling exit=$?"
for r in tn_s4f_c10_s0 tn_s4h_c10e_s1 tn_s4f_c0_s2 tn_s4f_t2c11_s2; do
  $PY tools/tn/s4_embed.py $C/$r/best.pt $O/embed_$r.json; echo "embed $r exit=$?"
done
