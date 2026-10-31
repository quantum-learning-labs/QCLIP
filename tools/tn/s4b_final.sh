#!/usr/bin/env bash
# Step-4 report inputs on one pool GPU (alone, pinned): speed/memory profile (2 epochs each; run_meta.json epoch 1), then the
# geometry / influence / paraphrase analyses of the best 4b runs.
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
PY="taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python"; C=checkpoints/mscoco/tn; O=results/tn/step4/recall
FX="--epochs 2 --seed 0 --set lr_schedule=plateau --set warmup_epochs=1 --set save_best=false --set save_last=false"
CI="--set text.scale_free=true --set text.constraint={init_orth:\ true}"
run() { n=$1; shift; $PY scripts/train.py -cfg configs/tn/dec_t3.yaml $FX --run_name tn_s4_prof_$n --set mlflow_db=mlf_dbs/tn_s4_prof_$n.db "$@"; echo "prof $n exit=$?"; }
run c0
run c10 --set text.scale_free=true --set 'text.constraint={init_orth: true}'
run c11 --set text.scale_free=true --set 'text.constraint={init_orth: true}' --set grad_clip=1.0
run c10plain --set text.scale_free=true --set 'text.constraint={init_orth: true}' --set text.head=plain
run c11rb32 --set text.scale_free=true --set 'text.constraint={init_orth: true}' --set grad_clip=1.0 --set text.root_bond_dim=32
for r in tn_s4b_h0_c10plain_s0 tn_s4b_rb32_c11_s1; do
  $PY tools/tn/s4_embed.py $C/$r/best.pt $O/embed_$r.json; echo "embed $r exit=$?"
  $PY tools/tn/s4_influence.py $C/$r/best.pt $O/influence_$r.json; echo "infl $r exit=$?"
  $PY tools/tn/s4_paraphrase.py $C/$r/best.pt $O/para_$r.json; echo "para $r exit=$?"
  $PY tools/tn/s4_head.py $C/$r/best.pt; echo "head $r exit=$?"
done
