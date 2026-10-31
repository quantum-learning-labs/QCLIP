#!/usr/bin/env bash
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=4
PY="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python"; C=checkpoints/mscoco/tn; O=results/tn/step4/recall
for r in tn_s4b_bow_s0 tn_s4f_t2c11_s2 tn_s4f_c10_s0 tn_s4h_c10e_s1; do $PY tools/tn/s4_influence.py $C/$r/best.pt $O/influence_$r.json; echo "infl $r exit=$?"; done
