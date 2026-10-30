#!/usr/bin/env bash
# Step-4 check 4: resume tn_s4_c1_res (1 epoch, save_state) to 2 epochs (compare epoch 1 with tn_s4_c1_st)
set -u
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python scripts/train.py --resume checkpoints/mscoco/tn/tn_s4_c1_res/last.pt --epochs 2; echo "res2 exit=$?"
