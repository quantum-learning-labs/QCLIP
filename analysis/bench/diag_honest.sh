#!/bin/bash
# memory diagnosis of debias.py on the honest profile (CPU, 20 pairs, 1 rep, nothing written)
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
/usr/bin/time -v /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/debias.py results/bench_b1/discard_cu --backends honest_miami --reps 1 --limit 20 --chunk 16 --ckpt /cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_kd1_discard_kd_ws_s3/best.pt 2>&1 | grep -vE "it/s]|s/it]" | grep -E "debias|Maximum resident|Elapsed|Error|Traceback|Killed|done" 
