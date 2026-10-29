#!/bin/bash
# run_b2u.sh <cores> <set>: only the qlimit-map benchmark rerun (exact + 3 Aer backends) -> results/bench_b2/<set>
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 AER_THREADS=8
S=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc
s=$2
case ${s%_*} in ps) ck=$S/hp_hp2_nodiscard_tl3_qlr3e-3_s6/best.pt;; discard) ck=$S/hp_kd1_discard_kd_ws_s3/best.pt;; discard6) ck=$S/hp_kd1_discard_kd_ws_s6/best.pt;; esac
meas=$([ ${s##*_} = swap ] && echo destructive_swap || echo compute_uncompute)
taskset -c $1 /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/backends.py -cp $ck --meas $meas -o results/bench_b2/$s --reps 5 --shots 8192 2>&1 | grep -E "Device|exact accuracy| acc |done|Error|Traceback"
