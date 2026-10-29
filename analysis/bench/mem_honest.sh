#!/bin/bash
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
C=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_hp2_nodiscard_tl3_qlr3e-3_s6/best.pt
for n in 64 190; do taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/mem_honest.py results/bench_b1/ps_cu $C $n 2>&1 | grep -E "maxrss|run|Vm"; done
