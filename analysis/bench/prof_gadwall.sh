#!/bin/bash
# prof_aer.py on gadwall CPU: full coupling map vs qlimit-restricted, 16 compute-uncompute circuits
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
C=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_hp2_nodiscard_tl3_qlr3e-3_s6/best.pt
for ql in none meta; do echo "##### coupling map: $ql"; taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/prof_aer.py results/bench_b1/ps_cu $C 16 $ql 2>&1 | grep -vE "it/s]|s/it]|Transpiling"; done
