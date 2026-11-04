#!/usr/bin/env bash
# n=4 exact eval (e4_sc4 / e0_sc4 format) for all 9 sc4 n4 models; waits for warm s1 (lane sc4_n4_a) before its eval.
cd /cs/research/pplv/comp_bridge/qclip_wt/shotcost
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/cs/research/pplv/comp_bridge/.venv/bin/python
V=checkpoints/svo-probes/vqc; S=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc
ev() { taskset -c 16-23 $PY analysis/bench/shotcost_branches.py --trace_mode auto -o results/shotcost/e4_sc4 --e0_out results/shotcost/e0_sc4 --sets val,std,swap_full -cp "$@"; }
ev $V/hp_sc4_discard_e3_n4_tl2_pw0_r2_s0/best.pt $S/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt $S/hp_sc4_discard_e3_n4_tl2_pw0_ws_s0/best.pt \
   $S/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s1/best.pt $V/hp_sc4_discard_e3_n4_tl2_pw0_s1/best.pt \
   $S/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s2/best.pt $V/hp_sc4_discard_e3_n4_tl2_pw0_s2/best.pt $V/hp_sc4_discard_e3_n4_tl2_pw0_ws_s2/best.pt
until grep -q "=== all done" /cs/research/pplv/comp_bridge/logs/sc4_n4_a.log; do sleep 300; done
ev $V/hp_sc4_discard_e3_n4_tl2_pw0_ws_s1/best.pt
echo "=== n4 eval all done $(date -Is)"
