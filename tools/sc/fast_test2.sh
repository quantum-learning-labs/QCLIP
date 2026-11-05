cd /cs/research/pplv/comp_bridge/qclip_wt/shotcost
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
PY=/cs/research/pplv/comp_bridge/.venv/bin/python; ck=/cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_sc4_discard_e3_n4_tl2_pw0_r2_s0/best.pt
nice $PY analysis/bench/shotcost_branches_fast.py --engine tn --min_nq 21 --sets std -o results/shotcost/fast_test2/tn -cp $ck 2>&1 | grep -av "Compiling\|it/s\|pair/s" | tail -6 | cut -c1-220
$PY analysis/bench/cmp_branches.py results/shotcost/e4_sc4/hp_sc4_discard_e3_n4_tl2_pw0_r2_s0__std.npz results/shotcost/fast_test2/tn/hp_sc4_discard_e3_n4_tl2_pw0_r2_s0__std.npz
