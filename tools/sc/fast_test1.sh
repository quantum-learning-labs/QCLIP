cd /cs/research/pplv/comp_bridge/qclip_wt/shotcost
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY=/cs/research/pplv/comp_bridge/.venv/bin/python; Q=/cs/research/pplv/comp_bridge/qclip
for m in discard_e3_n3_tl2_pw0_s1 nodiscard_e3_n3_tl2_qlr3e-3_s1; do
  ck=$Q/checkpoints/svo-probes/vqc/hp_sc4_$m/best.pt; [ -f $ck ] || ck=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_sc4_$m/best.pt
  for e in sv tn gpu; do
    taskset -c 8-15 $PY analysis/bench/shotcost_branches_fast.py --engine $e -l 150 --sets std -o results/shotcost/fast_test1/$e -cp $ck 2>&1 | grep -av "Compiling\|it/s\|pair/s" | tail -4
    echo "--- $m $e vs exact reference"; $PY analysis/bench/cmp_branches.py results/shotcost/e4_sc4/hp_sc4_${m}__std.npz results/shotcost/fast_test1/$e/hp_sc4_${m}__std.npz
  done
done
