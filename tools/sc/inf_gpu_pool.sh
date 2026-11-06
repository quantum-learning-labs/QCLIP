cd /cs/research/pplv/comp_bridge/qclip_wt/shotcost
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/cs/research/pplv/comp_bridge/.venv/bin/python; S=/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc; V=/cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc
for sfx in d s; do O=results/shotcost/viz_sc4_infq_$sfx; mkdir -p $O; [ -f $O/qpu_stats.csv ] || echo "ckpt,text_n,idx,nq,kind,ibm_duration_s,ibm_n2q,ibm_n1q,ionq_n1q,ionq_n2q" > $O/qpu_stats.csv; done
L="$S/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt"; for n in 5 6 7 8; do L="$L $V/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
taskset -c 0-15 $PY analysis/bench/v3_energy.py -cp $L -o results/shotcost/viz_sc4_infq_d --pairs 40 --no_scaling --precision double --nq_cap 30 --readout probs
taskset -c 0-15 $PY analysis/bench/v3_energy.py -cp $V/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/best.pt -o results/shotcost/viz_sc4_infq_s --pairs 40 --no_scaling --precision single --nq_cap 31 --readout probs
echo "=== inf gpu done $(date -Is)"
