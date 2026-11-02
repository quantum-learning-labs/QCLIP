export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
for c in hp_xo1_nodiscard_n2_tl3_qlr0.003_s0 hp_xo1_discard_n2_tl3_pw0_ws_s0; do
taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_noise.py -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/$c/best.pt -o results/shotcost/e6 --profiles ideal,ibm_kingston,honest_miami -n 300
done
