export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_noise.py -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_discard_n2_tl3_pw0_s0/best.pt -o results/shotcost/e6 --profiles ideal,ibm_kingston,honest_miami -n 300
