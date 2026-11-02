export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_noise.py -o results/shotcost/e6 -n 300"; C=/cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_discard_n2_tl3_pw0_ws_s0/best.pt
$P -cp $C --profiles honest_miami --det_khz 10 --zz_khz 20
$P -cp $C --profiles honest_miami --dd --det_khz 10 --zz_khz 20
