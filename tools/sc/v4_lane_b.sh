export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
until grep -q "exit=" /cs/research/pplv/comp_bridge/logs/sc_e6_b.log; do sleep 60; done   # start when this lane's E6 run ends
P="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_noise.py -o results/shotcost/e6 -n 300"
$P -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_discard_n2_tl3_pw0_ws_s0/best.pt --profiles honest_miami --dd
$P -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_discard_n2_tl3_pw0_ws_s0/best.pt --profiles honest_miami --det_khz 10 --zz_khz 20
$P -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_discard_n2_tl3_pw0_ws_s0/best.pt --profiles honest_miami --dd --det_khz 10 --zz_khz 20
$P -cp /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_do1_discard_tl3_pw0_initnd_s0/best.pt --profiles ideal,ibm_kingston,honest_miami
