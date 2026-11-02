export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P="taskset -c 8-15 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_noise.py.new -cp /cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_hp2_nodiscard_tl3_qlr3e-3_s0/best.pt -o results/shotcost/v4_smoke -n 6 --qmax 8 --profiles honest_miami"
$P; $P --rebuild; $P --dd; $P --det_khz 10 --zz_khz 20; $P --dd --det_khz 10 --zz_khz 20
