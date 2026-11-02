export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
L=/cs/research/pplv/comp_bridge/logs
until grep -q "exit=" $L/sc_v4b_a.log && grep -q "exit=" $L/sc_v4b_b.log; do sleep 60; done   # GPU to itself for NVML
exec taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/v3_energy.py -o results/shotcost/viz -cp  /cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_hp2_nodiscard_tl3_qlr3e-3_s0/best.pt /cs/research/pplv/comp_bridge/qclip/checkpoints/svo-probes/vqc/hp_xo1_nodiscard_n2_tl3_qlr0.003_s0/best.pt /cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_sc3_nodiscard_n3_tl3_qlr3e-3_s0/best.pt
