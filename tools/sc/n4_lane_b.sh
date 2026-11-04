cd /cs/research/pplv/comp_bridge/qclip_wt/shotcost; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
R="taskset -c 16-23 bash scripts/run_hp.sh sc4 discard"
$R e3_n4_tl2_pw0 2 -- --set mlflow_db=mlf_dbs/sc4_pl_e3_n4_tl2_s2.db --set text.n=4 --set text.layers=2 --set embedding_qubits=3 --set purity_weight=0.0 --set text.trace_mode=auto
$R e3_n4_tl2_pw0_ws 2 -- --set mlflow_db=mlf_dbs/sc4_ws_e3_n4_tl2_s2.db --set text.n=4 --set text.layers=2 --set embedding_qubits=3 --set purity_weight=0.0 --set text.trace_mode=auto --init_from /cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s2/best.pt
