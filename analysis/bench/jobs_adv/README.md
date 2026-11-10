# Cluster job scripts for the advantage study (shot/energy cost, QPU vs best classical)

SGE scripts submitted from the MSI with `~/comp_bridge_agent/bin/submit.sh <script> [-hold_jid name]` (copies of tools/sc5 on the MSI).
Each bakes in its cluster code dir `~/qclip_hpc/code/<sha>` (git archive of this branch + symlinks to d8c1b80 data/configs and the
39974f7 checkpoints). Outputs land in `<code dir>/results/shotcost/adv/`; copies used by the page are in results/shotcost/adv/adv_data_msi/
(aggregated by analysis/bench/viz_page/adv_agg.py into adv.json).

Main groups: job_e1_* (L4 GPU energy), job_e3_*/tw_*/ctg* (contraction paths), job_e5_*/esp* (QPU gate counts), job_grid_l* (energy grid,
e and n grown together), job_fill_* (4/8-layer 16-34 qubit cells), job_esp3_* (best of 5 Kingston transpile seeds), job_ctg3_* (cotengra,
greedy trials, 600 s + hard stop, every point from 40 qubits).
