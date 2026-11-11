#!/bin/bash
# exact optimal contraction path (dynamic programming, 30 min / 64 GB cap per network) + a 2 min anytime trace, smallest first:
# trained 2-layer circuits at 6-13 qubits (where the optimum should be computable), then 16/19/22 qubits at 2 layers (likely timeouts, recorded)
#$ -S /bin/bash
#$ -N anyt_opt
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"; grep -m1 'model name' /proc/cpuinfo
H=$HOME/qclip_hpc; cd $H/code/0bd46bf
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=1,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=0bd46bf $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
T=""; for c in e2_n1 e3_n1 e4_n1 e5_n1 e6_n1 e3_n2 e7_n1 e8_n1 e3_n3 e3_n4; do T="$T $CK/hp_sc4_nodiscard_${c}_tl2_qlr3e-3_s0/best.pt"; done
for c in e3_n5 e3_n6; do T="$T $CK/hp_sc4i_nodiscard_${c}_tl2_qlr3e-3_s0/best.pt"; done
$X analysis/bench/path_anytime.py -cp $T -o $O/anyt_opt.csv --budget 120 --opt_s 1800 --mem_gb 64
echo "[job] end=$(date -Is)"
