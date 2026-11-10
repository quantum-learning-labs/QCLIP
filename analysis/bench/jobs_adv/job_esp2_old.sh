#!/bin/bash
# QPU gate counts with all readout circuits for the earlier e=3 checkpoints (2/4/8 layers, n4-n40), so every point is priced on the same CU circuits
#$ -S /bin/bash
#$ -N esp2_old
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/8e1eb8b
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=8e1eb8b $H/ubuntu2404 $H/env/bin/python"
CK=checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; done; }
$X analysis/bench/qpu_esp.py -cp $(ck e3_n5_tl2 e3_n6_tl2 e3_n7_tl2 e3_n8_tl2 e3_n9_tl2 e3_n10_tl2 e3_n12_tl2 e3_n15_tl2 e3_n20_tl2 e3_n25_tl2 e3_n32_tl2 e3_n40_tl2 e3_n5_tl4 e3_n9_tl4 e3_n12_tl4 e3_n15_tl4 e3_n20_tl4 e3_n5_tl8 e3_n9_tl8 e3_n12_tl8 e3_n15_tl8 e3_n20_tl8) $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt -o $O/qpu_esp2_old.csv --k 5 --circ all
echo "[job] end=$(date -Is)"
