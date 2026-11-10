#!/bin/bash
# QPU gate counts with all readout circuits, grid checkpoints at 8 layers (replaces the qpu_esp step of grid_l8)
#$ -S /bin/bash
#$ -N esp2_l8
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
$X analysis/bench/qpu_esp.py -cp $(ck e3_n25_tl8 e3_n32_tl8 e3_n40_tl8 e6_n15_tl8 e8_n20_tl8 e8_n32_tl8) -o $O/qpu_esp2_grid_l8.csv --k 5 --circ all
echo "[job] end=$(date -Is)"
