#!/bin/bash
# energy figure: smoke of qpu_iqm.py on the 16-qubit 4-layer circuit (CPU)
#$ -S /bin/bash
#$ -N ef_iqm_smoke
#$ -l h_rt=01:00:00
#$ -l tmem=16G
#$ -l h_vmem=32G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/136a567
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=136a567 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
$X analysis/bench/qpu_iqm.py -cp $CK/hp_sc4i_nodiscard_e3_n4_tl4_qlr3e-3_s0/best.pt -o results/shotcost/adv/qpu_iqm_smoke.csv --k 1 --seeds 2
cat results/shotcost/adv/qpu_iqm_smoke.csv
echo "[job] end=$(date -Is)"
