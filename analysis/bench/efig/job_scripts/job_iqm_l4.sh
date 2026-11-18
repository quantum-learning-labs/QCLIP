#!/bin/bash
# energy figure: IQM (Aphrodite, best of 5) stats for the existing 4-layer e=3 circuits (16-124 q; >54 q get logical-count rows)
#$ -S /bin/bash
#$ -N ef_iqm_l4
#$ -l h_rt=12:00:00
#$ -l tmem=16G
#$ -l h_vmem=64G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/136a567
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=136a567 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
B=""; for n in 4 5 6 7 8 9 10 12 15 20 25 32 40; do B="$B $CK/hp_sc4i_nodiscard_e3_n${n}_tl4_qlr3e-3_s0/best.pt"; done
$X analysis/bench/qpu_iqm.py -cp $B -o $O/qpu_iqm_ef_l4.csv --k 3 --seeds 5
echo "[job] end=$(date -Is)"
