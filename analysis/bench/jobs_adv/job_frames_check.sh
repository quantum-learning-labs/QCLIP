#!/bin/bash
# frames(typical=k) vs the old compile-all-then-filter: same 40 typical decisions (identical circuits), and the compile time of each.
#$ -S /bin/bash
#$ -N frames_chk
#$ -l h_rt=3:00:00
#$ -l tmem=16G
#$ -l h_vmem=48G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/9881307
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=9881307 $H/ubuntu2404 $H/env/bin/python"
CK=checkpoints/svo-probes/vqc
$X analysis/bench/frames_filter_check.py $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/best.pt $CK/hp_sc4i_nodiscard_e3_n15_tl8_qlr3e-3_s0/best.pt 2>&1 | grep -v "Compiling Qiskit DAGs"
echo "[job] end=$(date -Is)"
