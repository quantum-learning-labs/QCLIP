#!/bin/bash
# Decisions per circuit shape on the std test set (how often a contraction path / transpiled layout is reused): compiles all ~1.8k decisions, 2 layers
#$ -S /bin/bash
#$ -N shape_reuse
#$ -l h_rt=06:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/3de11a1
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=3de11a1 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
$X analysis/bench/shape_reuse.py -cp $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt -o $O/shape_reuse_n4_l2.csv
$X analysis/bench/shape_reuse.py -cp $CK/hp_sc4i_nodiscard_e3_n12_tl8_qlr3e-3_s0/best.pt -o $O/shape_reuse_n12_l8.csv
echo "[job] end=$(date -Is)"
