#!/bin/bash
# E1: per-decision GPU energy of the exact engines on an L4 (most energy-efficient GPU available: generous to the classical side).
# 2 layers 16-31 qubits: aer (v14 baseline), cusv, cusv_rho, lazy (<=28 live), tn, tn_rho; 34-64 qubits and layers 4/8: tn, tn_rho (width <= 29).
#$ -S /bin/bash
#$ -N e1_big
#$ -l h_rt=24:00:00
#$ -l tmem=48G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=l4
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/04d6e92
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=04d6e92 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
O=results/shotcost/adv/e1_big
BIG="$CK/hp_sc4i_nodiscard_e3_n20_tl8_qlr3e-3_s0/best.pt"; for n in 25 32 40; do BIG="$BIG $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
$X analysis/bench/inf_engines.py -cp $BIG -o $O --pairs 40 --engines tn,tn_rho --tn_width_max 29
echo "[job] end=$(date -Is)"
