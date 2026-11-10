#!/bin/bash
# E1: per-decision GPU energy of the exact engines on an L4 (most energy-efficient GPU available: generous to the classical side).
# 2 layers 16-31 qubits: aer (v14 baseline), cusv, cusv_rho, lazy (<=28 live), tn, tn_rho; 34-64 qubits and layers 4/8: tn, tn_rho (width <= 29).
#$ -S /bin/bash
#$ -N e1_full
#$ -l h_rt=24:00:00
#$ -l tmem=48G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=l4
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/698a2be
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=698a2be $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
O=results/shotcost/adv/e1_l4
L2="$CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt"; for n in 5 6 7 8 9; do L2="$L2 $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
$X analysis/bench/inf_engines.py -cp $L2 -o $O --pairs 40 --engines aer,cusv,cusv_rho,lazy,tn,tn_rho --live_max 28 --tn_width_max 29
BIG=""; for n in 10 12 15 20; do BIG="$BIG $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
for l in 4 8; do for n in 5 9 12 15 20; do BIG="$BIG $CK/hp_sc4i_nodiscard_e3_n${n}_tl${l}_qlr3e-3_s0/best.pt"; done; done
$X analysis/bench/inf_engines.py -cp $BIG -o $O --pairs 40 --engines tn,tn_rho --tn_width_max 29
echo "[job] end=$(date -Is)"
