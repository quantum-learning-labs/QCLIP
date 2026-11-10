#!/bin/bash
# E4 (b): MPS bond-dimension sweep (torch, CUDA-12 cuSOLVER preloaded) on an L4: energy, time, error vs exact TN (or largest chi)
#$ -S /bin/bash
#$ -N e4_mps_b
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
SP=$(ls -d $H/env/lib/python3*/site-packages)
PRE=$SP/nvidia/nvjitlink/lib/libnvJitLink.so.12:$SP/nvidia/cublas/lib/libcublasLt.so.12:$SP/nvidia/cublas/lib/libcublas.so.12:$SP/nvidia/cusparse/lib/libcusparse.so.12:$H/libs_cusolver12/nvidia/cusolver/lib/libcusolver.so.11
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=698a2be,LD_PRELOAD=$PRE $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
L=""; for n in 5 9 12 15 20; do L="$L $CK/hp_sc4i_nodiscard_e3_n${n}_tl8_qlr3e-3_s0/best.pt"; done
$X analysis/bench/mps_engine.py -cp $L -o results/shotcost/adv/e4_mps_b --pairs 10 --chis 2,4,8,16,32,64,128,256 --min_s 5 --max_s_per_dec 120
echo "[job] end=$(date -Is)"
