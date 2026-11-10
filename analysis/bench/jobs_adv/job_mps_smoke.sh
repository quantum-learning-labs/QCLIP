#!/bin/bash
# smoke: MPS engine (cuTensorNet) chi sweep vs exact TN at 16, 19, 31 qubits (2 layers), 6 decisions, short energy windows
#$ -S /bin/bash
#$ -N mps_smoke
#$ -l h_rt=2:00:00
#$ -l tmem=32G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=(a6000|a100|rtx3090|a40|rtx4090|rtx6000|l4|a10|rtx4070ti)
#$ -l h=!hogthrob-206-2*
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/0d1f4f2
SP=$(ls -d $H/env/lib/python3*/site-packages)
PRE=$SP/nvidia/nvjitlink/lib/libnvJitLink.so.12:$SP/nvidia/cublas/lib/libcublasLt.so.12:$SP/nvidia/cublas/lib/libcublas.so.12:$SP/nvidia/cusparse/lib/libcusparse.so.12:$H/libs_cusolver12/nvidia/cusolver/lib/libcusolver.so.11
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=0d1f4f2,LD_PRELOAD=$PRE $H/ubuntu2404"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
$X $H/env/bin/python analysis/bench/mps_engine.py -cp $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt $CK/hp_sc4i_nodiscard_e3_n5_tl2_qlr3e-3_s0/best.pt $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/best.pt \
   -o results/shotcost/mps_smoke --pairs 6 --chis 2,4,8,16,32,64 --min_s 2
echo "[job] end=$(date -Is)"
