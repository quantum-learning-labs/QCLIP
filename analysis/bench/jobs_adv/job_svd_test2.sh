#!/bin/bash
# GPU SVD with the CUDA-12 cuSOLVER (downloaded wheel, outside the env) preloaded: torch linalg + cuTensorNet in one process
#$ -S /bin/bash
#$ -N svd_test2
#$ -l h_rt=0:30:00
#$ -l tmem=16G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=(a6000|a100|rtx3090|rtx4070ti|l4|a10|rtx6000|a40|rtx4090)
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/7b5fde8
SP=$(ls -d $H/env/lib/python3*/site-packages)
PRE=$SP/nvidia/nvjitlink/lib/libnvJitLink.so.12:$SP/nvidia/cublas/lib/libcublasLt.so.12:$SP/nvidia/cublas/lib/libcublas.so.12:$SP/nvidia/cusparse/lib/libcusparse.so.12:$H/libs_cusolver12/nvidia/cusolver/lib/libcusolver.so.11
ls -la $SP/nvidia/cusparse/lib/ $SP/nvidia/cublas/lib/
singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,PYTHONUNBUFFERED=1,LD_PRELOAD=$PRE $H/ubuntu2404 $H/env/bin/python analysis/bench/gpu_svd_test.py 2>&1 | grep -v Warning
echo "[job] end=$(date -Is)"
