#!/bin/bash
# which GPU SVD backends work in the cluster env (torch cusolver/magma, cupy, cuTensorNet; with and without preloading pip CUDA libs)
#$ -S /bin/bash
#$ -N svd_test
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
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache $H/ubuntu2404"
SP=$(ls -d $H/env/lib/python3*/site-packages)
echo "--- pip CUDA libs:"; ls $SP/nvidia/*/lib/ 2>/dev/null | grep -E "cusolver|nvJitLink|cublas|cusparse" | sort -u
echo "=== plain"; $X $H/env/bin/python analysis/bench/gpu_svd_test.py 2>&1 | grep -v Warning
JL=$(ls $SP/nvidia/nvjitlink/lib/libnvJitLink.so* 2>/dev/null | head -1)
CS=$(ls $SP/nvidia/cusolver/lib/libcusolver.so* 2>/dev/null | head -1)
CSP=$(ls $SP/nvidia/cusparse/lib/libcusparse.so* 2>/dev/null | head -1)
CB=$(ls $SP/nvidia/cublas/lib/libcublasLt.so* 2>/dev/null | head -1):$(ls $SP/nvidia/cublas/lib/libcublas.so* 2>/dev/null | head -1)
for pre in "$JL" "$JL:$CSP:$CS" "$JL:$CB:$CSP:$CS"; do
  echo "=== LD_PRELOAD=$pre"
  singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,PYTHONUNBUFFERED=1,LD_PRELOAD=$pre $H/ubuntu2404 $H/env/bin/python analysis/bench/gpu_svd_test.py 2>&1 | grep -E "OK|FAIL"
done
echo "[job] end=$(date -Is)"
