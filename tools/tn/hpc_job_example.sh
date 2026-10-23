#!/bin/bash
# TN train + benchmark on one cluster GPU (template: copy, EDIT run name, seed, config, code sha). Submit from the MSI:
#   ~/comp_bridge_agent/bin/submit.sh <copy>.sh   (code copy: git archive <sha> -> ~/qclip_hpc/code/<sha>, data -> symlink to ~/qclip_hpc/data_tn)
#$ -S /bin/bash
#$ -N tn_<tag>_s<seed>
#$ -l h_rt=08:00:00
#$ -l tmem=48G
#$ -l h_vmem=64G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=a100_80|h100|a100|l40s|a6000|a40|rtx6000ada|rtx4090|rtx3090|a10|l4
#$ -l h=!dip-207-2*&!strangepork*&!seymour*&!animal-206-2*
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv
H=$HOME/qclip_hpc; cd $H/code/ccd1411
export CUDA_CACHE_PATH=$HOME/.nv_jit_cache
X="singularity exec --nv -B $HOME --env PYTHONPATH=.,PYTHONUNBUFFERED=1,QCLIP_GIT_COMMIT=ccd1411 $H/ubuntu2404"
R=tn_TAG_sSEED   # EDIT: run name; also --seed and the config below
$X $H/env/bin/python scripts/train.py -cfg configs/tn/ref_clip.yaml --seed 0 --run_name $R --set mlflow_db=mlf_dbs/$R.db 2>&1 | grep --line-buffered -vE 'it/s\]$|s/it\]$'
$X $H/env/bin/python scripts/benchmark.py -cp checkpoints/mscoco/tn/$R/last.pt 2>&1 | grep --line-buffered -vE 'it/s\]$|s/it\]$'
echo "[job] end=$(date -Is)"
