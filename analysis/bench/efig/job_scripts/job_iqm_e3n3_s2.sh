#!/bin/bash
# energy figure, like-for-like calibration: IQM fake Garnet noisy DM emulation of PS models (3:3:13:2)
#$ -S /bin/bash
#$ -N ef_iqm_e3n3_s2
#$ -l h_rt=23:59:00
#$ -l tmem=48G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=a100|a100_80|h100|l40s|a6000|a40|rtx6000ada|rtx4090|rtx3090
#$ -l h=!dip-207-2*&!strangepork*&!seymour*&!animal-206-2*
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/45ae99b; mkdir -p $HOME/.nv_jit_cache
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache $H/ubuntu2404"
for sp in 3:3:13:2; do IFS=: read e n qmax s <<<"$sp"
  c=$CK/hp_sc4_nodiscard_e${e}_n${n}_tl2_qlr3e-3_s$s/best.pt; [ -f $c ] || { echo "[job] missing $c"; continue; }
  t0=$(date +%s); $X $H/env/bin/python analysis/bench/shotcost_noise.py -cp $c -o results/shotcost/e6_sc4_iqm --profiles iqm_garnet -n 300 --qmax $qmax
  echo "[job] $sp took $(( $(date +%s) - t0 )) s"
done
echo "[job] end=$(date -Is)"
