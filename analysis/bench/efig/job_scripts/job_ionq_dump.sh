#!/bin/bash
# energy figure, like-for-like calibration: logical calibration circuits (CU, 100 std pairs) of the 27 PS models for the IonQ cloud simulator
#$ -S /bin/bash
#$ -N ef_ionq_dump
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/c73bdbd
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
for cfg in 3:1:10 3:3:13 2:1:10 3:2:10 4:1:10 5:1:10 6:1:11 7:1:12 8:1:12; do IFS=: read e n qmax <<<"$cfg"
  for s in 0 1 2; do r=hp_sc4_nodiscard_e${e}_n${n}_tl2_qlr3e-3_s$s
    [ -f results/shotcost/ionq_dump/$r.qpy ] || $X analysis/bench/shotcost_noise.py -cp $CK/$r/best.pt -o results/shotcost/ionq_dump --profiles none -n 100 --qmax $qmax --no_O --dump results/shotcost/ionq_dump/$r
  done; done
echo "[job] end=$(date -Is)"
