#!/bin/bash
#$ -S /bin/bash
#$ -N e2_ckpt
#$ -l h_rt=12:00:00
#$ -l tmem=24G
#$ -l h_vmem=48G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/7b5fde8
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QCLIP_GIT_COMMIT=7b5fde8 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
L2="$CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt"; for n in 5 6 7 8 9 10 12 15 20; do L2="$L2 $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
L48=""; for l in 4 8; do for n in 5 9 12 15 20; do L48="$L48 $CK/hp_sc4i_nodiscard_e3_n${n}_tl${l}_qlr3e-3_s0/best.pt"; done; done
for l in 4 8; do for n in 5 9 12 15 20; do D=$CK/hp_sc4i_nodiscard_e3_n${n}_tl${l}_qlr3e-3_s0; [ -f $D/best.pt ] || $X analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n --layers $l -o $D; done; done
$X analysis/bench/inf_engines.py -cp $L48 -o results/shotcost/inf_diag_layers --pairs 40 --diag
echo "[job] end=$(date -Is)"
