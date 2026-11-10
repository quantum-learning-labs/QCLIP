#!/bin/bash
#$ -S /bin/bash
#$ -N e5_esp_l2b
#$ -l h_rt=12:00:00
#$ -l tmem=24G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/7b5fde8
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=7b5fde8 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
L2="$CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt"; for n in 5 6 7 8 9 10 12 15 20; do L2="$L2 $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
L48=""; for l in 4 8; do for n in 5 9 12 15 20; do L48="$L48 $CK/hp_sc4i_nodiscard_e3_n${n}_tl${l}_qlr3e-3_s0/best.pt"; done; done
B="$CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt";for n in 5 6 7 8 9 10 12 15 20; do B="$B $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
$X analysis/bench/qpu_esp.py -cp $B -o results/shotcost/adv/qpu_esp_l2b.csv --k 5
echo "[job] end=$(date -Is)"
