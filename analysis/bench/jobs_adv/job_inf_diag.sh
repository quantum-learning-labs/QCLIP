#!/bin/bash
# untrained init checkpoints for text.n = 10,12,15,20 (34..64 qubits), then structure statistics (live-wire peak of the lazy statevector,
# TN largest intermediate and flops) for n = 4..20: decides how far each engine can go. CPU only.
#$ -S /bin/bash
#$ -N inf_diag
#$ -l h_rt=12:00:00
#$ -l tmem=24G
#$ -l h_vmem=48G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/4cb7b24
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QCLIP_GIT_COMMIT=4cb7b24 $H/ubuntu2404"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
for n in 10 12 15 20; do
  D=$CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0
  [ -f $D/best.pt ] || $X $H/env/bin/python analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n -o $D
done
L="$CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt"; for n in 5 6 7 8 9 10 12 15 20; do L="$L $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
$X $H/env/bin/python analysis/bench/inf_engines.py -cp $L -o results/shotcost/inf_diag --pairs 40 --diag
echo "[job] end=$(date -Is)"
