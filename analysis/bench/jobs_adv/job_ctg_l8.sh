#!/bin/bash
# cotengra (greedy+labels, subtree reconfiguration, 600 s per network) best paths, l8 points where random-greedy-128 is the only search
#$ -S /bin/bash
#$ -N ctg_l8
#$ -l h_rt=14:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/669a3d4
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=669a3d4 $H/ubuntu2404 $H/env/bin/python"
CK=checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; done; }
$X analysis/bench/tw_bounds.py -cp $(ck e3_n5_tl2) -o $O/ctg_smoke.csv --k 1 --opts ctg --ctg_s 10 --ctg_reconf --nets ps || exit 1
$X analysis/bench/tw_bounds.py -cp $(ck e3_n20_tl8 e3_n25_tl8 e3_n32_tl8 e3_n40_tl8 e6_n15_tl8 e8_n20_tl8 e8_n32_tl8) -o $O/tw_bounds_ctg_l8.csv --k 1 --opts ctg --ctg_s 600 --ctg_reconf
echo "[job] end=$(date -Is)"
