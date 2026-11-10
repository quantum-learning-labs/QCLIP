#!/bin/bash
# cotengra (greedy trials, subtree reconfiguration, 600 s, hard stop 901 s) for every point from 40 qubits not already covered, 2 layers: same search budget at every size
#$ -S /bin/bash
#$ -N ctg3_a
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/e6aea71
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=e6aea71 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; done; }
$X analysis/bench/tw_bounds.py -cp $(ck e3_n12_tl2 e3_n15_tl2 e3_n20_tl2 e3_n25_tl2 e3_n32_tl2 e3_n40_tl2 e6_n15_tl2 e8_n20_tl2 e8_n32_tl2) -o $O/tw_bounds_ctg3_a.csv --k 1 --opts ctg --ctg_s 600 --ctg_reconf --ctg_methods greedy
echo "[job] end=$(date -Is)"
