#!/bin/bash
# QPU gate counts, all readout circuits, best of 5 Kingston transpile seeds per circuit: every checkpoint at 2 layers
#$ -S /bin/bash
#$ -N esp3_l2
#$ -l h_rt=20:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/e6aea71
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=e6aea71 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; done; }
$X analysis/bench/qpu_esp.py -cp $(ck e3_n5_tl2 e3_n6_tl2 e3_n7_tl2 e3_n8_tl2 e3_n9_tl2 e3_n10_tl2 e3_n12_tl2 e3_n15_tl2 e3_n20_tl2 e3_n25_tl2 e3_n32_tl2 e3_n40_tl2 e6_n15_tl2 e8_n20_tl2 e8_n32_tl2) $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt -o $O/qpu_esp3_l2.csv --k 3 --circ all --seeds 5
echo "[job] end=$(date -Is)"
