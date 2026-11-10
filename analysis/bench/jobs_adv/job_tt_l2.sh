#!/bin/bash
# One-off compile cost on the QPU side: wall seconds of one Kingston (and one IonQ-basis) transpile per circuit, every checkpoint at 2 layers
# (1 seed, first typical decision, CU and O circuits). Counts are not used from this file, only ibm_transpile_s / ionq_transpile_s.
#$ -S /bin/bash
#$ -N tt_l2
#$ -l h_rt=10:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"; grep -m1 'model name' /proc/cpuinfo
H=$HOME/qclip_hpc; cd $H/code/3de11a1
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=3de11a1 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; done; }
$X analysis/bench/qpu_esp.py -cp $H/code/39974f7/checkpoints/svo-probes/vqc/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt $(ck e3_n5_tl2 e3_n6_tl2 e3_n7_tl2 e3_n8_tl2 e3_n9_tl2 e3_n10_tl2 e3_n12_tl2 e3_n15_tl2 e3_n20_tl2 e3_n25_tl2 e3_n32_tl2 e3_n40_tl2 e6_n15_tl2 e8_n20_tl2 e8_n32_tl2) -o $O/qpu_tt_l2.csv --k 1 --circ all --seeds 1
echo "[job] end=$(date -Is)"
