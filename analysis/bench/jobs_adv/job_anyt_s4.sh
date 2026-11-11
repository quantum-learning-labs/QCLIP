#!/bin/bash
# anytime path search, 4 layers, 16-49 qubits, 30 min trace per network
#$ -S /bin/bash
#$ -N anyt_s4
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"; grep -m1 'model name' /proc/cpuinfo
H=$HOME/qclip_hpc; cd $H/code/0bd46bf
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=1,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=0bd46bf $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
ck() { for x in "$@"; do if [ "$x" = e3_n4_tl2 ]; then printf '%s ' $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt; else printf '%s ' $CK/hp_sc4i_nodiscard_${x}_qlr3e-3_s0/best.pt; fi; done; }
$X analysis/bench/path_anytime.py -cp $(ck e3_n4_tl4 e3_n5_tl4 e3_n9_tl4 e3_n12_tl4 e3_n15_tl4) -o $O/anyt_s4.csv --budget 1800
echo "[job] end=$(date -Is)"
