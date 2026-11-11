#!/bin/bash
# smoke: path_anytime on the smallest network (2 layers, 16 qubits): 10 s trace, 60 s exact optimiser
#$ -S /bin/bash
#$ -N anyt_smoke
#$ -l h_rt=01:00:00
#$ -l tmem=16G
#$ -l h_vmem=64G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/0bd46bf
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=1,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=0bd46bf $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
$X analysis/bench/path_anytime.py -cp $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt -o $O/anyt_smoke.csv --budget 10 --opt_s 60 --mem_gb 24
echo "[job] end=$(date -Is)"
