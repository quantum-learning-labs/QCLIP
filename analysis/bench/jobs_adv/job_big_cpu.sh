#!/bin/bash
# big points: untrained 2-layer ckpts n=25,32,40 (79,100,124 qubits) + the 8-layer 64-qubit point recompiled with the get_char fix (04d6e92):
# structure, Kingston/IonQ statistics, best contraction paths (greedy + random-greedy-128). CPU only.
#$ -S /bin/bash
#$ -N big_cpu
#$ -l h_rt=24:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/04d6e92
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=04d6e92 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
for n in 25 32 40; do D=$CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0; [ -f $D/best.pt ] || $X analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n --layers 2 -o $D; done
B="$CK/hp_sc4i_nodiscard_e3_n20_tl8_qlr3e-3_s0/best.pt"; for n in 25 32 40; do B="$B $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
O=results/shotcost/adv
$X analysis/bench/inf_engines.py -cp $B -o $O/diag_big --pairs 40 --diag
$X analysis/bench/qpu_esp.py -cp $B -o $O/qpu_esp_big.csv --k 5
$X analysis/bench/tw_bounds.py -cp $B -o $O/tw_bounds_big.csv --k 3 --opts greedy,random-greedy-128
echo "[job] end=$(date -Is)"
