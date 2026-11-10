#!/bin/bash
# energy-scaling grid, 8 layers: grow the output register e and the qubits per wire n together (3:25 3:32 3:40 6:15 8:20 8:32) -> untrained ckpts,
# structure, Kingston/IonQ statistics, best contraction paths (greedy + random-greedy-128). CPU only (no simulation here).
#$ -S /bin/bash
#$ -N grid_l8
#$ -l h_rt=24:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/d8c1b80
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=d8c1b80 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
B=""
for en in 3:25 3:32 3:40 6:15 8:20 8:32; do e=${en%:*}; n=${en#*:}; D=$CK/hp_sc4i_nodiscard_e${e}_n${n}_tl8_qlr3e-3_s0
  [ -f $D/best.pt ] || $X analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n --layers 8 --e $e -o $D
  B="$B $D/best.pt"; done
O=results/shotcost/adv
$X analysis/bench/inf_engines.py -cp $B -o $O/diag_grid_l8 --pairs 40 --diag
$X analysis/bench/qpu_esp.py -cp $B -o $O/qpu_esp_grid_l8.csv --k 5
$X analysis/bench/tw_bounds.py -cp $B -o $O/tw_bounds_grid_l8.csv --k 2 --opts greedy,random-greedy-128
echo "[job] end=$(date -Is)"
