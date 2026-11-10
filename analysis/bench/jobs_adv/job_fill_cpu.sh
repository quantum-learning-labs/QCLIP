#!/bin/bash
# fill the empty 4- and 8-layer cells of Figure 7: e=3, n = 4, 6, 7, 8, 10 (16, 22, 25, 28, 34 qubits) -> untrained ckpts, structure,
# QPU gate counts (all readout circuits), best paths (greedy, random-greedy-128, cotengra 60 s with hard stop). CPU only.
#$ -S /bin/bash
#$ -N fill_cpu
#$ -l h_rt=16:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/7a2dad4
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=7a2dad4 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
B=""
for L in 4 8; do for n in 4 6 7 8 10; do D=$CK/hp_sc4i_nodiscard_e3_n${n}_tl${L}_qlr3e-3_s0
  [ -f $D/best.pt ] || $X analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n --layers $L --e 3 -o $D
  B="$B $D/best.pt"; done; done
$X analysis/bench/inf_engines.py -cp $B -o $O/diag_fill --pairs 40 --diag
$X analysis/bench/qpu_esp.py -cp $B -o $O/qpu_esp2_fill.csv --k 5 --circ all
$X analysis/bench/tw_bounds.py -cp $B -o $O/tw_bounds_fill.csv --k 2 --opts greedy,random-greedy-128,ctg --ctg_s 60 --ctg_reconf
echo "[job] end=$(date -Is)"
