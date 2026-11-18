#!/bin/bash
# energy figure (log depth L = round(log2 qubits)): 5 layers, e=3, text.n in {7 8 9 10 12}: untrained ckpts, structure, Kingston/IonQ (best of 5),
# IQM (Aphrodite, best of 5), contraction paths (greedy + random-greedy-128, then cotengra 600 s greedy+reconf with hard stop). CPU only.
#$ -S /bin/bash
#$ -N ef_cpu_l5
#$ -l h_rt=24:00:00
#$ -l tmem=32G
#$ -l h_vmem=128G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/136a567
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=136a567 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv; mkdir -p $O
B=""
for n in 7 8 9 10 12; do D=$CK/hp_sc4i_nodiscard_e3_n${n}_tl5_qlr3e-3_s0
  [ -f $D/best.pt ] || $X analysis/bench/make_init_ckpt.py -cfg $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/config.yaml --n $n --layers 5 --e 3 -o $D
  B="$B $D/best.pt"; done
$X analysis/bench/inf_engines.py -cp $B -o $O/diag_ef_l5 --pairs 40 --diag
$X analysis/bench/qpu_esp.py -cp $B -o $O/qpu_esp_ef_l5.csv --k 3 --circ all --seeds 5
$X analysis/bench/qpu_iqm.py -cp $B -o $O/qpu_iqm_ef_l5.csv --k 3 --seeds 5
$X analysis/bench/tw_bounds.py -cp $B -o $O/tw_bounds_ef_l5.csv --k 2 --opts greedy,random-greedy-128
$X analysis/bench/tw_bounds.py -cp $B -o $O/tw_bounds_ef_ctg_l5.csv --k 1 --opts ctg --ctg_s 600 --ctg_reconf --ctg_methods greedy
echo "[job] end=$(date -Is)"
