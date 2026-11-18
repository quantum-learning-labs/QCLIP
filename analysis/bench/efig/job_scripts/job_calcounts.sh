#!/bin/bash
# energy figure: Kingston (best of 5) and IQM (Aphrodite, best of 5) two-qubit counts of the 54 trained 6-13 q circuits behind the noise
# calibration (noise_calib.csv), so ln F = a - eps n2q can be refitted and bootstrapped per backend. CPU only.
#$ -S /bin/bash
#$ -N ef_calcounts
#$ -l h_rt=12:00:00
#$ -l tmem=16G
#$ -l h_vmem=64G
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/136a567
X="singularity exec -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,QISKIT_PARALLEL=FALSE,QCLIP_GIT_COMMIT=136a567 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv
B=""
for v in e2_n1 e3_n1 e3_n2 e3_n3 e4_n1 e5_n1 e6_n1 e7_n1 e8_n1; do for s in 0 1 2; do
  for r in hp_sc4_nodiscard_${v}_tl2_qlr3e-3_s$s hp_sc4_discard_${v}_tl2_pw0_ws_s$s; do [ -f $CK/$r/best.pt ] && B="$B $CK/$r/best.pt" || echo "missing $r"; done
done; done
$X analysis/bench/qpu_esp.py -cp $B -o $O/qpu_esp_calib.csv --k 5 --circ all --seeds 5
$X analysis/bench/qpu_iqm.py -cp $B -o $O/qpu_iqm_calib.csv --k 5 --seeds 5
echo "[job] end=$(date -Is)"
