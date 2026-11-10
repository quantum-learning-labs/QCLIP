#!/bin/bash
# smoke: inf_engines.py (aer baseline, cuStateVec, lazy statevector, tensor network) on n=4,5 with 8 decisions; checks vs CPU reference
#$ -S /bin/bash
#$ -N inf_eng_smoke
#$ -l h_rt=1:00:00
#$ -l tmem=32G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=a6000
#$ -l h=!hogthrob-206-2*
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/4cb7b24
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=4cb7b24 $H/ubuntu2404"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
$X $H/env/bin/python analysis/bench/inf_engines.py -cp $CK/hp_sc4_nodiscard_e3_n4_tl2_qlr3e-3_s0/best.pt $CK/hp_sc4i_nodiscard_e3_n5_tl2_qlr3e-3_s0/best.pt \
   -o results/shotcost/inf_eng_smoke --pairs 8 --min_s 3
echo "[job] end=$(date -Is)"
