#!/bin/bash
# GPU energy (L4) for the fill checkpoints of job_fill_cpu.sh: cuStateVec (state and density matrix) and tensor network (both networks).
#$ -S /bin/bash
#$ -N fill_gpu
#$ -l h_rt=12:00:00
#$ -l tmem=32G
#$ -l h_vmem=128G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=l4
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/7a2dad4
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=7a2dad4 $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc; O=results/shotcost/adv/e1_fill
B=""; for L in 4 8; do for n in 4 6 7 8 10; do B="$B $CK/hp_sc4i_nodiscard_e3_n${n}_tl${L}_qlr3e-3_s0/best.pt"; done; done
$X analysis/bench/inf_engines.py -cp $B -o $O --pairs 40 --engines cusv,cusv_rho,tn,tn_rho --tn_width_max 29
echo "[job] end=$(date -Is)"
