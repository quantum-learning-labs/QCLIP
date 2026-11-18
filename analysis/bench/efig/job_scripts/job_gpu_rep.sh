#!/bin/bash
# energy figure: run-to-run spread of the measured GPU energies. Repeats (4 runs, fresh process each) on an L4 like the originals:
# statevector engines (cuStateVec, lazy torch) + exact TN (ps, rho) at 22/28/31 q (2 layers), MPS (chi 4, 16) at 31 q. Small outputs.
#$ -S /bin/bash
#$ -N ef_gpu_rep
#$ -l h_rt=06:00:00
#$ -l tmem=32G
#$ -l h_vmem=96G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l gpu_type=l4
#$ -j y
set -uo pipefail
echo "[job] $JOB_NAME $JOB_ID on $(hostname -s) start=$(date -Is)"
H=$HOME/qclip_hpc; cd $H/code/698a2be; mkdir -p $HOME/.nv_jit_cache
SP=$H/env/lib/python3.12/site-packages
PRE=$SP/nvidia/nvjitlink/lib/libnvJitLink.so.12:$SP/nvidia/cublas/lib/libcublasLt.so.12:$SP/nvidia/cublas/lib/libcublas.so.12:$SP/nvidia/cusparse/lib/libcusparse.so.12:$H/libs_cusolver12/nvidia/cusolver/lib/libcusolver.so.11
X="singularity exec --nv -B $HOME --env PYTHONPATH=.:analysis/bench,OMP_NUM_THREADS=4,PYTHONUNBUFFERED=1,CUDA_CACHE_PATH=$HOME/.nv_jit_cache,QCLIP_GIT_COMMIT=698a2be,LD_PRELOAD=$PRE $H/ubuntu2404 $H/env/bin/python"
CK=$H/code/39974f7/checkpoints/svo-probes/vqc
L2=""; for n in 6 8 9; do L2="$L2 $CK/hp_sc4i_nodiscard_e3_n${n}_tl2_qlr3e-3_s0/best.pt"; done
for rep in 1 2 3 4; do
  $X analysis/bench/inf_engines.py -cp $L2 -o results/shotcost/adv/ef_rep/r$rep --pairs 40 --engines cusv,lazy,tn,tn_rho --live_max 28 --tn_width_max 29
  $X analysis/bench/mps_engine.py -cp $CK/hp_sc4i_nodiscard_e3_n9_tl2_qlr3e-3_s0/best.pt -o results/shotcost/adv/ef_rep/mps_r$rep --pairs 10 --chis 4,16 --min_s 5 --max_s_per_dec 120
done
echo "[job] end=$(date -Is)"
