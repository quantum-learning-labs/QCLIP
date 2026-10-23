#!/usr/bin/env bash
# Step-2 check of one code change, on one GPU: training profile (same seed/steps as the baseline), benchmark of the
# step-1 3090 Ti reference checkpoint, then regression against the baseline outputs. Never overwrites results.
#   bash tools/tn/check_change.sh <tag> [steps=300] [--set k=v ...]   (overrides go to both the profile and the benchmark)
#   -> results/tn/step2/{prof_<tag>.json, bench_<tag>.json, regress_<tag>.txt}
set -u
tag=$1; steps=${2:-300}; shift; shift || true
R=results/tn/step2
# pinned to the P-cores (0-15) of the pool's i9-12900K: the run is CPU-bound and E-cores are ~25% slower
PY="taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1
$PY tools/tn/profile_train.py -cfg configs/tn/ref_clip.yaml --seed 0 --steps $steps -o $R/prof_$tag.json "$@"
echo "profile exit=$?"
$PY scripts/benchmark.py -cp checkpoints/mscoco/tn/tn_s1_clip_s0/last.pt -o $R/bench_$tag.json "$@"
echo "benchmark exit=$?"
{
  echo "== losses vs baseline (prof_base.json)"; $PY tools/tn/regress.py losses $R/prof_base.json $R/prof_$tag.json
  echo "== benchmark of tn_s1_clip_s0/last.pt vs bench_ref_s0.json"; $PY tools/tn/regress.py bench $R/bench_ref_s0.json $R/bench_$tag.json
} > $R/regress_$tag.txt 2>&1
cat $R/regress_$tag.txt
