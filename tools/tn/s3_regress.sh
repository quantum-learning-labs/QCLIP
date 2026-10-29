#!/usr/bin/env bash
# Step-3 speed-up check on one GPU: old code (worktree qclip_wt/s3old = 9caeb47) vs new code (this worktree), same seed.
# Training profile (1 epoch; losses must match), a timed val pass, benchmarks of fixed checkpoints (metrics must match
# except exact ties). Never overwrites: results/tn/step3/<tag>_*.
#   bash tools/tn/s3_regress.sh <tag> <cfg>... -- <checkpoint run>...
set -u
tag=$1; shift
cfgs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do cfgs+=("$1"); shift; done; shift
ckpts=("$@")
NEW=$PWD; OLD=/cs/research/pplv/comp_bridge/qclip_wt/s3old; R=$NEW/results/tn/step3
PY="taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
for c in "${cfgs[@]}"; do for code in old new; do
  dir=$([ $code = old ] && echo $OLD || echo $NEW)
  (cd $dir && $PY $NEW/tools/tn/profile_train.py -cfg configs/tn/$c.yaml --seed 0 --steps 222 --val_eval -o $R/${tag}_prof_${code}_$c.json)
  echo "prof $code $c exit=$?"
done; done
for k in "${ckpts[@]}"; do for code in old new; do
  dir=$([ $code = old ] && echo $OLD || echo $NEW)
  (cd $dir && $PY scripts/benchmark.py -cp checkpoints/mscoco/tn/$k/last.pt --set text.tn_engine=levelwise -o $R/${tag}_bench_${code}_$k.json)
  echo "bench $code $k exit=$?"
done; done
{
for c in "${cfgs[@]}"; do echo "== losses $c (old vs new)"; $PY tools/tn/regress.py losses $R/${tag}_prof_old_$c.json $R/${tag}_prof_new_$c.json; done
for k in "${ckpts[@]}"; do echo "== bench $k (old vs new)"; $PY tools/tn/regress.py bench $R/${tag}_bench_old_$k.json $R/${tag}_bench_new_$k.json; done
} > $R/${tag}_regress.txt 2>&1
cat $R/${tag}_regress.txt
