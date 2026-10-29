#!/usr/bin/env bash
# Step-3 speed-ups, one change at a time on one GPU: profile_train (1 epoch, seed 0, --val_eval) of <cfg> with the code of
# each commit (worktrees qclip_wt/s3old = 9caeb47 baseline, qclip_wt/c3_<sha>), then a benchmark of <ckpt> with the
# first and last code. Results results/tn/step3/<tag>_*; never overwrites.
#   bash tools/tn/s3_chain.sh <tag> <cfg> <ckpt run> <sha>...
set -u
tag=$1 cfg=$2 ck=$3; shift 3
WT=/cs/research/pplv/comp_bridge/qclip_wt; TOOL=$WT/step3/tools/tn/profile_train.py; R=$WT/step3/results/tn/step3
PY="taskset -c 0-15 /cs/research/pplv/comp_bridge/.venv/bin/python"
export PYTHONPATH=. PYTHONUNBUFFERED=1 TN_PATH_WORKERS=8
codes=(s3old "$@")
for c in "${codes[@]}"; do
  d=$WT/$c; [ $c = s3old ] || d=$WT/c3_$c
  (cd $d && $PY $TOOL -cfg configs/tn/$cfg.yaml --seed 0 --steps 222 --val_eval -o $R/${tag}_prof_$c.json)
  echo "prof $c exit=$?"
done
first=$WT/s3old; last=$WT/c3_${codes[-1]}
for c in first last; do
  d=$([ $c = first ] && echo $first || echo $last)
  (cd $d && $PY scripts/benchmark.py -cp checkpoints/mscoco/tn/$ck/last.pt --set text.tn_engine=levelwise -o $R/${tag}_bench_$c.json)
  echo "bench $c exit=$?"
done
{ for c in "${codes[@]:1}"; do echo "== losses s3old vs $c"; $PY $WT/step3/tools/tn/regress.py losses $R/${tag}_prof_s3old.json $R/${tag}_prof_$c.json; done
  echo "== bench $ck first vs last"; $PY $WT/step3/tools/tn/regress.py bench $R/${tag}_bench_first.json $R/${tag}_bench_last.json; } > $R/${tag}_regress.txt 2>&1
cat $R/${tag}_regress.txt
