#!/bin/bash
# run_twirl.sh <cores> <set>...: twirl-limit debiasing for each set once its bench_b2 rerun is complete (waits for it)
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
c=$1; shift
for s in "$@"; do
  d=results/bench_b2/$s
  until [ -e $d/raw.pkl ]; do sleep 60; done
  echo "##### $s $(date -Is)"
  taskset -c $c /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/twirl_limit.py $d --threads 8 2>&1 | grep -E "_twirl acc|done|Error|Traceback"
done
echo "##### all done $(date -Is)"
