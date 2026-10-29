#!/bin/bash
# IonQ forte-1 simulator, client-side debiasing (4 variants), every benchmark set in turn
for s in ps_swap ps_cu discard_swap discard_cu discard6_swap discard6_cu; do
  echo "== $s"; bash analysis/bench/run_ionq.sh results/bench_b1/$s --noise forte-1 --reps 5 --shots 8192 --debias 4 2>&1 | grep -E "done|ideal|Traceback|Error"
done
