#!/bin/bash
# IonQ cloud SIMULATOR (forte-1 noise) for every dumped PS calibration model; skips ones already done
cd /cs/research/pplv/comp_bridge/qclip_wt/energyfig
for q in results/shotcost/ionq_dump/*.qpy; do N=$(basename $q .qpy); O=results/shotcost/ionq_calib/${N}__ionq_forte-1.npz
  [ -f $O ] && continue
  echo "== $N $(date -u +%H:%M:%S)"
  PYTHONPATH=.:analysis/bench /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/ionq_calib.py results/shotcost/ionq_dump/$N $O --shots 4096 --batch 50 || echo "FAILED $N"
done
echo "ALL DONE $(date -u)"
