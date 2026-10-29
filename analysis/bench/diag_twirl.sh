#!/bin/bash
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/diag_twirl.py results/bench_b2/ps_swap honest_miami 6 64 2>&1 | grep -vE "it/s]|s/it]|Transpiling"
