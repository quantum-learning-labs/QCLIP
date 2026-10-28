#!/bin/bash
# run.sh <checkpoint> <meas> <out> [extra args]: one backends.py run with the env the pool needs
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
exec /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/backends.py -cp "$1" --meas "$2" -o "$3" "${@:4}"
