#!/bin/bash
# run_ionq.sh <bench dir> [ionq_run.py args]: IonQ cloud SIMULATOR stage (CPU only, needs network)
cd /cs/research/pplv/comp_bridge/qclip_wt/bench
export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1
exec /cs/research/pplv/comp_bridge/.venv/bin/python -W ignore analysis/bench/ionq_run.py "$@"
