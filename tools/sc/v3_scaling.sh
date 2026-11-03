export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
exec taskset -c 0-7 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/v3_energy.py --scaling_only -o results/shotcost/viz -cp none
