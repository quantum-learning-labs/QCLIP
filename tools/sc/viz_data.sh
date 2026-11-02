export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec taskset -c 0-3 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/viz_data.py results/shotcost results/shotcost/viz/viz_data.json
