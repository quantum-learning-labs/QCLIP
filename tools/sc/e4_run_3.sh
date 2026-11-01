export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec taskset -c 12-15 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_branches.py -o results/shotcost/e4 --sets val,std -cp $(cat tools/sc/shard_03)
