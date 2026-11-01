export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=5
exec taskset -c 5-9 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_branches.py -o results/shotcost/e4 --e0_out results/shotcost/e0_n3 --sets val,std,swap_full -cp $(cat tools/sc/n3shard_01)
