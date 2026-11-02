export PYTHONPATH=.:analysis/bench PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec taskset -c 8-11 /cs/research/pplv/comp_bridge/.venv/bin/python analysis/bench/shotcost_branches.py -o results/shotcost/e4 --e0_out results/shotcost/e0_sc2 --sets val,std,swap_full -cp $(cat tools/sc/sc2shard_02)
