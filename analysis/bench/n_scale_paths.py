# For the heaviest sentence structures at text.n = n: largest intermediate (elements, batch B) of the trace-inside network under several path optimisers.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/n_scale_paths.py -cp <ckpt> --n 8 [--top 4] [--B 256]
import argparse, time
from pathlib import Path
import opt_einsum as oe
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
from trace_check import recipes_of, doubled_str
ap = argparse.ArgumentParser(); ap.add_argument('-cp', required=True); ap.add_argument('--n', type=int, required=True); ap.add_argument('--top', type=int, default=4); ap.add_argument('--B', type=int, default=256)
a = ap.parse_args()
config, _, _ = setup_exp(resolve_eval_config(None, Path(a.cp)), seed=0, overrides=[f'text.n={a.n}', 'text.layers=2'])
ansatz, im, tm, _ = build_experiment(config, 'cpu'); de = DataEngine(config, ansatz, 'cpu'); c = de.compile_text('val')
S = sorted({s for s, _ in recipes_of(c)}, key=lambda s: -len(s))[:a.top]
for s in S:
    inp, out = s.split('->'); shp = [(a.B, *[2] * len(t)) for t in inp.split(',')]; d = doubled_str(s, tm.out_q)
    for opt in ('greedy', 'random-greedy-64', 'dp', 'auto-hq'):
        t = time.time()
        try:
            info = oe.contract_path(d, *(shp * 2), shapes=True, optimize=opt)[1]
            print(f'nq={len(out)} ops={len(inp.split(","))} {opt:18s} largest={int(info.largest_intermediate):>14,d} flops={float(info.opt_cost):.2e} path_search={time.time() - t:.1f}s', flush=True)
        except Exception as e: print(opt, 'failed', str(e)[:80], flush=True)
