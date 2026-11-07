# Largest intermediate / flops of the trace-inside (doubled) network for the heaviest sentence structures at text.n = n, comparing opt_einsum greedy
# with a cotengra hyper-optimiser (minimising the largest intermediate).  usage: ... n_scale_ctg.py -cp <ckpt> --n 8 [--top 3] [--B 256] [--repeats 24]
import argparse, time, math
from pathlib import Path
import opt_einsum as oe, cotengra as ctg
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
from trace_check import recipes_of, doubled_str
ap = argparse.ArgumentParser(); ap.add_argument('-cp', required=True); ap.add_argument('--n', type=int, required=True)
ap.add_argument('--top', type=int, default=3); ap.add_argument('--B', type=int, default=256); ap.add_argument('--repeats', type=int, default=24); ap.add_argument('--methods', default='greedy,labels')
a = ap.parse_args()
config, _, _ = setup_exp(resolve_eval_config(None, Path(a.cp)), seed=0, overrides=[f'text.n={a.n}', 'text.layers=2'])
ansatz, im, tm, _ = build_experiment(config, 'cpu'); de = DataEngine(config, ansatz, 'cpu'); c = de.compile_text('val')
S = sorted({s for s, _ in recipes_of(c)}, key=lambda s: -len(s))[:a.top]
for s in S:
    d = doubled_str(s, tm.out_q); inp, out = d.split('->'); inputs = inp.split(',')
    size = {ch: 2 for ch in set(d) - set(',->')}; size['$'] = a.B
    t = time.time(); info = oe.contract_path(d, *[tuple(size[ch] for ch in x) for x in inputs], shapes=True, optimize='greedy')[1]
    print(f'n={a.n} ops={len(inputs)} greedy      largest={int(info.largest_intermediate):>14,d} flops={float(info.opt_cost):.2e} t={time.time() - t:.1f}s', flush=True)
    t = time.time(); opt = ctg.HyperOptimizer(minimize='size', max_repeats=a.repeats, methods=a.methods.split(','), parallel=False, progbar=False)
    tree = opt.search([tuple(x) for x in inputs], tuple(out), size)
    print(f'n={a.n} ops={len(inputs)} cotengra    largest={int(2 ** tree.contraction_width()):>14,d} flops={float(tree.contraction_cost()):.2e} t={time.time() - t:.1f}s', flush=True)
