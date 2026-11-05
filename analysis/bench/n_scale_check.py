# Feasibility of training/evaluating bigger text.n: contraction cost (largest intermediate, flops; batch 64) of the current vs doubled
# (trace-inside) path for the sentence structures of the val split, for several text.n.  Config comes from a discard checkpoint's config.yaml.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/n_scale_check.py -cp <ckpt> --ns 5 6 7 8 9 [-o out.json]
import argparse, json, time
from pathlib import Path
import pandas as pd
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
from trace_check import recipes_of, path_stats
ap = argparse.ArgumentParser(); ap.add_argument('-cp', required=True); ap.add_argument('--ns', type=int, nargs='+', required=True); ap.add_argument('-o', default=None)
a = ap.parse_args(); res = {}
for n in a.ns:
    t = time.time()
    config, _, _ = setup_exp(resolve_eval_config(None, Path(a.cp)), seed=0, overrides=[f'text.n={n}', 'text.layers=2'])
    ansatz, image_model, text_model, _ = build_experiment(config, 'cpu')
    de = DataEngine(config, ansatz, 'cpu'); c_val = de.compile_text('val')
    seen, st = set(), []
    for s, tl in recipes_of(c_val):
        if s in seen: continue
        seen.add(s); st.append(path_stats(s, tl, text_model.out_q))
    d = pd.DataFrame(st)
    r = dict(n=n, structures=len(d), nq_max=int(d.nq.max()), cur_largest_max=int(d.cur_largest.max()), dbl_largest_max=int(d.dbl_largest.max()),
             cur_flops_sum=float(d.cur_flops.sum()), dbl_flops_sum=float(d.dbl_flops.sum()), secs=time.time() - t)
    res[n] = r; print(json.dumps(r), flush=True)
if a.o: json.dump(res, open(a.o, 'w'), indent=1)
