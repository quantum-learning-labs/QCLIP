"""Step-4 trainability summary of TN runs (steps.csv, trainstats.csv, paramstats.csv, mlflow-free).
Per run: epochs done, first non-finite epoch, loss spikes (step loss > rolling median of the previous 200 steps + 5 MAD),
grad-norm p99/max for type tensors and words (after clipping; pre-clip total if clipped), pre-LayerNorm norm min/max,
type-tensor norm ratio to init (median/min) at the last epoch, and the max sigma-free indicator of collapse.
  s4_trainstats.py [--json out.json] <run_dir>...
"""
import csv, json, math, sys
import numpy as np

def rows(path):
    try: return list(csv.DictReader(open(path)))
    except FileNotFoundError: return []

def summary(d):
    out = {'run': d.rstrip('/').split('/')[-1]}
    st = rows(f'{d}/steps.csv')
    loss = np.array([float(r['loss']) for r in st]); ep = np.array([int(r['epoch']) for r in st])
    out['epochs'] = int(ep.max()) + 1 if len(ep) else 0
    bad = ~np.isfinite(loss)
    out['nan_epoch'] = int(ep[bad][0]) if bad.any() else None
    spikes = 0
    fin = loss[~bad]
    for i in range(200, len(fin)):
        w = fin[i - 200:i]; med = np.median(w); mad = np.median(np.abs(w - med)) + 1e-12
        spikes += fin[i] > med + 5 * 1.4826 * mad
    out['loss_spikes'] = int(spikes)
    ts = rows(f'{d}/trainstats.csv')
    if ts:
        def col(c): return np.array([float(r[c]) for r in ts if c in r and r[c] not in ('', 'nan')], dtype=float)
        for c in ('g_type', 'g_word'):
            x = col(c); x = x[np.isfinite(x)]
            if len(x): out[f'{c}_p99'], out[f'{c}_max'] = float(np.percentile(x, 99)), float(x.max())
        x = col('g_preclip') if 'g_preclip' in ts[0] else np.array([])
        x = x[np.isfinite(x)] if len(x) else x
        if len(x): out['g_preclip_p99'], out['g_preclip_max'] = float(np.percentile(x, 99)), float(x.max())
        lo, hi = col('pre_min'), col('pre_max')
        if len(lo): out['pre_min'], out['pre_max'] = float(np.nanmin(lo)), float(np.nanmax(hi[np.isfinite(hi)])) if np.isfinite(hi).any() else float('inf')
    ps = rows(f'{d}/paramstats.csv')
    if ps:
        last = ps[-1]
        for k in ('type_rel_med', 'type_rel_min', 'type_rel_max', 'word_rel_med'):
            if k in last: out[k] = float(last[k])
    try:
        meta = json.load(open(f'{d}/run_meta.json'))
        out['samples_per_s'] = float(np.median([e['train_samples_per_s'] for e in meta['epochs']]))
        out['peak_mem_gb'] = max(e['peak_mem_gb'] or 0 for e in meta['epochs'])
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return out

if __name__ == '__main__':
    args = sys.argv[1:]; js = None
    if args and args[0] == '--json': js, args = args[1], args[2:]
    res = [summary(d) for d in args]
    keys = ['run', 'epochs', 'nan_epoch', 'loss_spikes', 'g_type_p99', 'g_type_max', 'g_word_p99', 'g_word_max', 'g_preclip_max',
            'pre_min', 'pre_max', 'type_rel_med', 'type_rel_min', 'samples_per_s']
    print(' | '.join(keys))
    for r in res:
        print(' | '.join(f"{r[k]:.3g}" if isinstance(r.get(k), float) else str(r.get(k, '-')) for k in keys))
    if js: json.dump(res, open(js, 'w'), indent=1)
