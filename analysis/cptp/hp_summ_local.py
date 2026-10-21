# Stage summary with the stdlib only (runs on the MSI). usage: hp_summ_local.py <res dir>/<tag> <base dir> [seeds]
import json, glob, os, sys, sqlite3, statistics as st, collections
res, base = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
rows = collections.defaultdict(dict)
for d in glob.glob(f'{res}/*/s*/'):
    try: v = json.load(open(d + 'val.json')); b = json.load(open(d + 'benchmark_best.json'))
    except Exception: continue
    r = b['results']; rows[d.split('/')[-3]][int(d.split('/')[-2][1:])] = (v['val_metrics']['hard_neg_acc'], r['std_hard_neg_acc'], r['swap_hard_neg_acc_norm'])
db = sqlite3.connect(f'file:{base}/mlf_dbs/svo-probes_discard_baseline.db?mode=ro', uri=True)
for m in ('nodiscard', 'discard'):
    for s in seeds:
        f = f'{base}/results/discard_baseline/base/base_{m}_s{s}/benchmark_best.json'
        if not os.path.exists(f): continue
        b = json.load(open(f)); r = b['results']
        val = db.execute("select m.value from metrics m join runs r on r.run_uuid=m.run_uuid where r.name=? and r.status='FINISHED' and m.key='hard_neg_acc' and m.step=?", (f'base_{m}_s{s}', b['epoch'])).fetchone()[0]
        rows[f'{m}_BASE'][s] = (val, r['std_hard_neg_acc'], r['swap_hard_neg_acc_norm'])
def f(xs): return f'{st.mean(xs):.3f}±{(st.stdev(xs) if len(xs) > 1 else 0):.3f}'
print(f'{"method_variant":22s} n  {"val (select)":>12s}  {"test std":>11s}  {"swap_norm":>11s}  best std  best swap')
for m in ('nodiscard', 'discard'):
    ks = [k for k in rows if k.split('_')[0] == m]
    ks.sort(key=lambda k: -st.mean([rows[k][s][0] for s in seeds if s in rows[k]] or [0]))
    for k in ks:
        xs = [rows[k][s] for s in seeds if s in rows[k]]
        if not xs: continue
        c = list(zip(*xs))
        print(f'{k:22s} {len(xs)}  {f(c[0])}  {f(c[1])}  {f(c[2])}  {max(c[1]):.3f}     {max(c[2]):.3f}')
    print()
