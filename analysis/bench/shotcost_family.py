# Family-level summary of shotcost_report.py's JSON: mean +- sd over seeds. usage: shotcost_family.py <report.json>
import json, sys, re, collections
import numpy as np
R = json.load(open(sys.argv[1]))
NAMES = {'hp_hp2_nodiscard_tl3_qlr3e-3': 'PS m4', 'hp_hp2_discard_tl3_pw0': 'plain m4', 'hp_do1_discard_tl3_pw0_initnd': 'warm m4',
         'hp_kd1_discard_kd_ws': 'distill m4', 'hp_xo1_nodiscard_n2_tl3_qlr0.003': 'PS m8', 'hp_xo1_discard_n2_tl3_pw0': 'plain m8',
         'hp_xo1_discard_n2_tl3_pw0_ws': 'warm m8'}
ORDER = list(NAMES.values())
ms = lambda v: f'{np.mean(v):7.3f}+-{np.std(v, ddof=1) if len(v) > 1 else 0:.3f}'
def fmt_shots(v, n):
    v = [x for x in v if x is not None]
    if len(v) < n: return f"{'-' if not v else f'{np.median(v):.0f}':>7s}({len(v)}/{n})"
    return f'{np.median(v):7.0f}      '
fam = collections.defaultdict(list)
for k, r in R['per_ckpt'].items():
    fam[(NAMES[r['family']], r['set'])].append(r)
for s in ('std', 'swap_full'):
    for ro in ('compute_uncompute', 'destructive_swap'):
        print(f'\n== {s} / {ro}: exact acc (mean+-sd), acceptance, median over seeds of shots per decision to reach acc (n seeds reaching it)')
        print(f"{'model':11s} seeds {'exact acc':>15s} accept | " + ' '.join(f'{t:>13s}' for t in ('0.70', '0.75', '0.80', '0.85')))
        for name in ORDER:
            rs = fam.get((name, s), [])
            if not rs: continue
            print(f"{name:11s} {len(rs):5d} {ms([r['exact_acc'] for r in rs])} {np.mean([r['accept_mean'] for r in rs]):6.3f} | "
                  + ' '.join(f'{fmt_shots([r[ro]["shots_per_decision"][t] for r in rs], len(rs)):>13s}' for t in ('0.7', '0.75', '0.8', '0.85')))
pf = collections.defaultdict(list)
for k, o in R['paired'].items():
    ck, s = k.split('__'); pf[(NAMES[re.sub(r'_s\d+$', '', ck)], s)].append(o)
print('\n== paired discard / PS (same seed, pairs both rank correctly), geo-mean over pairs, then mean+-sd over seeds')
print(f"{'model':11s} {'set':9s} {'readout':7s} n_seeds  cost ratio        acceptance x  contrast        x variance       beta")
for (name, s), os_ in sorted(pf.items(), key=lambda x: (x[0][1], ORDER.index(x[0][0]))):
    for ro in ('compute_uncompute', 'destructive_swap'):
        g = lambda f: [f(o[ro]) for o in os_]
        print(f"{name:11s} {s:9s} {ro[:7]:7s} {len(os_):5d}  {ms(g(lambda x: x['cost_ratio_geo']))}  {ms(g(lambda x: x['terms_factor']['acceptance']))} "
              f"{ms(g(lambda x: x['terms_factor']['contrast']))}  {ms(g(lambda x: x['terms_factor']['variance']))}  {ms(g(lambda x: x['beta']))}")
print('\n== by k (CU, std and swap_full pooled per family): n pairs, PS acceptance, D_disc/D_ps, cost ratio (geo, mean over seeds)')
bk = collections.defaultdict(lambda: collections.defaultdict(list))
for (name, s), os_ in pf.items():
    for o in os_:
        for kv, b in o['by_k'].items():
            bk[(name, s)][int(kv)].append(b)
for (name, s), d in sorted(bk.items(), key=lambda x: (x[0][1], ORDER.index(x[0][0]))):
    for kv in sorted(d):
        b = d[kv]
        print(f"{name:11s} {s:9s} k={kv}: n~{np.mean([x['n'] for x in b]):6.0f}  a_ps {np.mean([x['a_ps_geo'] for x in b]):.4f}"
              f"  D ratio {np.mean([x['D_ratio_geo'] for x in b]):.3f}  cost ratio {np.mean([x['cost_ratio_cu_geo'] for x in b]):6.2f}")
