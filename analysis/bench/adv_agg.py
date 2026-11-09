# Advantage study aggregator (pure Python): CSVs from the cluster -> adv.json for the page.
# Inputs (this directory): gpu_engines.csv + structure.csv (E1, L4), tw_bounds_l2.csv / tw_bounds_l48.csv (E3), qpu_esp_l2.csv / qpu_esp_l48.csv (E5),
# mps_a.csv / mps_b.csv (E4), noise_calib.csv (E5 calibration).
# Output: adv.json with one row per (layers, nq): QPU structure (IBM n2q, duration; IonQ n1q/n2q), classical measurements and best-known path costs.
# The energy model itself (shots, fidelity, frontier) is evaluated in the page so its assumptions can be changed interactively.
import csv, json, math, collections, os, re

H = os.path.dirname(os.path.abspath(__file__))
rd = lambda f: list(csv.DictReader(open(os.path.join(H, f)))) if os.path.exists(os.path.join(H, f)) else []
fl = lambda x: float(x) if x not in ('', 'nan', None) else None
mean = lambda v: sum(v) / len(v) if v else None
layers_of = lambda ck: int(re.search(r'_tl(\d+)_', ck).group(1))
BAD = {'hp_sc4i_nodiscard_e3_n20_tl8_qlr3e-3_s0'}       # 1722/1833 sentences failed to compile: typical width biased (43 not 64 qubits)

rows = collections.defaultdict(dict)
def R(ck, nq): return rows[(layers_of(ck), int(float(nq)))]

for r in rd('structure.csv'):                            # e1 structure (greedy) incl. tn widths
    if r['ckpt'] in BAD or r['ckpt'] == 'ckpt': continue
    x = R(r['ckpt'], r['nq']); x.update(ckpt=r['ckpt'], n2q_logical=fl(r['n2q_mean']), greedy_ps_width=fl(r['tn_width']), greedy_ps_lf=fl(r['tn_log10_flops']),
                                         greedy_rho_width=fl(r['tn_rho_width']), greedy_rho_lf=fl(r['tn_rho_log10_flops']))
for r in rd('gpu_engines.csv'):
    if r['ckpt'] in BAD or r['ckpt'] == 'ckpt' or not fl(r['J_gross_per_decision']): continue
    x = R(r['ckpt'], r['nq']); x.setdefault('gpu', {})[r['engine']] = dict(J=fl(r['J_gross_per_decision']), s=fl(r['s_per_decision']), err=fl(r['max_abs_err']), gpu=r['gpu'], note=r['note'])
best = collections.defaultdict(lambda: collections.defaultdict(list))
for f in ('tw_bounds_l2.csv', 'tw_bounds_l48_fromlog.csv', 'tw_bounds_l8fast.csv'):
    for r in rd(f):
        if r['ckpt'] in BAD or r['width'] in ('nan', ''): continue
        best[(layers_of(r['ckpt']), int(r['nq']), r['net'], int(r['idx']))][r['opt']].append((fl(r['width']), fl(r['log10_flops']), fl(r['tw_lb'])))
per = collections.defaultdict(list)
for (L, nq, net, i), d in best.items():
    cand = [v[0] for v in d.values()]
    w, lf, lb = min(cand, key=lambda t: t[1])           # best (min flops) path over optimizers
    per[(L, nq, net)].append((w, lf, lb, d.get('greedy', [(None, None, None)])[0][1]))
for (L, nq, net), v in per.items():
    x = rows[(L, nq)]; x[f'best_{net}_width'] = mean([t[0] for t in v]); x[f'best_{net}_lf'] = mean([t[1] for t in v]); x[f'tw_lb_{net}'] = max(t[2] for t in v)
    g = [t[3] for t in v if t[3] is not None]; x[f'greedy_{net}_lf_tw'] = mean(g)
for f in ('qpu_esp_l2b.csv', 'qpu_esp_l48.csv'):
    acc = collections.defaultdict(list)
    for r in rd(f):
        if r['ckpt'] in BAD: continue
        acc[(r['ckpt'], r['nq'])].append(r)
    for (ck, nq), v in acc.items():
        x = R(ck, nq)
        for k in ('ibm_duration_s', 'ibm_n2q', 'ibm_n1q', 'ionq_n1q', 'ionq_n2q'): x[k] = mean([fl(t[k]) for t in v])
for f in ('mps_a.csv', 'mps_b.csv'):
    for r in rd(f):
        if r['ckpt'] in BAD or r['ckpt'] == 'ckpt' or not fl(r['chi']): continue
        x = R(r['ckpt'], r['nq'])
        x.setdefault('mps', {}).setdefault(r['net'], []).append(dict(chi=int(r['chi']), J=fl(r['J_gross_per_decision']), s=fl(r['s_per_decision']), dw=fl(r['dw_mean']),
                                                                     mrel=fl(r['margin_rel_err_median']), agree=fl(r['agree']), amp=fl(r['amp_rel_err_mean']), ref=r['reference']))
cal = rd('noise_calib.csv')
out = dict(rows=[dict(layers=L, nq=nq, **v) for (L, nq), v in sorted(rows.items())],
           noise_calib=[dict(e=int(r['e']), n=int(r['n']), method=r['method'], seed=int(r['seed']), nq=fl(r['nq_mean']), F=fl(r['F_eff'])) for r in cal],
           noise_fit=dict(a=-0.187, b=-0.00331, note='ln F_eff = a + b * n2q_native (Kingston CU readout, 54 noisy runs, 6-13 qubits)'))
json.dump(out, open(os.path.join(H, 'adv.json'), 'w'), indent=1)
for r in out['rows']:
    print(r['layers'], r['nq'], {k: (round(v, 2) if isinstance(v, float) else '…') for k, v in r.items() if k in ('ibm_n2q', 'best_ps_lf', 'best_rho_lf', 'greedy_ps_lf', 'greedy_rho_lf')},
          {e: round(g['J'], 3) for e, g in r.get('gpu', {}).items()}, {n: len(v) for n, v in r.get('mps', {}).items()})
