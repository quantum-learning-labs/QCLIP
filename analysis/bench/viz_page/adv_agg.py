# Advantage study aggregator (pure Python): CSVs from the cluster -> adv.json for the page.
# Inputs (this directory): gpu_engines.csv + structure.csv (E1, L4), tw_bounds_l2.csv / tw_bounds_l48.csv (E3), qpu_esp_l2.csv / qpu_esp_l48.csv (E5),
# mps_a.csv / mps_b.csv (E4), noise_calib.csv (E5 calibration).
# Output: adv.json with one row per (layers, nq): QPU structure (IBM n2q, duration; IonQ n1q/n2q), classical measurements and best-known path costs.
# The energy model itself (shots, fidelity, frontier) is evaluated in the page so its assumptions can be changed interactively.
import csv, json, math, collections, os, re, glob

H = os.path.dirname(os.path.abspath(__file__))
rd1 = lambda f: list(csv.DictReader(open(os.path.join(H, f)))) if os.path.exists(os.path.join(H, f)) else []
rd = lambda f: sum((rd1(os.path.basename(g)) for g in sorted(glob.glob(os.path.join(H, f)))), []) if '*' in f else rd1(f)
fl = lambda x: float(x) if x not in ('', 'nan', None) else None
mean = lambda v: sum(v) / len(v) if v else None
layers_of = lambda ck: int(re.search(r'_tl(\d+)_', ck).group(1))
BADNQ = {('hp_sc4i_nodiscard_e3_n20_tl8_qlr3e-3_s0', 43)}   # before the get_char fix (04d6e92) 1722/1833 sentences failed to compile: typical width 43, not 64
isbad = lambda r: (r.get('ckpt'), int(float(r['nq'])) if r.get('nq') not in (None, '', 'nq') else None) in BADNQ

rows = collections.defaultdict(dict)
def R(ck, nq):
    x = rows[(layers_of(ck), int(float(nq)))]; x.setdefault('e', int(re.search(r'_e(\d+)_', ck).group(1))); x.setdefault('n', int(re.search(r'_n(\d+)_', ck).group(1))); return x

for r in rd('structure.csv') + rd('structure_big.csv') + rd('structure_grid_l*.csv'):                            # e1 structure (greedy) incl. tn widths
    if isbad(r) or r['ckpt'] == 'ckpt': continue
    x = R(r['ckpt'], r['nq']); x.update(ckpt=r['ckpt'], n2q_logical=fl(r['n2q_mean']), greedy_ps_width=fl(r['tn_width']), greedy_ps_lf=fl(r['tn_log10_flops']),
                                         greedy_rho_width=fl(r['tn_rho_width']), greedy_rho_lf=fl(r['tn_rho_log10_flops']))
for r in rd('gpu_engines.csv') + rd('gpu_engines_big.csv') + rd('gpu_engines_grid.csv'):
    if isbad(r) or r['ckpt'] == 'ckpt' or not fl(r['J_gross_per_decision']): continue
    x = R(r['ckpt'], r['nq']); x.setdefault('gpu', {})[r['engine']] = dict(J=fl(r['J_gross_per_decision']), s=fl(r['s_per_decision']), err=fl(r['max_abs_err']), gpu=r['gpu'], note=r['note'])
best = collections.defaultdict(lambda: collections.defaultdict(list))   # keyed by network shape (ckpt, tensors, indices): typical decisions share one network, so a path searched on one serves all
for f in ('tw_bounds_l2.csv', 'tw_bounds_l48_fromlog.csv', 'tw_bounds_l8fast.csv', 'tw_bounds_big.csv', 'tw_bounds_grid_l*.csv', 'tw_bounds_ctg_l*.csv'):
    for r in rd(f):
        if isbad(r) or r['width'] in ('nan', ''): continue
        best[(layers_of(r['ckpt']), int(r['nq']), r['net'], (r['ckpt'], r['n_tensors'], r['n_indices']))][r['opt']].append((fl(r['width']), fl(r['log10_flops']), fl(r['tw_lb'])))
per = collections.defaultdict(list)
for (L, nq, net, i), d in best.items():
    cand = [(v[0], o) for o, v in d.items()]
    (w, lf, lb), by = min(cand, key=lambda t: t[0][1])  # best (min flops) path over optimizers, and which one found it
    per[(L, nq, net)].append((w, lf, lb, d.get('greedy', [(None, None, None)])[0][1], by))
for (L, nq, net), v in per.items():
    x = rows[(L, nq)]; x[f'best_{net}_width'] = mean([t[0] for t in v]); x[f'best_{net}_lf'] = mean([t[1] for t in v]); x[f'tw_lb_{net}'] = max(t[2] for t in v)
    g = [t[3] for t in v if t[3] is not None]; x[f'greedy_{net}_lf_tw'] = mean(g)
    x[f'best_{net}_by'] = collections.Counter(t[4] for t in v).most_common(1)[0][0]
for f in ('qpu_esp_l2b.csv', 'qpu_esp_l48.csv', 'qpu_esp_big.csv', 'qpu_esp_grid_l*.csv'):
    acc = collections.defaultdict(list)
    for r in rd(f):
        if isbad(r): continue
        acc[(r['ckpt'], r['nq'])].append(r)
    for (ck, nq), v in acc.items():
        x = R(ck, nq)
        for k in ('ibm_duration_s', 'ibm_n2q', 'ibm_n1q', 'ionq_n1q', 'ionq_n2q'): x[k] = mean([fl(t[k]) for t in v])
# qpu_esp2_*.csv (--circ all): per decision the CU circuits (cu_pos, cu_neg: same counts) and the O-readout (e <= 6 only).
# The noise fit was measured on CU circuits, so the CU counts replace the older O-readout counts above; O is kept as *_O.
for f in ('qpu_esp2_*.csv',):
    acc = collections.defaultdict(list)
    for r in rd(f):
        if isbad(r) or r['ckpt'] == 'ckpt': continue
        acc[(r['ckpt'], r['nq'], 'O' if r['circ'] == 'O' else 'cu')].append(r)
    for (ck, nq, circ), v in acc.items():
        x = R(ck, nq); sfx = '' if circ == 'cu' else '_O'
        for k in ('ibm_duration_s', 'ibm_n2q', 'ibm_n1q', 'ionq_n1q', 'ionq_n2q'): x[k + sfx] = mean([fl(t[k]) for t in v])
        if circ == 'cu': x['qpu_circ'] = 'cu'
for f in ('mps_a.csv', 'mps_b.csv'):
    for r in rd(f):
        if isbad(r) or r['ckpt'] == 'ckpt' or not fl(r['chi']): continue
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
