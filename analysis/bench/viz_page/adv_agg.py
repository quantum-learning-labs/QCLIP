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

for r in rd('structure.csv') + rd('structure_big.csv') + rd('structure_grid_l*.csv') + rd('structure_fill.csv'):                            # e1 structure (greedy) incl. tn widths
    if isbad(r) or r['ckpt'] == 'ckpt': continue
    x = R(r['ckpt'], r['nq']); x.update(ckpt=r['ckpt'], n2q_logical=fl(r['n2q_mean']), greedy_ps_width=fl(r['tn_width']), greedy_ps_lf=fl(r['tn_log10_flops']),
                                         greedy_rho_width=fl(r['tn_rho_width']), greedy_rho_lf=fl(r['tn_rho_log10_flops']), greedy_path_s=fl(r.get('path_s')))
for r in rd('gpu_engines.csv') + rd('gpu_engines_big.csv') + rd('gpu_engines_grid.csv') + rd('gpu_engines_fill.csv'):
    if isbad(r) or r['ckpt'] == 'ckpt' or not fl(r['J_gross_per_decision']): continue
    x = R(r['ckpt'], r['nq']); x.setdefault('gpu', {})[r['engine']] = dict(J=fl(r['J_gross_per_decision']), s=fl(r['s_per_decision']), err=fl(r['max_abs_err']), gpu=r['gpu'], note=r['note'])
best = collections.defaultdict(lambda: collections.defaultdict(list))   # keyed by network shape (ckpt, tensors, indices): typical decisions share one network, so a path searched on one serves all
for f in ('tw_bounds_l2.csv', 'tw_bounds_l48_fromlog.csv', 'tw_bounds_l8fast.csv', 'tw_bounds_big.csv', 'tw_bounds_grid_l*.csv', 'tw_bounds_ctg_l*.csv', 'tw_bounds_fill*.csv', 'tw_bounds_ctg3_*.csv'):
    for r in rd(f):
        if isbad(r) or r['width'] in ('nan', ''): continue
        best[(layers_of(r['ckpt']), int(r['nq']), r['net'], (r['ckpt'], r['n_tensors'], r['n_indices']))][r['opt']].append((fl(r['width']), fl(r['log10_flops']), fl(r['tw_lb']), fl(r.get('path_s'))))
per = collections.defaultdict(list)
for (L, nq, net, i), d in best.items():
    cand = [(t, o) for o, v in d.items() for t in v]   # every row: one shape can have several runs of the same optimiser (e.g. cotengra at two budgets)
    (w, lf, lb, _), by = min(cand, key=lambda t: t[0][1])  # best (min flops) path over optimizers, and which one found it
    per[(L, nq, net)].append((w, lf, lb, d.get('greedy', [(None, None, None, None)])[0][1], by))
for (L, nq, net), v in per.items():
    x = rows[(L, nq)]; x[f'best_{net}_width'] = mean([t[0] for t in v]); x[f'best_{net}_lf'] = mean([t[1] for t in v]); x[f'tw_lb_{net}'] = max(t[2] for t in v)
    g = [t[3] for t in v if t[3] is not None]; x[f'greedy_{net}_lf_tw'] = mean(g)
    x[f'best_{net}_by'] = collections.Counter(t[4] for t in v).most_common(1)[0][0]
# One-off costs. Every path found, per network shape, with its search time: the page picks, for a given number of decisions per shape,
# the path with the lowest (contraction + search / reuse) energy, so a long search only pays off when the shape is reused enough.
cands = collections.defaultdict(list)
for (L, nq, net, i), d in best.items():
    c = [dict(o=o, lf=round(t[1], 3), s=round(t[3], 2)) for o, v in d.items() for t in v if t[1] is not None and len(t) > 3 and t[3] is not None]
    if c: cands[(L, nq, net)].append(c)
for (L, nq, net), v in cands.items(): rows[(L, nq)][f'paths_{net}'] = v
# transpile seconds per circuit and seed (qpu_tt_*: dedicated timing runs; any later qpu_esp file with the column is read too)
for f in ('qpu_tt_*.csv', 'qpu_esp4_*.csv'):
    acc = collections.defaultdict(list)
    for r in rd(f):
        if isbad(r) or r['ckpt'] == 'ckpt' or r.get('circ') == 'O' or not r.get('ibm_transpile_s'): continue
        acc[(r['ckpt'], r['nq'])].append(r)
    for (ck, nq), v in acc.items():
        x = R(ck, nq); x['ibm_tt'] = min(fl(t['ibm_transpile_s']) for t in v); x['ionq_tt'] = min(fl(t['ionq_transpile_s']) for t in v)   # min over the CU circuits (same shape): the first call of a process also builds the backend target (~12 s, once per session)
# anytime path search (path_anytime.py): per network, the best path's log10 flops after s seconds of one cotengra search (+ the greedy start),
# and the exact optimum where the dynamic programme finished (status ok) or its cap (timeout / memory)
anchors = collections.defaultdict(dict)   # circuits outside the size grid (trained 6-13 qubit models): kept apart, as the exact-optimum anchor
for r in rd('anyt_*.csv'):
    if r['ckpt'] == 'ckpt' or isbad(r): continue
    key = (layers_of(r['ckpt']), int(float(r['nq'])))
    if key in rows: x = R(r['ckpt'], r['nq'])
    else: x = anchors[(r['ckpt'], key[1])]; x.update(layers=key[0], nq=key[1], e=int(re.search(r'_e(\d+)_', r['ckpt']).group(1)), n=int(re.search(r'_n(\d+)_', r['ckpt']).group(1)))
    net = r['net']; lf = fl(r['log10_flops'])
    if r['kind'] in ('trace', 'greedy') and lf is not None:
        x.setdefault(f'anyt_{net}', []).append(dict(s=round(fl(r['elapsed_s']), 2), lf=round(lf, 3), w=fl(r['width']), k=r['kind']))
    elif r['kind'] == 'optimal':
        x[f'optimal_{net}'] = dict(status=r['status'], s=round(fl(r['elapsed_s']), 1), lf=None if lf is None else round(lf, 3), cap=fl(r['checkpoint_s']))
for x in list(rows.values()) + list(anchors.values()):
    for net in ('ps', 'rho'):
        if f'anyt_{net}' in x: x[f'anyt_{net}'] = sorted(x[f'anyt_{net}'], key=lambda t: t['s'])
# decisions per circuit shape on the std test set (shape_reuse.py)
reuse = None
for f in sorted(glob.glob(os.path.join(H, 'shape_reuse_*.csv'))):
    v = list(csv.DictReader(open(f))); sh = collections.Counter(r['text_shape'] for r in v)
    img = len(set(r['img_pos_shape'] for r in v) | set(r['img_neg_shape'] for r in v))
    reuse = reuse or dict(decisions=len(v), shapes=len(sh), largest=max(sh.values()), singletons=sum(1 for c in sh.values() if c == 1), image_shapes=img, src=os.path.basename(f))
    print('reuse', os.path.basename(f), len(v), 'decisions', len(sh), 'shapes', 'image shapes', img)
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
for f in ('qpu_esp2_*.csv', 'qpu_esp3_*.csv'):   # esp3 (best of 5 transpile seeds) read last: overrides the single-seed counts
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
           reuse=reuse, opt_anchor=sorted(anchors.values(), key=lambda x: (x['layers'], x['nq'], x['e'])),
           noise_fit=dict(a=-0.187, b=-0.00331, note='ln F_eff = a + b * n2q_native (Kingston CU readout, 54 noisy runs, 6-13 qubits)'))
json.dump(out, open(os.path.join(H, 'adv.json'), 'w'), indent=1)
for r in out['rows']:
    print(r['layers'], r['nq'], {k: (round(v, 2) if isinstance(v, float) else '…') for k, v in r.items() if k in ('ibm_n2q', 'best_ps_lf', 'best_rho_lf', 'greedy_ps_lf', 'greedy_rho_lf')},
          {e: round(g['J'], 3) for e, g in r.get('gpu', {}).items()}, {n: len(v) for n, v in r.get('mps', {}).items()})
