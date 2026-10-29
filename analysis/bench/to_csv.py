# table.json (summarize.py) -> CSV datasets for the benchmark dashboard. Hardware estimates at a common budget
# of 1024 shots per circuit (2048 per decision) with the same assumptions as summarize.py.
#   python analysis/bench/to_csv.py results/bench_b2/table.json <out dir> [model to leave out ...]
import csv, json, sys
from pathlib import Path

t = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
A = t['assumptions']
MODEL = {'ps': 'Post-selection s6', 'discard': 'Discard s3 (best val)', 'discard6': 'Discard s6'}
SKIP = set(sys.argv[3:])   # e.g. discard: seed 3 is at chance on the swap set, even exact
READ = {'swap': 'Swap test', 'cu': 'Compute-uncompute'}
BACK = {'exact_inf': 'Exact (infinite shots)', 'exact_shots': 'Exact, 8192 shots', 'ibm_miami': 'IBM Miami',
        'honest_miami': 'Honest (Miami)', 'ibm_kingston': 'IBM Kingston', 'ionq_forte-1': 'IonQ Forte-1'}
PLAT = {'ibm_miami': 'IBM Miami', 'honest_miami': 'Honest (Miami)', 'ibm_kingston': 'IBM Kingston', 'ionq_forte-1': 'IonQ Forte-1'}
BUDGET = 2048   # shots per decision (1024 per circuit)


def hw(kind, dur):
    a = A[kind]
    per = dur + (a['rep_delay_s'] if kind == 'ibm' else a['shot_overhead_s'])
    s = BUDGET * per
    return s, s * a['power_W'], (s * a['usd_per_qpu_s'] if kind == 'ibm' else BUDGET * a['usd_per_shot'])


def r3(x, n=4):
    return '' if x is None else round(float(x), n)


res, curves, sat, split, dists, aux = [], [], [], [], [], []
for r in t['rows']:
    m, rd = r['model'].rsplit('_', 1)
    if m in SKIP:
        continue
    b = r['backend']
    if b.endswith('_debias_twirl'): bp, mit = b[:-13], 'debias (twirl limit)'
    elif b.endswith('_debias'): bp, mit = b[:-7], 'debias (4 variants)'
    else: bp, mit = b, 'none'
    run = f"{m}|{rd}|{b}"
    base = {'run': run, 'model': MODEL[m], 'readout': READ[rd], 'backend': BACK[bp] + ('' if mit == 'none' else ' + ' + mit),
            'platform': BACK[bp], 'mitigation': mit}
    st = r.get('stats') or {}
    kind = 'ionq' if b.startswith('ionq') else 'ibm' if b.startswith(('ibm', 'honest')) else None
    q = hw(kind, st['duration_s_mean']) if kind and st.get('duration_s_mean') else (None, None, None)
    s95 = r.get('shots_per_circuit_95')
    res.append({**base, 'accuracy': r3(r['acc_mean']), 'accuracy_sd': r3(r.get('acc_std')), 'accuracy_var': r3(r.get('acc_var'), 6),
                'agree_with_exact': r3(r.get('agree_with_exact'), 3), 'hellinger_fidelity': r3(r.get('hellinger_fid_mean')),
                'tvd': r3(r.get('tvd_mean')), 'score_spearman': r3(r.get('score_spearman_vs_exact'), 3),
                'margin_vs_exact': r3(r.get('margin_ratio_vs_exact'), 3), 'acceptance': r3(r.get('acceptance_mean'), 3),
                'acc_at_1k': r3((r.get('acc_at_shots') or {}).get('1024')),
                'shots95_per_decision': 2 * s95 if s95 else '', 'below_shot_noise': r3(r.get('frac_pairs_below_shot_noise'), 3),
                'depth': r3(st.get('depth_mean'), 1), 'depth_2q': r3(st.get('depth_2q_mean'), 1), 'gates_2q': r3(st.get('n2q_mean'), 1),
                'gates_2q_max': r3(st.get('n2q_max'), 0), 'qubits': r3(st.get('active_qubits_mean'), 1),
                'esp': r3(st.get('esp_mean') or st.get('esp_mean_model'), 3),
                'duration_us': r3(st['duration_s_mean'] * 1e6, 1) if st.get('duration_s_mean') else '',
                'sim_gpu_energy_kJ': r3(r['sim_gpu_energy_J'] / 1e3, 1) if r.get('sim_gpu_energy_J') else '',
                'sim_wall_s': r3(r.get('sim_wall_s') or r.get('wall_s'), 0),
                'qpu_s_per_decision': r3(q[0], 4), 'energy_J_per_decision': r3(q[1], 1), 'usd_per_decision': r3(q[2], 3)})
    for s, v in (r.get('acc_at_shots') or {}).items():
        curves.append({**base, 'shots_per_circuit': int(s), 'accuracy': r3(v)})
    sa = r.get('saturation') or {}
    g = lambda k: (sa.get(k) or {}).get('median')
    if sa and 'S_sat_analytic_0.01' in sa:
        sat.append({**base, 'k_eff': r3(g('keff_bhatt'), 1), 'k_eff_exact': r3(g('keff_bhatt_exact'), 1),
                    's_sat_1pct': r3(g('S_sat_analytic_0.01'), 0), 's_sat_1pct_fit': r3(g('S_sat_fit_0.01'), 0),
                    's_sat_01pct': r3(g('S_sat_analytic_0.001'), 0), 'noise_bias': r3(g('bias_vs_exact'), 4),
                    's_star': r3(g('S_star_crossover'), 0)})
        for s, v in sa['split_infid_curve'].items():
            split.append({**base, 'shots': int(s), 'split_half_infidelity': r3(v, 5),
                          'infidelity_to_exact': r3(sa['exact_infid_curve'].get(s), 5)})
    for k, v in (r.get('output_register_dist') or {}).items():
        dists.append({**base, 'outcome': k, 'probability': r3(v, 5)})
    for k, v in (r.get('aux_hamming_dist') or {}).items():
        aux.append({**base, 'aux_ones': int(k), 'probability': r3(v, 5)})

# with vs without debiasing, one row per model, readout and noisy platform that has both
eff = []
by = {(x['model'], x['readout'], x['platform'], x['mitigation']): x for x in res}
for (mo, rd, pl, mi), y in by.items():
    x = by.get((mo, rd, pl, 'none'))
    if mi == 'none' or x is None:
        continue
    # shown in the chart: twirl limit for the Aer backends (all sets), 4 variants for IonQ (its simulator ignores debiasing)
    shown = (mi == 'debias (twirl limit)') or (pl == 'IonQ Forte-1' and mi == 'debias (4 variants)')
    row = {'key': f"{mo}|{rd}|{pl}|{mi}", 'model': mo, 'readout': rd, 'platform': pl, 'mitigation': mi, 'in_chart': int(shown)}
    for f in ('accuracy', 'accuracy_sd', 'agree_with_exact', 'hellinger_fidelity', 'margin_vs_exact', 'acc_at_1k', 'below_shot_noise'):
        a_, b_ = x[f], y[f]
        row[f + '_off'] = a_; row[f + '_on'] = b_
        row[f + '_change'] = round(b_ - a_, 4) if a_ != '' and b_ != '' else ''
    eff.append(row)
if eff:
    with open(out / 'debias_effect.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(eff[0])); w.writeheader(); w.writerows(eff)
    print('debias_effect', len(eff))

for name, rows in [('results', res), ('curves', curves), ('saturation', sat), ('split_half', split), ('dists', dists), ('aux', aux)]:
    with open(out / f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(name, len(rows))
with open(out / 'assumptions.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['platform', 'parameter', 'value'])
    for k, d in A.items():
        for p, v in d.items():
            w.writerow([k, p, v])
