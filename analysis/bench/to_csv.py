# table.json (summarize.py) + <set>/energy.json (energy.py) + classical/classical.json (classical_energy.py) -> CSV
# datasets for the benchmark dashboard. Hardware time, energy and price are ESTIMATES from COST below, per decision
# (2 circuits) at the shots each backend needs to reach TARGET accuracy (energy-to-solution), plus per shot.
#   python analysis/bench/to_csv.py results/bench_b2/table.json <out dir> <classical.json> [model to leave out ...]
import csv, json, sys
from pathlib import Path

t = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
root = Path(sys.argv[1]).parent
cl = json.load(open(sys.argv[3]))
MODEL = {'ps': 'Post-selection s6', 'discard': 'Discard s3 (best val)', 'discard6': 'Discard s6'}
SKIP = set(sys.argv[4:])   # e.g. discard: seed 3 is at chance on the swap set, even exact
READ = {'swap': 'Swap test', 'cu': 'Compute-uncompute'}
BACK = {'exact_inf': 'Exact (infinite shots)', 'exact_shots': 'Exact, 8192 shots', 'ibm_miami': 'IBM Miami',
        'honest_miami': 'Honest (Miami)', 'ibm_kingston': 'IBM Kingston', 'ionq_forte-1': 'IonQ Forte-1'}
PLAT = {'ibm_miami': 'IBM Miami', 'honest_miami': 'Honest (Miami)', 'ibm_kingston': 'IBM Kingston', 'ionq_forte-1': 'IonQ Forte-1'}
TARGET = 0.8   # accuracy on the swap set behind energy-to-solution (one of energy.py's targets)
# IonQ: Knitter et al. 2026 (arXiv 2605.02798) App. B, gates run one after another, ~5 kW from the Forte Enterprise's
# own power logs; price AWS Braket list. IBM: default repetition delay, 2 s job overhead shared by the 190 circuits of
# one job, ~25 kW for a System One class machine incl. dilution refrigerator (public figures, not measured); list price.
COST = {
    'ionq': {'t1q_s': 1.1e-4, 't2q_s': 9e-4, 'shot_overhead_s': 1.5e-3, 'circuit_overhead_s': 10.0, 'power_W': 5e3,
             'usd_per_shot': 0.08, 'usd_per_task': 0.30},
    'ibm': {'rep_delay_s': 250e-6, 'job_overhead_s': 2.0, 'circuits_per_job': 190, 'power_W': 25e3, 'usd_per_qpu_s': 1.60},
}
SRC = {'ionq': 'Knitter et al. 2026, arXiv 2605.02798 (App. B); price: AWS Braket', 'ibm': 'assumed (public figures); price: IBM list'}


def timing(kind, st):
    # seconds per shot and fixed seconds per circuit
    c = COST[kind]
    if kind == 'ionq':
        return st['n1q_mean'] * c['t1q_s'] + st['n2q_mean'] * c['t2q_s'] + c['shot_overhead_s'], c['circuit_overhead_s']
    return st['duration_s_mean'] + c['rep_delay_s'], c['job_overhead_s'] / c['circuits_per_job']


def per_decision(kind, ts, tc, shots):
    # QPU seconds, joules and USD for one decision (2 circuits) at `shots` per circuit
    c = COST[kind]; s = 2 * (shots * ts + tc)
    usd = s * c['usd_per_qpu_s'] if kind == 'ibm' else 2 * (shots * c['usd_per_shot'] + c['usd_per_task'])
    return s, s * c['power_W'], usd


CL_J = cl['decision_single']['gpu_J_per_item']   # one CLIP decision on its own (batch of 2 captions), GPU energy


def meta_pairs(r):
    return r.get('pairs') or 95


def r3(x, n=4):
    return '' if x is None else round(float(x), n)


res, curves, sat, split, dists, aux, ecurve = [], [], [], [], [], [], []
EJ, STATS = {}, {}
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
    if r['model'] not in EJ:
        EJ[r['model']] = json.load(open(root / r['model'] / 'energy.json'))['backends']
    ej = EJ[r['model']].get(b, {})
    s_t = (ej.get('shots_to') or {}).get(str(TARGET))
    en = {}
    if kind == 'ibm' and not st.get('duration_s_mean'):   # debias rows carry no schedule: same circuits as the plain row
        st = STATS[(r['model'], bp)]
    if kind and mit == 'none':
        STATS[(r['model'], bp)] = st
    if kind:
        ts, tc = timing(kind, st)
        en = {'shot_ms': r3(ts * 1e3, 3), 'circuit_overhead_s': r3(tc, 3), 'J_per_shot': r3(ts * COST[kind]['power_W'], 2),
              'mJ_per_2q_gate_shot': r3(ts * COST[kind]['power_W'] / st['n2q_mean'] * 1e3, 1), 'shots_to_target': s_t or ''}
        if s_t:
            q = per_decision(kind, ts, tc, s_t)
            en.update(qpu_s_per_decision=r3(q[0], 3), kJ_per_decision=r3(q[1] / 1e3, 3), usd_per_decision=r3(q[2], 2),
                      kWh_eval_set=r3(q[1] * meta_pairs(r) / 3.6e6, 3), x_classical=r3(q[1] / CL_J, 0))
        if mit == 'none':
            for sh, acc in ej.get('curve', {}).items():
                ecurve.append({**base, 'shots_per_circuit': int(sh), 'accuracy': r3(acc), 'J_per_decision': r3(per_decision(kind, ts, tc, int(sh))[1], 1)})
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
                'sim_gpu_energy_kJ': r3(r['sim_gpu_energy_J'] / 1e3, 1) if r.get('sim_gpu_energy_J') else '',
                'sim_wall_s': r3(r.get('sim_wall_s') or r.get('wall_s'), 0),
                **{k: en.get(k, '') for k in ('shot_ms', 'circuit_overhead_s', 'J_per_shot', 'mJ_per_2q_gate_shot', 'shots_to_target',
                                               'qpu_s_per_decision', 'kJ_per_decision', 'usd_per_decision', 'kWh_eval_set', 'x_classical')}})
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

cls = [{'model': cl['model'], 'gpu': cl['gpu'], 'eval_set': k[4:], 'pairs': cl['pairs_' + k[4:]], 'accuracy': r3(v)}
       for k, v in cl.items() if k.startswith('acc_')]
for k, lab in (('decision_single', 'One decision (2 captions)'), ('decision_batched', '95 decisions batched'), ('image_encode_batch64', 'Image encoder, per image (batch 64)')):
    m = cl[k]
    cls.append({'model': cl['model'], 'gpu': cl['gpu'], 'eval_set': lab, 'pairs': '', 'accuracy': '',
                'J_per_item': r3(m['gpu_J_per_item'], 5), 'ms_per_item': r3(m['s_per_item'] * 1e3, 3), 'gpu_W': r3(m['gpu_W'], 0),
                'idle_W': r3(cl['idle_W'], 0)})
for row in cls:
    for f in ('J_per_item', 'ms_per_item', 'gpu_W', 'idle_W'):
        row.setdefault(f, '')
for name, rows in [('results', res), ('curves', curves), ('saturation', sat), ('split_half', split), ('dists', dists), ('aux', aux),
                   ('energy_curve', ecurve), ('classical', cls)]:
    with open(out / f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(name, len(rows))
with open(out / 'assumptions.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['platform', 'parameter', 'value', 'source'])
    for k, d in COST.items():
        for p, v in d.items():
            w.writerow([k, p, v, SRC[k]])
    w.writerow(['all', 'target_accuracy', TARGET, 'energy-to-solution target on the swap set'])
