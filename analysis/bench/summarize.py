# Merge the backend benchmark outputs (backends.py, ionq_run.py) into one table with hardware estimates.
#   python analysis/bench/summarize.py results/bench_b1 > results/bench_b1/table.json
# Hardware time / energy / price figures are ESTIMATES from the assumptions below (no hardware was run).
import json, sys
from pathlib import Path

ASSUME = {
    # IBM superconducting: per shot = scheduled circuit duration + default repetition delay; system power incl. dilution
    # refrigerator, compressor, control electronics (commonly quoted ~15-25 kW for a System One class machine).
    'ibm': {'rep_delay_s': 250e-6, 'job_overhead_s': 2.0, 'power_W': 25e3, 'usd_per_qpu_s': 1.60},
    # IonQ trapped ion (Forte-class): gate/readout times of the published Aria-class figures, per-shot cooling + state
    # prep overhead; errors forte-1 class (1q 99.98 %, 2q 99.6 %, SPAM 99.5 %); system power assumed ~10 kW (no cryostat);
    # price: AWS Braket list price for Forte ($0.30 per task + $0.08 per shot).
    'ionq': {'t1q': 135e-6, 't2q': 600e-6, 'tmeas': 300e-6, 'shot_overhead_s': 1e-3, 'job_overhead_s': 2.0,
             'e1q': 2e-4, 'e2q': 4e-3, 'espam': 5e-3, 'power_W': 10e3, 'usd_per_task': 0.30, 'usd_per_shot': 0.08},
}


def hw_estimate(kind, dur_s, shots_per_decision, n_circuits_per_decision=2):
    a = ASSUME[kind]
    per_shot = dur_s + (a['rep_delay_s'] if kind == 'ibm' else a['shot_overhead_s'])
    t = shots_per_decision * per_shot   # QPU seconds per decision (job overheads amortised away)
    usd = t * a['usd_per_qpu_s'] if kind == 'ibm' else shots_per_decision * a['usd_per_shot']
    return {'qpu_s_per_decision': t, 'energy_J_per_decision': t * a['power_W'], 'usd_per_decision': usd, 'per_shot_s': per_shot}


def main(root):
    out = []
    for d in sorted(Path(root).iterdir()):
        if not (d / 'summary.json').exists():
            continue
        s = json.load(open(d / 'summary.json'))
        rows = s['rows'] + [r for f in sorted(d.glob('summary_ionq_*.json')) + sorted(d.glob('summary_debias.json')) for r in json.load(open(f))['rows']]
        meta = s['meta']
        sat = json.load(open(d / 'saturation.json')) if (d / 'saturation.json').exists() else {}   # saturation.py
        ex = rows[0]
        ionq_stats = dict(ex['stats_ionq_native'])
        a = ASSUME['ionq']
        ionq_stats['duration_s_mean'] = a['t1q'] * ionq_stats['depth_mean'] + a['t2q'] * ionq_stats['depth_2q_mean'] + a['tmeas']
        ionq_stats['esp_mean_model'] = (1 - a['e1q']) ** ionq_stats['n1q_mean'] * (1 - a['e2q']) ** ionq_stats['n2q_mean'] * (1 - a['espam']) ** ionq_stats['nmeas_mean']
        for r in rows:
            r = dict(r)
            r['model'] = d.name; r['discard'] = meta['discard']; r['meas'] = meta['meas']; r['pairs'] = meta['pairs']
            b = r['backend']
            st = r.get('stats') or (ionq_stats if b.startswith('ionq') else ex['stats_logical'])
            r['stats'] = st
            r['saturation'] = sat.get(b)
            s95 = r.get('shots_per_circuit_95')
            r['shots_per_decision_95'] = 2 * s95 if s95 else None
            kind = 'ionq' if b.startswith('ionq') else 'ibm' if b.startswith(('ibm', 'honest')) else None
            if kind and s95 and st.get('duration_s_mean'):
                r['hw'] = hw_estimate(kind, st['duration_s_mean'], 2 * s95)
                r['hw_at_run_shots'] = hw_estimate(kind, st['duration_s_mean'], 2 * meta['shots'])
            out.append(r)
    json.dump({'assumptions': ASSUME, 'rows': out}, sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1])
