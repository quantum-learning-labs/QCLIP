# Shot curves on a fine grid for every backend of a results dir (backends.py, ionq_run.py, debias.py, twirl_limit.py),
# and the shots per circuit at which each backend's expected accuracy first reaches fixed targets (the accuracy behind
# energy-to-solution). A target relative to each backend's own large-shot accuracy was unstable: on 95 pairs one pair is
# ~1 point and near-tie pairs set the shot count. Writes energy.json next to summary.json; to_csv.py turns it into energy.
#   python analysis/bench/energy.py results/bench_b2/<set>
import argparse, json, pickle
from pathlib import Path
import numpy as np
import bench_lib as B

CURVE = [2 ** k for k in range(4, 21)]            # points of the accuracy-vs-energy curve
FINE = sorted(set(int(round(2 ** (k / 8))) for k in range(32, 193)))   # 16 .. 2^24, 8 points per octave


def sources(d):
    raw = pickle.load(open(d / 'raw.pkl', 'rb'))
    yield 'exact_inf', raw['exact']
    for k, v in raw.items():
        if isinstance(v, dict) and 'extra' in v:
            yield k, v['extra']
    for f in sorted(d.glob('raw_debias*.pkl')):
        for k, v in pickle.load(open(f, 'rb')).items():
            yield k, v['extra']
    for f in sorted(d.glob('raw_ionq_*.pkl')):
        name = f.stem[4:]            # raw_ionq_forte-1(_debias).pkl -> ionq_forte-1(_debias)
        yield name, pickle.load(open(f, 'rb'))['extra']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir'); p.add_argument('--targets', default='0.75,0.8,0.85')
    a = p.parse_args()
    d = Path(a.dir); meta = json.load(open(d / 'summary.json'))['meta']; n = meta['pairs']
    tg = [float(x) for x in a.targets.split(',')]
    out = {'targets': tg, 'backends': {}}
    for name, ex in sources(d):
        m, acc = ex['m'], ex['a']
        mp, ap, mn, an = m[:n], acc[:n], m[n:], acc[n:]
        asym = float(np.mean((np.asarray(mp) > np.asarray(mn)) + 0.5 * (np.asarray(mp) == np.asarray(mn))))
        fine = {s: B.expected_acc(mp, ap, mn, an, meta['meas'], s) for s in FINE}
        s_t = {t: next((s for s in FINE if fine[s] >= t), None) for t in tg}
        out['backends'][name] = {'large_shot_acc': asym, 'shots_to': s_t,
                                 'curve': {s: B.expected_acc(mp, ap, mn, an, meta['meas'], s) for s in CURVE}}
        print(f'{d.name} {name}: large-shot {asym:.3f}, shots/circuit to reach', s_t, flush=True)
    json.dump(out, open(d / 'energy.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
