# Shot saturation of the output distributions, from the counts saved by backends.py / ionq_run.py (no new runs).
# Per model x readout x backend, over circuits (median and mean):
#  - K_eff: effective number of outcomes, perplexity exp(H) and Bhattacharyya (sum sqrt p)^2
#  - split-half curve: Hellinger infidelity 1-F_H and TVD between two disjoint S-shot samples, S = 16..4096
#  - S_sat(eps): shots per circuit at which the expected infidelity to the backend's own distribution is eps,
#      analytic (K_eff - 1) / (4 eps), and fitted from the split-half curve (1-F_H between two samples ~ c / S,
#      one sample vs the truth ~ c / (2 S))
#  - distance to exact vs S (subsamples of rep 0), and for noisy backends the bias (infidelity to exact that remains
#      at infinite shots) and the crossover S* = (K_eff - 1) / (4 bias) beyond which more shots do not bring the
#      distribution closer to exact.
#   PYTHONPATH=.:analysis/bench python analysis/bench/saturation.py results/bench_b1/<dir> ...  (writes <dir>/saturation.json)
import json, pickle, sys
from pathlib import Path
import numpy as np

GRID = [2 ** k for k in range(4, 13)]   # shots per sample; two disjoint samples need 2 S <= 8192
EPS = (0.01, 0.001)
rng = np.random.default_rng(0)


def vec(d, keys):
    v = np.array([d.get(k, 0.0) for k in keys], float)
    return v / v.sum() if v.sum() > 0 else v


def infid(p, q):
    return 1.0 - float(np.sum(np.sqrt(p * q))) ** 2


def tvd(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def keff(p):
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(np.sqrt(p).sum() ** 2)


def shots_array(counts, keys):
    idx = {k: i for i, k in enumerate(keys)}
    return np.repeat([idx[k] for k in counts], [int(c) for c in counts.values()])


def circuit(counts, exact, pooled_hel=None, pooled_shots=None):
    keys = sorted(set(counts) | set(exact))
    pe, pc = vec(exact, keys), vec(counts, keys)
    shots = shots_array(counts, keys)
    n = len(shots)
    perm = rng.permutation(shots)
    split, to_exact = {}, {}
    for s in GRID:
        if 2 * s > n:
            break
        a = np.bincount(perm[:s], minlength=len(keys)) / s
        b = np.bincount(perm[s:2 * s], minlength=len(keys)) / s
        split[s] = (infid(a, b), tvd(a, b))
        to_exact[s] = (infid(a, pe), tvd(a, pe))
    xs = np.array(list(split)); ys = np.array([v[0] for v in split.values()])
    c = float(np.exp(np.mean(np.log(np.maximum(ys, 1e-12)) + np.log(xs))))   # 1 - F ~ c / S, fit in log space
    ke_c, kb_c = keff(pc)
    ke_e, kb_e = keff(pe)
    out = {'keff_perp': ke_c, 'keff_bhatt': kb_c, 'keff_perp_exact': ke_e, 'keff_bhatt_exact': kb_e,
           'split_infid': {s: v[0] for s, v in split.items()}, 'split_tvd': {s: v[1] for s, v in split.items()},
           'exact_infid': {s: v[0] for s, v in to_exact.items()}, 'exact_tvd': {s: v[1] for s, v in to_exact.items()},
           'c_fit': c}
    for e in EPS:
        out[f'S_sat_analytic_{e}'] = (kb_c - 1) / (4 * e)
        out[f'S_sat_fit_{e}'] = c / (2 * e)
    if pooled_hel is not None:   # infidelity to exact at pooled_shots minus its shot-noise part = noise bias
        bias = max((1 - pooled_hel) - (kb_c - 1) / (4 * pooled_shots), 1e-9)
        out['bias_vs_exact'] = bias
        out['S_star_crossover'] = (kb_c - 1) / (4 * bias)
    return out


def agg(rows):
    def stat(key):
        v = np.array([r[key] for r in rows if r.get(key) is not None], float)
        return {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'p90': float(np.percentile(v, 90))} if len(v) else None
    res = {k: stat(k) for k in rows[0] if not isinstance(rows[0][k], dict)}
    for k in ('split_infid', 'split_tvd', 'exact_infid', 'exact_tvd'):
        ss = sorted({s for r in rows for s in r[k]})
        res[k + '_curve'] = {s: float(np.median([r[k][s] for r in rows if s in r[k]])) for s in ss}
    return res


def main(dirs):
    for d in map(Path, dirs):
        raw = pickle.load(open(d / 'raw.pkl', 'rb'))
        meta = json.load(open(d / 'summary.json'))['meta']
        exact = raw['exact']['probs']
        pooled = meta['shots'] * meta['reps']
        res = {}
        for name, v in raw.items():
            if not isinstance(v, dict) or 'counts_rep0' not in v:
                continue
            hel = v['extra'].get('hellinger') if name != 'exact_shots' else None
            res[name] = agg([circuit(c, exact[j], hel[j] if hel else None, pooled) for j, c in enumerate(v['counts_rep0'])])
        if (d / 'raw_debias.pkl').exists():   # debias.py rows, same structure as raw.pkl entries
            for name, v in pickle.load(open(d / 'raw_debias.pkl', 'rb')).items():
                res[name] = agg([circuit(c, exact[j], v['extra']['hellinger'][j], pooled) for j, c in enumerate(v['counts_rep0'])])
        for f in sorted(d.glob('raw_ionq_*.pkl')):
            v = pickle.load(open(f, 'rb'))
            hel = v['extra']['hellinger']
            res['ionq_' + f.stem.split('raw_ionq_')[1]] = agg([circuit(c, exact[j], hel[j], pooled) for j, c in enumerate(v['counts'][0])])
        res['exact_inf'] = {'keff_perp': {'median': float(np.median([keff(vec(p, sorted(p)))[0] for p in exact]))},
                            'keff_bhatt': {'median': float(np.median([keff(vec(p, sorted(p)))[1] for p in exact]))}}
        json.dump(res, open(d / 'saturation.json', 'w'), indent=1)
        for n, r in res.items():
            if 'S_sat_analytic_0.01' in r:
                print(d.name, n, 'K_eff', round(r['keff_bhatt']['median'], 1), 'S_sat(1%) analytic', round(r['S_sat_analytic_0.01']['median']),
                      'fit', round(r['S_sat_fit_0.01']['median']), 'S*', round(r['S_star_crossover']['median']) if r.get('S_star_crossover') else '-')


if __name__ == '__main__':
    main(sys.argv[1:])
