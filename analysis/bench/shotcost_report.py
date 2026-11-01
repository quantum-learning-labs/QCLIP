# Shot-cost study, step 2: from shotcost.py's exact (m, a, k) per circuit, for each checkpoint, eval set and readout:
#  - expected accuracy vs shots (normal approximation, as bench_lib.expected_acc; rejected PS shots are charged) and the
#    shots PER DECISION (2 circuits) to reach absolute accuracy targets;
#  - cost decomposition per correctly ranked pair: shots ~ V/D^2, V = v_pos/a_pos + v_neg/a_neg = Vraw/a_eff with
#    Vraw = v_pos + v_neg, so log cost = log Vraw - log a_eff - 2 log D exactly (D = m_pos - m_neg, v per accepted shot);
#  - paired comparison discard vs PS of the same seed on the pairs both rank correctly: log cost ratio split into
#    acceptance, contrast and variance terms, and beta = log(D_disc/D_ps) / log(a_ps) (0: no contrast loss,
#    0.5: costs equal, 1: Theorem A's p0 scaling);
#  - the same quantities binned by k (post-selected / discarded qubits per circuit).
# usage: shotcost_report.py <dir with npz/json> [out.json]
import glob, json, os, re, sys, collections
import numpy as np
from scipy.stats import norm

TARGETS = (0.70, 0.75, 0.80, 0.85)
GRID = np.unique(np.round(2 ** np.arange(2, 26.01, 0.125)).astype(np.int64))
READOUTS = ('compute_uncompute', 'destructive_swap')


def var(m, ro):
    return np.maximum(1 - m * m, 1e-12) if ro == 'destructive_swap' else np.maximum(m * (1 - m), 1e-12)


def load(d):
    out = {}
    for f in sorted(glob.glob(f'{d}/*.npz')):
        meta = json.load(open(f[:-4] + '.json'))
        out[os.path.basename(f)[:-4]] = (dict(np.load(f)), meta)
    return out


def family(ckpt_name):
    return re.sub(r'_s\d+$', '', ckpt_name)


def curve(x, ro):
    V = var(x['m_pos'], ro) / x['a_pos'] + var(x['m_neg'], ro) / x['a_neg']
    D = x['m_pos'] - x['m_neg']
    acc = np.array([np.mean(norm.cdf(D / np.sqrt(V / s))) for s in GRID])
    return acc


def shots_to(acc, t):
    # shots per DECISION (2 circuits) at which the expected accuracy first reaches t, log-interpolated on the grid
    i = np.argmax(acc >= t)
    if acc[i] < t:
        return None
    if i == 0:
        return float(2 * GRID[0])
    x0, x1, y0, y1 = np.log(GRID[i - 1]), np.log(GRID[i]), acc[i - 1], acc[i]
    return float(2 * np.exp(x0 + (t - y0) / (y1 - y0) * (x1 - x0)))


def decomp(x, ro, sel):
    vp, vn = var(x['m_pos'], ro)[sel], var(x['m_neg'], ro)[sel]
    ap, an = x['a_pos'][sel], x['a_neg'][sel]
    D = (x['m_pos'] - x['m_neg'])[sel]
    Vraw = vp + vn
    a_eff = Vraw / (vp / ap + vn / an)
    return {'logVraw': np.log(Vraw), 'loga': np.log(a_eff), 'logD': np.log(D), 'logcost': np.log(Vraw) - np.log(a_eff) - 2 * np.log(D)}


def main():
    d = sys.argv[1]
    data = load(d)
    res = {'per_ckpt': {}, 'paired': {}, 'by_k': {}}
    for key, (x, meta) in data.items():
        ck, s = key.split('__')
        correct = x['m_pos'] > x['m_neg']
        row = {'family': family(ck), 'set': s, 'discard': meta['discard'], 'text_n': meta['text_n'], 'pairs': int(len(correct)),
               'exact_acc': meta['exact_acc'], 'accept_mean': float(np.mean(np.r_[x['a_pos'], x['a_neg']])),
               'k_mean': float(np.mean(np.r_[x['k_pos'], x['k_neg']])), 'median_margin_correct': float(np.median((x['m_pos'] - x['m_neg'])[correct]))}
        for ro in READOUTS:
            acc = curve(x, ro)
            dc = decomp(x, ro, correct)
            row[ro] = {'shots_per_decision': {str(t): shots_to(acc, t) for t in TARGETS},
                       'acc_at': {str(int(2 * s_)): float(a_) for s_, a_ in zip(GRID, acc) if s_ in (32, 128, 512, 2048, 8192, 32768, 131072)},
                       'geo': {k: float(np.exp(np.mean(v))) for k, v in dc.items()}}
        res['per_ckpt'][key] = row
    # paired: discard ckpt vs PS ckpt of the same seed, text_n and set
    ps = {(r['set'], r['text_n'], re.search(r'_s(\d+)__', k + '__').group(1)): k for k, r in res['per_ckpt'].items() if not r['discard']}
    for k, r in res['per_ckpt'].items():
        if not r['discard']:
            continue
        seed = re.search(r'_s(\d+)__', k + '__').group(1)
        kp = ps.get((r['set'], r['text_n'], seed))
        if kp is None:
            continue
        xd, xp = data[k][0], data[kp][0]
        if len(xd['m_pos']) != len(xp['m_pos']):
            print('pair count mismatch', k, kp); continue
        both = (xd['m_pos'] > xd['m_neg']) & (xp['m_pos'] > xp['m_neg'])
        out = {'ps': kp, 'common_correct': int(both.sum())}
        for ro in READOUTS:
            a, b = decomp(xd, ro, both), decomp(xp, ro, both)
            terms = {'acceptance': float(np.mean(b['loga'] - a['loga'])),    # log(a_ps/a_d) < 0: PS pays for rejection
                     'contrast': float(np.mean(2 * (b['logD'] - a['logD']))),  # >0: discard margin smaller
                     'variance': float(np.mean(a['logVraw'] - b['logVraw']))}
            tot = float(np.mean(a['logcost'] - b['logcost']))
            out[ro] = {'log_cost_ratio_disc_over_ps': tot, 'cost_ratio_geo': float(np.exp(tot)),
                       'terms_log': terms, 'terms_factor': {t: float(np.exp(v)) for t, v in terms.items()},
                       'beta': float(np.mean(a['logD'] - b['logD']) / np.mean(b['loga']))}
        # k-binned (pairs whose pos circuit has k post-selected qubits in the PS model)
        kk = xp['k_pos']
        out['by_k'] = {}
        for kv in np.unique(kk):
            sel = both & (kk == kv)
            if sel.sum() < 20:
                continue
            a, b = decomp(xd, 'compute_uncompute', sel), decomp(xp, 'compute_uncompute', sel)
            out['by_k'][int(kv)] = {'n': int(sel.sum()), 'a_ps_geo': float(np.exp(np.mean(b['loga']))),
                                    'D_ratio_geo': float(np.exp(np.mean(a['logD'] - b['logD']))),
                                    'cost_ratio_cu_geo': float(np.exp(np.mean(a['logcost'] - b['logcost'])))}
        res['paired'][k] = out
    # print
    print(f"{'checkpoint':48s} {'set':9s} {'acc':>6s} {'accept':>6s} {'k':>4s} | CU shots/decision to " + ' '.join(f'{t:.2f}' for t in TARGETS)
          + ' | swap shots/decision to ' + ' '.join(f'{t:.2f}' for t in TARGETS))
    fmt = lambda v: f'{v:8.0f}' if v is not None else '       -'
    for k, r in sorted(res['per_ckpt'].items(), key=lambda kv: (kv[1]['set'], kv[1]['text_n'] or 0, kv[1]['discard'], kv[0])):
        print(f"{k.split('__')[0]:48s} {r['set']:9s} {r['exact_acc']:6.3f} {r['accept_mean']:6.3f} {r['k_mean']:4.1f} | "
              + ' '.join(fmt(r['compute_uncompute']['shots_per_decision'][str(t)]) for t in TARGETS) + ' | '
              + ' '.join(fmt(r['destructive_swap']['shots_per_decision'][str(t)]) for t in TARGETS))
    print('\npaired (discard / PS, same seed, pairs both rank correctly): geo-mean cost ratio = acceptance x contrast x variance factors')
    for k, o in sorted(res['paired'].items()):
        for ro in READOUTS:
            t = o[ro]['terms_factor']
            print(f"{k:60s} {ro[:7]:7s} n={o['common_correct']:5d} ratio {o[ro]['cost_ratio_geo']:6.2f} = acc {t['acceptance']:.3f} x contrast {t['contrast']:6.2f}"
                  f" x var {t['variance']:.3f}  beta {o[ro]['beta']:.2f}")
        for kv, b in o['by_k'].items():
            print(f"{'':60s}   k={kv}: n={b['n']:5d} a_ps {b['a_ps_geo']:.3f}  D_d/D_ps {b['D_ratio_geo']:.3f}  CU cost ratio {b['cost_ratio_cu_geo']:.2f}")
    if len(sys.argv) > 2:
        json.dump(res, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
