# Shot-cost study E4 report: readout floor and branch-weighted readouts on image choice (val for fitting, test std).
# Inputs: shotcost_branches.py npz (P, U: [pairs, pos/neg, h]; G, Q: [pairs, h]). Readouts, shots per DECISION:
#   native_cu : the model's own readout, 2 circuits (PS: ratio estimator over accepted shots; discard: all shots) = E0
#   native_O  : 1 circuit measuring the eigenbasis of O = |vp><vp| - |vn><vn| (min-variance unbiased single-shot margin
#               readout), aux qubits PS-selected (PS) or ignored (discard); rejected PS shots charged
#   wO / wCU  : measure the aux qubits too and weight every shot by w[h] (h = Hamming weight of the aux outcome);
#               w fitted on val (max expected accuracy at FIT_SHOTS per decision), evaluated on test std.
# usage: shotcost_e4_report.py <e4 dir> [out.json]
import glob, json, os, re, sys, collections
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

TARGETS = (0.70, 0.75, 0.80, 0.85)
GRID = np.unique(np.round(2 ** np.arange(1, 26.01, 0.125)).astype(np.int64))
FIT_SHOTS = (64, 512)


def stats_cu_native(P, U, discard):
    if discard:
        m = U.sum(-1); a = np.ones_like(m)
    else:
        a = P[..., 0]; m = U[..., 0] / np.maximum(a, 1e-300)
    D = m[:, 0] - m[:, 1]
    V = (m[:, 0] * (1 - m[:, 0])) / np.maximum(a[:, 0], 1e-300) + (m[:, 1] * (1 - m[:, 1])) / np.maximum(a[:, 1], 1e-300)
    return D, np.maximum(V, 1e-18), 2          # mean and per-shot variance (per circuit, both circuits at S shots), circuits


def stats_w(G, Q, w):
    mu = G @ w; var = Q @ (w * w) - mu * mu
    return mu, np.maximum(var, 1e-18), 1


def stats_wcu(U, w):
    mu = U @ w; sec = U @ (w * w)
    var = sec - mu * mu
    return mu[:, 0] - mu[:, 1], np.maximum(var[:, 0] + var[:, 1], 1e-18), 2


def curve(D, V, circ):
    # S shots per circuit -> decision uses circ * S shots; acc(S) = mean Phi(D / sqrt(V / S))
    return np.array([np.mean(norm.cdf(D * np.sqrt(s / V))) for s in GRID]), circ


def shots_to(acc, circ, t):
    i = np.argmax(acc >= t)
    if acc[i] < t: return None
    if i == 0: return float(circ * GRID[0])
    x0, x1 = np.log(GRID[i - 1]), np.log(GRID[i])
    return float(circ * np.exp(x0 + (t - acc[i - 1]) / (acc[i] - acc[i - 1]) * (x1 - x0)))


def summary(D, V, circ):
    acc, c = curve(D, V, circ)
    return {'exact_acc': float(np.mean((D > 0) + 0.5 * (D == 0))),
            'shots_per_decision': {str(t): shots_to(acc, c, t) for t in TARGETS},
            'acc_at': {str(int(c * s)): float(a) for s, a in zip(GRID, acc) if s in (16, 64, 256, 1024, 4096, 16384)}}


def fit_w(stat_fn, hmax, n_h, shots_per_decision, circ, w0s):
    # weights on h = 0..hmax (higher h unused), fitted to maximise the expected accuracy at a fixed shot budget
    s = shots_per_decision / circ
    def pad(w):
        out = np.zeros(n_h); out[:hmax + 1] = w; return out
    def loss(w):
        D, V, _ = stat_fn(pad(w))
        return -np.mean(norm.cdf(D * np.sqrt(s / V)))
    best = None
    for w0 in w0s:
        r = minimize(loss, w0[:hmax + 1], method='Nelder-Mead' if hmax < 1 else 'L-BFGS-B', options={'maxiter': 2000})
        if best is None or r.fun < best.fun: best = r
    return pad(best.x / np.max(np.abs(best.x))), -best.fun


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(f'{d}/*__val.npz'))
    res = {}
    for fv in files:
        ck = os.path.basename(fv)[:-len('__val.npz')]
        ft = f'{d}/{ck}__std.npz'
        if not os.path.exists(ft): continue
        meta = json.load(open(ft[:-4] + '.json')); discard = meta['discard']
        val, te = dict(np.load(fv)), dict(np.load(ft))
        n_h = val['P'].shape[-1]
        used = np.flatnonzero((val['P'].sum((0, 1)) > 0) | (te['P'].sum((0, 1)) > 0))
        hmax = int(used.max())
        native_w = np.zeros(n_h); native_w[0] = 1.0
        if discard: native_w[:] = 1.0
        row = {'discard': discard, 'text_n': meta['text_n'], 'hmax': hmax,
               'native_cu': summary(*stats_cu_native(te['P'], te['U'], discard)),
               'native_O': summary(*stats_w(te['G'], te['Q'], native_w))}
        ps_w = np.zeros(n_h); ps_w[0] = 1.0; ones = np.ones(n_h)
        row['ps_weights_O'] = summary(*stats_w(te['G'], te['Q'], ps_w))
        row['all_weights_O'] = summary(*stats_w(te['G'], te['Q'], ones))
        row['fits'] = {}
        for fs in FIT_SHOTS:
            starts = [ps_w, ones, 2.0 ** -np.arange(n_h), np.r_[1.0, -0.3 * np.ones(n_h - 1)]]
            for kind, fn, circ in (('wO', lambda w: stats_w(val['G'], val['Q'], w), 1), ('wCU', lambda w: stats_wcu(val['U'], w), 2)):
                w, vacc = fit_w(fn, hmax, n_h, fs, circ, starts)
                tfn = (lambda w: stats_w(te['G'], te['Q'], w)) if kind == 'wO' else (lambda w: stats_wcu(te['U'], w))
                row['fits'][f'{kind}@{fs}'] = {'w': [float(x) for x in w[:hmax + 1]], 'val_acc_at_fit': float(vacc), 'test': summary(*tfn(w))}
        res[ck] = row
        print(ck, 'done', flush=True)
    # table
    cols = ['native_cu', 'native_O', 'ps_weights_O', 'all_weights_O', 'wO@64', 'wO@512', 'wCU@64', 'wCU@512']
    get = lambda r, c: r[c] if c in r else r['fits'][c]['test']
    for t in ('0.75', '0.8'):
        print(f'\n== test std: shots per decision to reach acc {t} (exact acc in brackets)')
        print(f"{'checkpoint':40s} " + ' '.join(f'{c:>16s}' for c in cols))
        for ck, r in sorted(res.items(), key=lambda kv: (kv[1]['text_n'], kv[1]['discard'], kv[0])):
            f = lambda c: get(r, c)['shots_per_decision'][t]
            print(f"{ck[:40]:40s} " + ' '.join(f"{(f'{f(c):.0f}' if f(c) else '-') + f' ({get(r, c)['exact_acc']:.3f})':>16s}" for c in cols))
    print('\nfitted weights by Hamming weight h (wO@64):')
    for ck, r in sorted(res.items()):
        print(f"{ck[:40]:40s} " + ' '.join(f'{x:+.2f}' for x in r['fits']['wO@64']['w']))
    if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
