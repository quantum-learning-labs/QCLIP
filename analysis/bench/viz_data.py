# Aggregates the shot-cost results into one small JSON for the V1-V4 plots (missing inputs are skipped).
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/viz_data.py results/shotcost results/shotcost/viz/viz_data.json
import glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import norm
R, OUT = sys.argv[1], sys.argv[2]
SIZES = {1: {'PS': 'hp_hp2_nodiscard_tl3_qlr3e-3', 'plain': 'hp_hp2_discard_tl3_pw0', 'warm': 'hp_do1_discard_tl3_pw0_initnd', 'distill': 'hp_kd1_discard_kd_ws'},
         2: {'PS': 'hp_sc2_nodiscard_n2_tl3_qlr3e-3', 'plain': 'hp_sc2_discard_n2_tl3_pw0', 'warm': 'hp_sc2_discard_n2_tl3_pw0_ws', 'distill': 'hp_sc2_discard_n2_tl3_pw0_kd'},
         3: {'PS': 'hp_sc3_nodiscard_n3_tl3_qlr3e-3', 'plain': 'hp_sc3_discard_n3_tl3_pw0', 'warm': 'hp_sc3_discard_n3_tl3_pw0_ws'}}
BUDGETS = (256, 1024, 4096)          # shots per decision
COST = {'ibm': dict(rep=250e-6, job=2.0, per_job=190, W=25e3), 'ionq': dict(t1=1.1e-4, t2=9e-4, shot=1.5e-3, circ=10.0, W=5e3)}


def cu_stats(P, U, discard):
    if discard: m = U.sum(-1); a = np.ones_like(m)
    else: a = P[..., 0]; m = U[..., 0] / np.maximum(a, 1e-300)
    D = m[:, 0] - m[:, 1]
    V = m[:, 0] * (1 - m[:, 0]) / np.maximum(a[:, 0], 1e-300) + m[:, 1] * (1 - m[:, 1]) / np.maximum(a[:, 1], 1e-300)
    return D, np.maximum(V, 1e-18), 2


def o_stats(P, G, Q, discard):
    if discard: D = G.sum(-1); V = Q.sum(-1) - D ** 2
    else:
        a = P[:, 0, 0]; D = G[:, 0] / np.maximum(a, 1e-300); V = (Q[:, 0] / np.maximum(a, 1e-300) - D ** 2) / np.maximum(a, 1e-300)
    return D, np.maximum(V, 1e-18), 1


def exp_acc(D, V, circ, shots_per_decision):
    return float(np.mean(norm.cdf(D * np.sqrt((shots_per_decision / circ) / V))))


def shots_to(D, V, circ, t):
    grid = np.unique(np.round(2 ** np.arange(1, 26.01, 0.125)).astype(np.int64))
    acc = np.array([np.mean(norm.cdf(D * np.sqrt(s / V))) for s in grid])
    i = np.argmax(acc >= t)
    if acc[i] < t: return None
    if i == 0: return float(circ * grid[0])
    x0, x1 = np.log(grid[i - 1]), np.log(grid[i])
    return float(circ * np.exp(x0 + (t - acc[i - 1]) / (acc[i] - acc[i - 1]) * (x1 - x0)))


def readouts(x, discard):
    return {'cu': cu_stats(x['P'], x['U'], discard), 'O': o_stats(x['P'], x['G'], x['Q'], discard)}


out = {'v1': [], 'v1_bins': [], 'v2': [], 'v3_qpu': [], 'v3_gpu_models': [], 'v3_gpu_scaling': [], 'v4': [], 'noisy': [], 'notes': {}}
# ---------- V1: exact accuracy and accuracy at fixed budgets vs qubits (std), all seeds ----------
for n, fams in SIZES.items():
    for meth, fam in fams.items():
        rows = []
        for f in sorted(glob.glob(f'{R}/e4/{fam}_s*__std.npz')):
            if not re.fullmatch(re.escape(fam) + r'_s\d+__std\.npz', os.path.basename(f)): continue
            x = np.load(f); disc = meth != 'PS'
            ro = readouts(x, disc)
            nq = x['k'][:, 0] + 3
            r = {'exact': float(np.mean(ro['cu'][0] > 0)), 'nq': float(nq.mean())}
            for k, (D, V, c) in ro.items():
                for b in BUDGETS: r[f'{k}@{b}'] = exp_acc(D, V, c, b)
                r[f'{k}_to75'] = shots_to(D, V, c, 0.75)
            rows.append(r)
            # per-pair bins by text qubits (exact correctness, CU expected acc at 1024)
            D, V, c = ro['cu']
            for q in np.unique(nq):
                sel = nq == q
                out['v1_bins'].append({'n': n, 'method': meth, 'seed_file': os.path.basename(f), 'nq': int(q), 'pairs': int(sel.sum()),
                                       'exact': float(np.mean(D[sel] > 0)), 'cu@1024': float(np.mean(norm.cdf(D[sel] * np.sqrt((1024 / c) / V[sel]))))})
        if not rows: continue
        agg = {'n': n, 'method': meth, 'seeds': len(rows), 'nq': rows[0]['nq']}
        for key in rows[0]:
            if key == 'nq': continue
            v = [r[key] for r in rows if r[key] is not None]
            agg[key] = float(np.mean(v)) if v else None; agg[key + '_sd'] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            if key.endswith('_to75'): agg[key + '_reached'] = len(v)
        out['v1'].append(agg)
# per-bin pooling over seeds
b = pd.DataFrame(out['v1_bins'])
if len(b):
    b = b.groupby(['n', 'method', 'nq']).apply(lambda g: pd.Series({'pairs': int(g.pairs.sum() / g.seed_file.nunique()),
        'exact': np.average(g.exact, weights=g.pairs), 'cu@1024': np.average(g['cu@1024'], weights=g.pairs), 'seeds': g.seed_file.nunique()}), include_groups=False).reset_index()
    out['v1_bins'] = b[b.pairs >= 20].to_dict('records')
# ---------- noisy runs (E6 / V4): large-shot accuracy and budgets, per profile ----------
for f in sorted(glob.glob(f'{R}/e6/*.npz')):
    ck, prof = os.path.basename(f)[:-4].split('__')
    meta = json.load(open(f[:-4] + '.json')); x = np.load(f)
    ro = readouts(x, meta['discard'])
    r = {'ckpt': ck, 'text_n': meta['text_n'], 'discard': meta['discard'], 'profile': prof, 'pairs': meta['pairs']}
    for k, (D, V, c) in ro.items():
        r[f'{k}_acc'] = float(np.mean(D > 0)); r[f'{k}_margin'] = float(np.median(np.abs(D)))
        for bb in BUDGETS: r[f'{k}@{bb}'] = exp_acc(D, V, c, bb)
        r[f'{k}_to75'] = shots_to(D, V, c, 0.75)
    out['noisy'].append(r)
# ---------- V2: accuracy vs 2q gates (pairs joined by size and dataset index) ----------
gp = f'{R}/viz/gates.csv'
if os.path.exists(gp):
    g = pd.read_csv(gp)
    g = g[g.ckpt.str.contains('nodiscard')]                      # gate counts depend on the sentence, not the trained angles
    for n, fams in SIZES.items():
        gn = g[g.text_n == n].set_index('idx')
        for meth, fam in fams.items():
            for f in sorted(glob.glob(f'{R}/e4/{fam}_s*__std.npz')):
                if not re.fullmatch(re.escape(fam) + r'_s\d+__std\.npz', os.path.basename(f)): continue
                x = np.load(f); idx = x['idx'] if 'idx' in x.files else np.arange(len(x['P']))
                D, V, c = readouts(x, meth != 'PS')['cu']
                keep = np.isin(idx, gn.index)
                cx = gn.loc[idx[keep], 'cx_cu_pos'].to_numpy()
                for lo, hi in zip(range(0, 200, 10), range(10, 210, 10)):
                    sel = (cx >= lo) & (cx < hi)
                    if sel.sum() == 0: continue
                    out['v2'].append({'n': n, 'method': meth, 'file': os.path.basename(f), 'cx_lo': lo, 'pairs': int(sel.sum()),
                                      'exact': float(np.mean(D[keep][sel] > 0)), 'cu@1024': float(np.mean(norm.cdf(D[keep][sel] * np.sqrt(512 / V[keep][sel]))))})
    v = pd.DataFrame(out['v2'])
    if len(v):
        v = v.groupby(['n', 'method', 'cx_lo']).apply(lambda q: pd.Series({'pairs': int(q.pairs.sum() / q.file.nunique()),
            'exact': np.average(q.exact, weights=q.pairs), 'cu@1024': np.average(q['cu@1024'], weights=q.pairs)}), include_groups=False).reset_index()
        out['v2'] = v[v.pairs >= 20].to_dict('records')
    # mean Kingston-native 2q gates of each noise run's pairs (counts vary across sizes, barely within one)
    for r in out['noisy']:
        x = np.load(f"{R}/e6/{r['ckpt']}__{r['profile']}.npz")
        gn = g[g.text_n == r['text_n']].set_index('idx')
        nat = gn.reindex(x['idx'])['nat2q_cu_pos'].to_numpy(dtype=float)
        r['nat2q_mean'] = float(np.nanmean(nat)) if np.isfinite(nat).any() else None
        r['nat2q_pairs'] = int(np.isfinite(nat).sum())
# ---------- V3: QPU energy per decision from device stats x shots to target; GPU measurements ----------
qp = f'{R}/viz/qpu_stats.csv'
e0 = json.load(open(f'{R}/e0_report_v3.json'))['per_ckpt'] if os.path.exists(f'{R}/e0_report_v3.json') else {}
if os.path.exists(qp):
    q = pd.read_csv(qp).groupby(['text_n', 'kind']).mean(numeric_only=True)
    for n, fams in SIZES.items():
        if (n, 'cu') not in q.index: continue
        st = q.loc[(n, 'cu')]
        t_ibm = st.ibm_duration_s + COST['ibm']['rep']; o_ibm = COST['ibm']['job'] / COST['ibm']['per_job']
        t_ion = st.ionq_n1q * COST['ionq']['t1'] + st.ionq_n2q * COST['ionq']['t2'] + COST['ionq']['shot']; o_ion = COST['ionq']['circ']
        for meth, fam in fams.items():
            for tset, tgt in (('swap_full', '0.8'), ('std', '0.75')):
                s = [r['compute_uncompute']['shots_per_decision'][tgt] for k, r in e0.items()
                     if k.split('__')[1] == tset and re.fullmatch(re.escape(fam) + r'_s\d+', k.split('__')[0])]
                s = [x for x in s if x is not None]
                if not s: continue
                S = float(np.median(s))                         # shots per decision (2 CU circuits)
                out['v3_qpu'].append({'n': n, 'method': meth, 'set': tset, 'target': float(tgt), 'seeds_reaching': len(s), 'nq': float(st.nq),
                    'shots_per_decision': S, 'ibm_J': (S * t_ibm + 2 * o_ibm) * COST['ibm']['W'], 'ionq_J': (S * t_ion + 2 * o_ion) * COST['ionq']['W'],
                    'ibm_J_shots_only': S * t_ibm * COST['ibm']['W'], 'ionq_J_shots_only': S * t_ion * COST['ionq']['W'],
                    'ibm_shot_s': float(t_ibm), 'ionq_shot_s': float(t_ion)})
for name, key in (('gpu_models.csv', 'v3_gpu_models'), ('gpu_scaling.csv', 'v3_gpu_scaling')):
    if os.path.exists(f'{R}/viz/{name}'): out[key] = pd.read_csv(f'{R}/viz/{name}').to_dict('records')
out['notes'] = {'clip_J_per_decision': 0.36, 'clip_note': 'CLIP ViT-B/32 zero-shot, one decision (2 captions), RTX 3090 Ti NVML (bench_b2/classical2)',
                'cost_model': COST, 'budgets': BUDGETS}
json.dump(out, open(OUT, 'w'), indent=1, default=float)
print({k: len(v) for k, v in out.items() if isinstance(v, list)})
