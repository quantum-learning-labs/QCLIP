# Aggregates the sc4 scaling sweep (2 layers; output qubits e at n=1, qubits per wire n at e=3) into one JSON for the figures page.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/viz_sc4.py results/shotcost results/shotcost/viz/viz_sc4.json
# Inputs (missing ones are skipped): e4_sc4/<ckpt>__{std,swap_full}.npz (shotcost_branches.py), viz_sc4/gates.csv (g_count.py),
# viz_sc4/qpu_stats.csv + gpu_models.csv (v3_energy.py --no_scaling), e6_sc4/<ckpt>__<profile>.npz (shotcost_noise.py).
import glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import norm

# helpers copied from viz_data.py (importing it would run that script)
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


R, OUT = sys.argv[1], sys.argv[2]
PAT = re.compile(r'hp_sc4_(nodiscard|discard)_e(\d+)_n(\d+)_tl2_(qlr3e-3|pw0|pw0_ws)(?:_r\d+)?_s(\d+)')
METH = {'qlr3e-3': 'PS', 'pw0': 'plain', 'pw0_ws': 'warm'}


def parse(ck):
    m = PAT.fullmatch(ck)
    return None if m is None else dict(e=int(m[2]), n=int(m[3]), method=METH[m[4]], seed=int(m[5]))


def rel_target(acc):                      # 95% of the model's own edge over chance: reachable by every model
    return 0.5 + 0.95 * (acc - 0.5)


rows = []
def latest(pattern_dirs, name_glob):           # one file per basename; later dirs (reruns) fill gaps only
    seen = {}
    for d in pattern_dirs:
        for f in sorted(glob.glob(f'{R}/{d}/{name_glob}')):
            seen.setdefault(os.path.basename(f), f)
    return [seen[k] for k in sorted(seen)]


def read_csvs(name):
    fs = sorted(glob.glob(f'{R}/viz_sc4*/{name}'))
    dfs = [d for d in (pd.read_csv(f) for f in fs) if len(d)]            # header-only placeholders would make columns object-typed
    return pd.concat(dfs, ignore_index=True) if dfs else None


for f in latest(('e4_sc4', 'e4_sc4_rerun'), 'hp_sc4_*__*.npz'):
    ck, tset = os.path.basename(f)[:-4].split('__')
    p = parse(ck)
    if p is None or tset not in ('std', 'swap_full'): continue
    x = np.load(f); disc = p['method'] != 'PS'
    ro = {'cu': cu_stats(x['P'], x['U'], disc), 'O': o_stats(x['P'], x['G'], x['Q'], disc)}
    acc = float(np.mean(ro['cu'][0] > 0))
    r = dict(p, ckpt=ck, set=tset, pairs=int(len(x['P'])), exact=acc, nq=float((x['k'][:, 0] + p['e']).mean()),
             k=float(x['k'][:, 0].mean()), rel_target=rel_target(acc))
    if not disc: r['acceptance'] = float(np.exp(np.mean(np.log(np.maximum(x['P'][:, :, 0], 1e-300)))))
    for kk, (D, V, c) in ro.items():
        for b in BUDGETS: r[f'{kk}@{b}'] = exp_acc(D, V, c, b)
        r[f'{kk}_to_abs'] = shots_to(D, V, c, 0.75 if tset == 'std' else 0.80)
        r[f'{kk}_to_rel'] = shots_to(D, V, c, rel_target(acc))
        ok = D > 0                                   # geometric-mean shots per correctly ranked pair (cost to resolve its sign at 1 sd)
        r[f'{kk}_geo_shots'] = float(np.exp(np.mean(np.log(c * V[ok] / D[ok] ** 2)))) if ok.any() else None
    rows.append(r)
df = pd.DataFrame(rows)
out = {'per_seed': rows, 'agg': [], 'gates': [], 'qpu': [], 'gpu': [], 'noisy': [], 'notes': {'cost_model': COST, 'budgets': BUDGETS}}

# gate counts (sentence-only: one count per config, from the PS seed-0 checkpoint)
g = read_csvs('gates.csv')
gates = None
if g is not None:
    g = pd.concat([g, pd.DataFrame([parse(c) or {} for c in g.ckpt])], axis=1).dropna(subset=['e'])
    gates = g.groupby(['e', 'n']).agg(cx_cu=('cx_cu_pos', 'mean'), cx_O=('cx_O', 'mean'), nat2q_cu=('nat2q_cu_pos', 'mean'),
                                      nat2q_O=('nat2q_O', 'mean'), gate_nq=('nq', 'mean')).reset_index()
    out['gates'] = gates.to_dict('records')

# per (e, n, method, set): mean and standard error over seeds
if len(df):
    num = [c for c in df.columns if c not in ('e', 'n', 'method', 'seed', 'ckpt', 'set')]
    for (e, n, meth, tset), q in df.groupby(['e', 'n', 'method', 'set']):
        a = {'e': int(e), 'n': int(n), 'method': meth, 'set': tset, 'seeds': len(q)}
        for c in num:
            v = q[c].dropna().astype(float)
            if c.endswith(('_to_abs', '_to_rel', '_geo_shots')):            # shot counts: geometric mean, SE on the log scale
                lv = np.log(v[v > 0])
                a[c] = float(np.exp(lv.mean())) if len(lv) else None
                a[c + '_logse'] = float(lv.std(ddof=1) / np.sqrt(len(lv))) if len(lv) > 1 else None
                if c.endswith('_to_abs'): a[c + '_reached'] = int(len(lv))
            else:
                a[c] = float(v.mean()) if len(v) else None
                a[c + '_se'] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        if gates is not None:
            gr = gates[(gates.e == e) & (gates.n == n)]
            if len(gr): a.update({k: float(gr.iloc[0][k]) for k in ('cx_cu', 'cx_O', 'nat2q_cu', 'nat2q_O')})
        out['agg'].append(a)

# QPU energy per decision: shots to target x time per shot + per-circuit overhead, x machine power
qs = read_csvs('qpu_stats.csv')
if qs is not None and out['agg']:
    qs = pd.concat([qs, pd.DataFrame([parse(c) or {} for c in qs.ckpt])], axis=1).dropna(subset=['e'])
    st = qs.groupby(['e', 'n', 'kind']).mean(numeric_only=True)
    for a in out['agg']:
        for ro, kind, circ in (('cu', 'cu', 2), ('O', 'O', 1)):
            if (a['e'], a['n'], kind) not in st.index: continue
            s = st.loc[(a['e'], a['n'], kind)]
            t_ibm = s.ibm_duration_s + COST['ibm']['rep']; o_ibm = COST['ibm']['job'] / COST['ibm']['per_job']
            t_ion = s.ionq_n1q * COST['ionq']['t1'] + s.ionq_n2q * COST['ionq']['t2'] + COST['ionq']['shot']; o_ion = COST['ionq']['circ']
            for tgt in ('abs', 'rel'):
                S = a.get(f'{ro}_to_{tgt}')
                if S is None: continue
                out['qpu'].append({'e': a['e'], 'n': a['n'], 'method': a['method'], 'set': a['set'], 'readout': ro, 'target': tgt, 'nq': a['nq'],
                    'shots_per_decision': S, 'logse': a.get(f'{ro}_to_{tgt}_logse'),
                    'ibm_J': (S * t_ibm + circ * o_ibm) * COST['ibm']['W'], 'ionq_J': (S * t_ion + circ * o_ion) * COST['ionq']['W'],
                    'ibm_J_shots_only': S * t_ibm * COST['ibm']['W'], 'ionq_J_shots_only': S * t_ion * COST['ionq']['W'],
                    'ibm_shot_s': float(t_ibm), 'ionq_shot_s': float(t_ion), 'ibm_n2q': float(s.ibm_n2q), 'ionq_n2q': float(s.ionq_n2q)})
g = read_csvs('gpu_models.csv')
if g is not None:
    out['gpu'] = [dict(r, **(parse(r['ckpt']) or {})) for r in g.to_dict('records')]

# noisy runs
for f in latest(('e6_sc4', 'e6_sc4_rerun'), '*.npz'):
    ck, prof = os.path.basename(f)[:-4].split('__')
    p = parse(ck)
    if p is None: continue
    x = np.load(f); disc = p['method'] != 'PS'
    r = dict(p, ckpt=ck, profile=prof, pairs=int(len(x['P'])))
    for kk, (D, V, c) in {'cu': cu_stats(x['P'], x['U'], disc), 'O': o_stats(x['P'], x['G'], x['Q'], disc)}.items():
        if np.isnan(D).all(): continue                 # O readout skipped above o_max output qubits
        r[f'{kk}_acc'] = float(np.mean(D > 0)); r[f'{kk}_acc_se'] = float(np.sqrt(r[f'{kk}_acc'] * (1 - r[f'{kk}_acc']) / len(D)))
        for b in BUDGETS: r[f'{kk}@{b}'] = exp_acc(D, V, c, b)
    out['noisy'].append(r)
json.dump(out, open(OUT, 'w'), indent=1, default=float)
print({k: len(v) for k, v in out.items() if isinstance(v, list)})
