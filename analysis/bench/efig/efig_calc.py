# Energy-figure model (paper figure: QPU vs best classical energy per decision, log-depth rule).
# Port of the advantage page's model (tools/viz_sc/body_sc4.html: classical(), qpu(), breakEven()) to Python, plus IQM and the
# log-depth rule L = round(log2 nq). Reads adv.json (+ optional qpu_iqm_*.csv) and writes efig.json: the series the live page draws.
# Every point carries a status: measured | modelled | interpolated | extrapolated | upper (classical path bound).
import json, math, os, glob, csv, statistics, collections, datetime

H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(H, 'adv.json')))

# ---- constants (same as the page unless marked NEW) ----
N0 = 179                     # noiseless shots per decision at 10% margin error; shot accounting divides per-shot variance by shots x acceptance (1/p0 included)
K = D['reuse']['decisions'] / D['reuse']['shapes']   # decisions per circuit shape in the std test set (1833 / 25): one-off costs / K
PCPU = 20.0                  # W, one busy CPU core (path search, transpile, host share of a GPU run); assumed: no power counter on the nodes
CAL = dict(a=-0.187, eIBM=0.00331, eIonQ=0.004, e1IonQ=2e-4)   # Kingston CU fit ln F = a - eIBM n2q; IonQ Forte 99.6% / 99.98% assumed
IBM = dict(P=25e3, rep=250e-6, job=2 * 2 / 190)                # 25 kW, 250 us repetition delay, 2 s job overhead / 190 circuits, 2 circuits
IONQ = dict(P=5e3, t1=1.1e-4, t2=9e-4, shot=1.5e-3, circ=2 * 10.0)   # Knitter et al. 2026 (arXiv 2605.02798)
# NEW: IQM Garnet-class (20 q Garnet published figures; routing on the 54 q Aphrodite lattice).
KING_MED2Q, KING_MEDRO = 1.80e-3, 9.52e-3     # FakeKingston snapshot medians (2q CZ error, readout error)
IQM = dict(P=24e3,                             # W, typical mean total electrical power, IQM Radiance spec (iqm.tech/products/iqm-radiance)
           e2=5e-3, e1=9e-4, ro=3e-2,          # median CZ, 1q, simultaneous readout errors (IQM Garnet benchmarks, arXiv 2408.12433)
           shot=100 * 10 * 100 * 5 / 2600 / (100 * 10 * 100))   # s per shot incl. overheads from CLOPS 2600 (M=100, K=10, S=100, D=5): 1.92 ms
IQM['eff'] = CAL['eIBM'] * IQM['e2'] / KING_MED2Q              # effective per-CZ log-fidelity, scaled from the Kingston fit by the median-error ratio
IQM['a'] = CAL['a'] * IQM['ro'] / KING_MEDRO                   # intercept (SPAM) scaled by the readout-error ratio
NOISE = [1, 0.1, 0.01]       # multiplier on the two-qubit (and IonQ one-qubit) gate error only; readout/SPAM intercept a unchanged

rows_all = D['rows']
def by(L, nq): return next((r for r in rows_all if r['layers'] == L and r['nq'] == nq), None)

# routing: Kingston routed 2q = logical 2q x median overhead at that depth (page default 'fit')
ov = collections.defaultdict(list)
for r in rows_all:
    if r.get('ibm_n2q') and r.get('n2q_logical'): ov[r['layers']].append(r['ibm_n2q'] / r['n2q_logical'])
ROUTE = {L: statistics.median(v) for L, v in ov.items()}
def route(L):   # median overhead at L; between measured depths, linear in L (marked interpolated with the row)
    if L in ROUTE: return ROUTE[L]
    lo = max(x for x in ROUTE if x < L); hi = min(x for x in ROUTE if x > L)
    return ROUTE[lo] + (ROUTE[hi] - ROUTE[lo]) * (L - lo) / (hi - lo)

def etafit(key):
    v = []
    for r in rows_all:
        for eng, lf in (('tn', r.get('greedy_ps_lf')), ('tn_rho', r.get('greedy_rho_lf'))):
            g = (r.get('gpu') or {}).get(eng)
            if g and lf is not None and lf >= 10.5 and g.get(key): v.append(10 ** lf / g[key])
    return statistics.median(v)
ETA, RATE = etafit('J'), etafit('s')     # FLOP/J and FLOP/s of exact TN on the L4 (compute-bound runs)

def tnc(meas, meas_s, paths, lf_any, lf_greedy, mult, gps):
    # classical TN energy for one network kind: per shape the path with the lowest (mult x run + search / K); geometric mean over shapes
    if meas is None and lf_any is None: return None
    over = max(1.0, meas - (10 ** lf_greedy / ETA if lf_greedy is not None else 0)) if meas is not None else 1.0
    def run(c):
        if meas is not None: return meas if c['o'] == 'greedy' else min(meas, max(10 ** c['lf'] / ETA, over))
        return max(1.0, 10 ** c['lf'] / ETA)
    shapes = paths if paths else [[{'o': 'greedy', 'lf': lf_any, 's': gps or 0}]]
    per = []
    for cs in shapes:
        best = None
        for c in cs:
            ru = run(c); t_run = (meas_s if (meas is not None and ru == meas) else max(0.016, 10 ** c['lf'] / RATE))
            host = mult * t_run * PCPU                       # host share of the GPU run (one busy core)
            tot = mult * ru + host + c['s'] * PCPU / K
            if best is None or tot < best['tot']: best = dict(tot=tot, meas=meas is not None and ru == meas, o=c['o'], lf=c['lf'])
        per.append(best)
    gm = math.exp(sum(math.log(p['tot']) for p in per) / len(per))
    return dict(J=gm, measured=all(p['meas'] for p in per), by=collections.Counter(p['o'] for p in per).most_common(1)[0][0])

def classical(r):
    g = r.get('gpu') or {}
    out = {}
    sv = g.get('cusv_rho')
    if sv: out['statevector'] = dict(J=sv['J'] + sv['s'] * PCPU, status='measured')
    lfP = r.get('best_ps_lf', r.get('greedy_ps_lf')); lfR = r.get('best_rho_lf', r.get('greedy_rho_lf'))
    gt, gr = g.get('tn'), g.get('tn_rho')
    T = tnc(gr['J'] if gr else None, gr['s'] if gr else None, r.get('paths_rho'), lfR, r.get('greedy_rho_lf'), 1, r.get('greedy_path_s'))
    A = tnc(gt['J'] if gt else None, gt['s'] if gt else None, r.get('paths_ps'), lfP, r.get('greedy_ps_lf'), N0, r.get('greedy_path_s'))
    if T: out['exact TN'] = dict(J=T['J'], status='measured' if T['measured'] else 'upper')
    if A: out['TN sampler'] = dict(J=A['J'], status='measured' if A['measured'] else 'upper')
    M = [m for m in ((r.get('mps') or {}).get('rho') or []) if m.get('J') and m.get('mrel') is not None and m['mrel'] <= 0.1
         and m.get('ref') == 'exact' and (m.get('agree') is None or m['agree'] >= 0.95)]
    if M: out['MPS'] = dict(J=min(m['J'] + m['s'] * PCPU for m in M), status='measured')
    if not out: return None
    k = min(out, key=lambda x: out[x]['J'])
    return dict(best=out[k]['J'], by=k, status=out[k]['status'], methods=out)

def qpu_counts(r):
    L = r['layers']; f = route(L) * r['n2q_logical'] / r['ibm_n2q']
    return dict(ibm_n2q=r['ibm_n2q'] * f, ibm_dur=r['ibm_duration_s'] * f, ibm_tt=r.get('ibm_tt') or 0,
                ionq_n2q=r['ionq_n2q'], ionq_n1q=r['ionq_n1q'], ionq_tt=r.get('ionq_tt') or 0,
                iqm_n2q=r.get('iqm_n2q'), iqm_dur=r.get('iqm_dur', 0.0), iqm_tt=r.get('iqm_tt', r.get('ibm_tt') or 0), iqm_src=r.get('iqm_src'))

def e_ibm(q, m):
    F = math.exp(CAL['a'] - CAL['eIBM'] * m * q['ibm_n2q']); N = N0 / F ** 2
    return (N * (q['ibm_dur'] + IBM['rep']) + IBM['job']) * IBM['P'] + q['ibm_tt'] * PCPU / K
def e_ionq(q, m):
    F = math.exp(CAL['a'] - (CAL['eIonQ'] * q['ionq_n2q'] + CAL['e1IonQ'] * q['ionq_n1q']) * m); N = N0 / F ** 2
    return (N * (q['ionq_n1q'] * IONQ['t1'] + q['ionq_n2q'] * IONQ['t2'] + IONQ['shot']) + IONQ['circ']) * IONQ['P'] + q['ionq_tt'] * PCPU / K
def e_iqm(q, m):
    if q['iqm_n2q'] is None: return None
    F = math.exp(IQM['a'] - IQM['eff'] * m * q['iqm_n2q']); N = N0 / F ** 2
    return N * (IQM['shot'] + q['iqm_dur']) * IQM['P'] + q['iqm_tt'] * PCPU / K
EF = dict(IBM=e_ibm, IonQ=e_ionq, IQM=e_iqm)
TODAY = dict(IBM=CAL['eIBM'], IonQ=CAL['eIonQ'], IQM=IQM['eff'])   # effective per-2q-gate error the model multiplies by m

def breakeven(q, Ec, f):
    # largest m (error multiplier) at which the QPU energy equals Ec; None if even an error-free QPU costs more
    if Ec is None or f(q, 1e-9) is None or f(q, 1e-9) >= Ec: return None
    if f(q, 1.0) <= Ec: lo, hi = 1.0, 1.0
    else: lo, hi = 1e-9, 1.0
    if f(q, 1.0) <= Ec:   # already wins today: push up
        hi = 1.0
        while f(q, hi) <= Ec and hi < 1e6: hi *= 2
        lo = hi / 2
    for _ in range(80):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
        if f(q, mid) <= Ec: lo = mid
        else: hi = mid
    return lo

# ---- IQM transpile stats (when landed): routed CZ, critical path, transpile s; beyond 54 q: logical x median overhead per depth ----
iqm = collections.defaultdict(list)
for fn in glob.glob(os.path.join(H, 'qpu_iqm_*.csv')):
    for r in csv.DictReader(open(fn)):
        if r.get('circ') == 'O': continue
        iqm[(int(r['layers']), int(float(r['nq'])))].append(r)
iqm_ov = collections.defaultdict(list)
for (L, nq), v in iqm.items():
    for r in v:
        if r['fits'] == '1' and float(r['logical_n2q']) > 0: iqm_ov[L].append(float(r['iqm_n2q']) / float(r['logical_n2q']))
IQM_ROUTE = {L: statistics.median(v) for L, v in iqm_ov.items()}
def iqm_overhead(L):
    if L in IQM_ROUTE: return IQM_ROUTE[L], 'IQM lattice, median at this depth'
    if IQM_ROUTE: Lc = min(IQM_ROUTE, key=lambda x: abs(x - L)); return IQM_ROUTE[Lc], f'IQM lattice, median at {Lc} layers'
    return route(L), 'placeholder: Kingston heavy-hex overhead (IQM transpile pending)'

def attach_iqm(r):
    v = iqm.get((r['layers'], r['nq']), [])
    f, src = iqm_overhead(r['layers'])
    fit = [x for x in v if x['fits'] == '1']
    r['iqm_n2q'] = r['n2q_logical'] * f; r['iqm_src'] = src
    r['iqm_dur'] = statistics.mean(float(x['iqm_crit_s']) for x in fit) if fit else 0.0
    if fit: r['iqm_tt'] = statistics.mean(float(x['iqm_transpile_s']) for x in fit)
    return r

# ---- depth rule ----
def L_of(nq): return int(round(math.log2(nq)))
SIZES = sorted({r['nq'] for r in rows_all if r['e'] == 3 and r['layers'] in (4, 8) and r['nq'] >= 16})

def lerp_row(nq, L):
    # a row at depth L from the measured 4- and 8-layer rows of that size: gate counts, durations, transpile times linear in L,
    # best-path log10 FLOP linear in L (interpolation of an upper bound). Marked 'interpolated' until the real L-layer circuit lands.
    a, b = by(4, nq), by(8, nq)
    if not a or not b: return None
    t = (L - 4) / 4
    li = lambda x, y: None if x is None or y is None else x + (y - x) * t
    r = dict(layers=L, nq=nq, e=3, n=a['n'], interp=True)
    for k in ('n2q_logical', 'ibm_n2q', 'ibm_duration_s', 'ibm_tt', 'ionq_n2q', 'ionq_n1q', 'ionq_tt', 'greedy_path_s'):
        r[k] = li(a.get(k), b.get(k))
    for kind in ('ps', 'rho'):
        la, lb = a.get(f'best_{kind}_lf'), b.get(f'best_{kind}_lf')
        r[f'best_{kind}_lf'] = li(la, lb); r[f'greedy_{kind}_lf'] = None
        sa = min((c['s'] for cs in (a.get(f'paths_{kind}') or []) for c in cs if c['o'] == a.get(f'best_{kind}_by')), default=600)
        r[f'paths_{kind}'] = [[{'o': 'interp', 'lf': r[f'best_{kind}_lf'], 's': sa}]] if r[f'best_{kind}_lf'] is not None else None
    return r

def row_at(nq, L):
    r = by(L, nq)
    if r and (r.get('paths_ps') or r.get('gpu') or L == 2): return dict(r)
    return lerp_row(nq, L) if L not in (4, 8) else (dict(r) if r else None)

def point(r):
    attach_iqm(r)
    c = classical(r)
    if c is None or not r.get('ibm_n2q'): return None
    q = qpu_counts(r)
    st = 'interpolated' if r.get('interp') else c['status']
    p = dict(nq=r['nq'], L=r['layers'], n=r['n'], classical=c['best'], cls_by=c['by'], cls_status=st,
             methods={k: dict(J=v['J'], status='interpolated' if r.get('interp') else v['status']) for k, v in c['methods'].items()},
             iqm_src=q['iqm_src'], qpu={}, be={})
    for b, f in EF.items():
        p['qpu'][b] = {str(m): f(q, m) for m in NOISE}
        m = breakeven(q, c['best'], f)
        p['be'][b] = None if m is None else TODAY[b] * m
    p['qpu_status'] = 'interpolated' if r.get('interp') else 'modelled'
    return p

series = {}
for name, rule in (('log2', L_of), ('L4', lambda n: 4), ('L8', lambda n: 8)):
    pts = []
    for nq in SIZES:
        r = row_at(nq, rule(nq))
        if r is None: continue
        p = point(r)
        if p: pts.append(p)
    series[name] = pts

# statevector extrapolation (cuStateVec, density matrix as in the page frontier): fit log10 J vs nq over the 3 largest measured sizes per depth
sv_ext = {}
for L in (2, 4, 8):
    m = [(r['nq'], (r['gpu']['cusv_rho']['J'])) for r in rows_all if r['layers'] == L and (r.get('gpu') or {}).get('cusv_rho')]
    if len(m) >= 3:
        m = sorted(m)[-3:]; xs = [x for x, _ in m]; ys = [math.log10(y) for _, y in m]
        xb, yb = sum(xs) / 3, sum(ys) / 3; s = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
        sv_ext[L] = dict(slope=s, icpt=yb - s * xb, from_nq=max(xs))

meta = dict(
    updated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    N0=N0, K=round(K, 1), PCPU=PCPU, ETA=ETA, RATE=RATE, route={str(k): round(v, 2) for k, v in ROUTE.items()},
    iqm_route={str(k): round(v, 2) for k, v in IQM_ROUTE.items()}, today=TODAY, noise=NOISE,
    consts=dict(IBM=IBM, IonQ=IONQ, IQM={k: v for k, v in IQM.items()}, CAL=CAL, KING_MED2Q=KING_MED2Q, KING_MEDRO=KING_MEDRO),
    sv_ext=sv_ext,
    pending=[])
json.dump(dict(meta=meta, series=series), open(os.path.join(H, 'efig.json'), 'w'), indent=0)
if __name__ == '__main__':
    for name, pts in series.items():
        print('==', name)
        for p in pts:
            print(f"{p['nq']:4d}q L{p['L']} cls {p['classical']:.3g} J ({p['cls_by']}, {p['cls_status']})  " +
                  '  '.join(f"{b} {p['qpu'][b]['1']:.3g}/{p['qpu'][b]['0.01']:.3g} be {('%.2g' % p['be'][b]) if p['be'][b] else '-'}" for b in EF))
    print('ETA %.3g FLOP/J RATE %.3g FLOP/s route %s IQM eff %.3g a %.3g shot %.4g s' % (ETA, RATE, ROUTE, IQM['eff'], IQM['a'], IQM['shot']))
