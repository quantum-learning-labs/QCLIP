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
import random
MC = None   # set to a dict during Monte Carlo draws (see the uncertainty section at the end)
def by(L, nq): return next((r for r in rows_all if r['layers'] == L and r['nq'] == nq), None)

# routing: Kingston routed 2q = logical 2q x median overhead at that depth (page default 'fit')
ov = collections.defaultdict(list)
for r in rows_all:
    if r.get('ibm_n2q') and r.get('n2q_logical'): ov[r['layers']].append(r['ibm_n2q'] / r['n2q_logical'])
ROUTE = {L: statistics.median(v) for L, v in ov.items()}
def ov_rel(L):   # individual-circuit Kingston overheads at (or nearest to) depth L, relative to their median
    Lc = L if L in ov else min(ov, key=lambda x: abs(x - L)); med = statistics.median(ov[Lc]); return [v / med for v in ov[Lc]]
def iqm_rel(L):
    Lc = L if L in iqm_ov else min(iqm_ov, key=lambda x: abs(x - L)); med = statistics.median(iqm_ov[Lc]); return [v / med for v in iqm_ov[Lc]]
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

GREEDY = ('greedy', 'greedy-interp')
# Path-quality model (one fit for all sizes and depths): how much the last search stage (cotengra, 600 s) improved on the stage before it
# (random-greedy-128), against the size of the best path (log10 FLOP). Taken as the scale of a further improvement a better search could
# still find. Cotengra failures (gain < -0.5 decades) are excluded.
_G = []
for _r in rows_all:
    for _k in ('ps', 'rho'):
        for _cs in (_r.get(f'paths_{_k}') or []):
            _bo = {}
            for _c in _cs:
                if _c.get('lf') is not None: _bo[_c['o']] = min(_bo.get(_c['o'], 1e9), _c['lf'])
            if 'random-greedy-128' in _bo and 'ctg' in _bo and _bo['random-greedy-128'] - _bo['ctg'] > -0.5:
                _G.append((_bo['ctg'], max(0.0, _bo['random-greedy-128'] - _bo['ctg'])))
_gx, _gy = [x for x, _ in _G], [y for _, y in _G]
_gxb, _gyb = sum(_gx) / len(_gx), sum(_gy) / len(_gy)
GAIN_B = sum((x - _gxb) * (y - _gyb) for x, y in zip(_gx, _gy)) / sum((x - _gxb) ** 2 for x in _gx); GAIN_A = _gyb - GAIN_B * _gxb
GAIN_RMS = math.sqrt(sum((y - (GAIN_A + GAIN_B * x)) ** 2 for x, y in zip(_gx, _gy)) / max(1, len(_gx) - 2))
def gain(lf): return max(0.0, GAIN_A + GAIN_B * lf)
def opt_gap(cs):
    # path-search uncertainty of one shape: best path of the best non-greedy optimiser vs the second-best one (None if < 2 optimisers)
    bo = {}
    for c in cs:
        if c.get('lf') is not None and c['o'] not in GREEDY: bo[c['o']] = min(bo.get(c['o'], 1e9), c['lf'])
    v = sorted(bo.values()); return (v[1] - v[0]) if len(v) > 1 else None
def row_gap(r, kind):
    g = [opt_gap(cs) for cs in (r.get(f'paths_{kind}') or [])]; g = [x for x in g if x is not None]
    return sum(g) / len(g) if g else 0.0
def tnc(meas, meas_s, paths, lf_any, lf_greedy, mult, gps, gap_fb=0.0):
    # classical TN energy for one network kind: per shape the path with the lowest (mult x run + search / K); geometric mean over shapes
    if meas is None and lf_any is None: return None
    over = max(1.0, meas - (10 ** lf_greedy / ETA if lf_greedy is not None else 0)) if meas is not None else 1.0
    def run(c):
        if meas is not None: return meas if c['o'] == 'greedy' else min(meas, max(10 ** c['lf'] / ETA, over))
        return max(1.0, 10 ** c['lf'] / ETA)
    shapes = paths if paths else [[{'o': 'greedy', 'lf': lf_any, 's': gps or 0}]]
    if MC: shapes = [MC['rng'].choice(shapes) for _ in shapes]          # workload: resample the circuit shapes of this size
    per = []
    for cs in shapes:
        best = None
        if MC:   # one-sided path-quality band: a path better than the best found by up to the gap between the two best distinct optimisers
            u = MC['rng'].random()
            cs = [dict(c, lf=c['lf'] - u * gain(c['lf'])) if c.get('lf') is not None and c['o'] not in GREEDY else c for c in cs]
        for c in cs:
            ru = run(c)
            if MC and meas is not None and ru == meas: ru = meas * MC['meas_f']   # measured: idle-subtracted (net) .. gross convention
            t_run = (meas_s if (meas is not None and ru == meas) else max(0.016, 10 ** c['lf'] / RATE))
            host = mult * t_run * PCPU                       # host share of the GPU run (one busy core)
            tot = mult * ru + host + c['s'] * PCPU / K
            if best is None or tot < best['tot']: best = dict(tot=tot, meas=meas is not None and ru == meas, o=c['o'], lf=c['lf'])
        per.append(best)
    gm = math.exp(sum(math.log(p['tot']) for p in per) / len(per))
    return dict(J=gm, measured=all(p['meas'] for p in per), by=collections.Counter(p['o'] for p in per).most_common(1)[0][0])

def classical(r):
    for kind in ('ps', 'rho'):   # path-search uncertainty: this row's optimiser gap, else the mean of the 4/8-layer rows of that size
        if f'gap_{kind}' not in r:
            gg = row_gap(r, kind)
            if not gg:
                nb = [row_gap(x, kind) for x in (by(4, r['nq']), by(8, r['nq'])) if x]; gg = sum(nb) / len(nb) if nb else 0.0
            r[f'gap_{kind}'] = gg
    g = r.get('gpu') or {}
    out = {}
    sv = g.get('cusv_rho')
    if sv: out['statevector'] = dict(J=sv['J'] + sv['s'] * PCPU, status='measured')
    lfP = r.get('best_ps_lf', r.get('greedy_ps_lf')); lfR = r.get('best_rho_lf', r.get('greedy_rho_lf'))
    gt, gr = g.get('tn'), g.get('tn_rho')
    T = tnc(gr['J'] if gr else None, gr['s'] if gr else None, r.get('paths_rho'), lfR, r.get('greedy_rho_lf'), 1, r.get('greedy_path_s'), r.get('gap_rho', 0.0))
    A = tnc(gt['J'] if gt else None, gt['s'] if gt else None, r.get('paths_ps'), lfP, r.get('greedy_ps_lf'), N0, r.get('greedy_path_s'), r.get('gap_ps', 0.0))
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
    if MC: f *= MC['rng'].choice(ov_rel(L))                      # workload: placement-dependent routing overhead of individual circuits
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
def F2(b, q, m):
    if b == 'IBM': return math.exp(CAL['a'] - CAL['eIBM'] * m * q['ibm_n2q']) ** 2
    if b == 'IonQ': return math.exp(CAL['a'] - (CAL['eIonQ'] * q['ionq_n2q'] + CAL['e1IonQ'] * q['ionq_n1q']) * m) ** 2
    return math.exp(IQM['a'] - IQM['eff'] * m * q['iqm_n2q']) ** 2
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
        base = (by(L, nq) or {}).get('n2q_logical') or float(r['logical_n2q'])   # same logical count as the Kingston overhead uses
        if r['fits'] == '1' and base > 0: iqm_ov[L].append(float(r['iqm_n2q']) / base)
IQM_ROUTE = {L: statistics.median(v) for L, v in iqm_ov.items()}
def iqm_overhead(L):
    if L in IQM_ROUTE: return IQM_ROUTE[L], 'IQM lattice, median at this depth'
    if IQM_ROUTE: Lc = min(IQM_ROUTE, key=lambda x: abs(x - L)); return IQM_ROUTE[Lc], f'IQM lattice, median at {Lc} layers'
    return route(L), 'placeholder: Kingston heavy-hex overhead (IQM transpile pending)'

def attach_iqm(r):
    v = iqm.get((r['layers'], r['nq']), [])
    f, src = iqm_overhead(r['layers'])
    fit = [x for x in v if x['fits'] == '1']
    # logical x median overhead at every size (as IBM's default): per-circuit transpiles vary with placement and make the curve jagged
    if MC and iqm_ov: f *= MC['rng'].choice(iqm_rel(r['layers']))
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
        # two candidates, as for measured rows: the best path (search time of the slower of the two depths' best searches) and
        # the greedy path (its search time); the model keeps the cheaper total per decision
        st = lambda x: min((c['s'] for cs in (x.get(f'paths_{kind}') or []) for c in cs if c['o'] == x.get(f'best_{kind}_by')), default=600)
        cand = []
        if r[f'best_{kind}_lf'] is not None: cand.append({'o': 'interp', 'lf': r[f'best_{kind}_lf'], 's': max(st(a), st(b))})
        g = li(a.get(f'greedy_{kind}_lf'), b.get(f'greedy_{kind}_lf'))
        if g is not None: cand.append({'o': 'greedy-interp', 'lf': g, 's': li(a.get('greedy_path_s') or 0, b.get('greedy_path_s') or 0)})
        r[f'paths_{kind}'] = [cand] if cand else None
        r[f'gap_{kind}'] = (row_gap(a, kind) + row_gap(b, kind)) / 2   # path-search uncertainty borrowed from the real 4/8-layer rows
    return r

def row_at(nq, L):
    r = by(L, nq)
    if r and (r.get('paths_ps') or r.get('gpu') or L == 2): return dict(r)
    return lerp_row(nq, L) if L not in (4, 8) else (dict(r) if r else None)


# ---- GPU statevector as its own comparison line (cf. Knitter et al. 2026, statevector-only baseline) ----
# best statevector engine per point: min over cuStateVec (state), the lazy torch statevector and Aer, J + host share. Measured at that
# depth where available; else the 2-layer measurement at that size x the logical two-qubit gate ratio (statevector cost ~ gates x 2^n)
# = 'scaled'; beyond the measured sizes an exponential fit (log10 J linear in qubits) over the 3 largest 2-layer sizes, same gate scaling
# = 'extrapolated'.
SV_ENG = ('cusv', 'lazy', 'aer')
def sv_of(r):
    g = r.get('gpu') or {}
    v = [(g[e]['J'] + g[e]['s'] * PCPU, e) for e in SV_ENG if g.get(e) and g[e].get('J')]
    return min(v) if v else None
SV2 = {r['nq']: sv_of(r) for r in rows_all if r['layers'] == 2 and sv_of(r)}
N2Q2 = {r['nq']: r['n2q_logical'] for r in rows_all if r['layers'] == 2 and r.get('n2q_logical')}
_m = sorted(SV2)[-3:]
_xs, _ys = _m, [math.log10(SV2[x][0] / N2Q2[x]) for x in _m]          # fit per two-qubit gate, so the depth scaling stays separate
_xb, _yb = sum(_xs) / 3, sum(_ys) / 3
SV_SLOPE = sum((x - _xb) * (y - _yb) for x, y in zip(_xs, _ys)) / sum((x - _xb) ** 2 for x in _xs); SV_ICPT = _yb - SV_SLOPE * _xb
SV_MAXQ = max(SV2)
# prediction interval of the statevector extrapolation: regression of log10(J per gate) on qubits over the measured 2-layer sizes >= 22 q
_pm = sorted(x for x in SV2 if x >= 22)
_px, _py = _pm, [math.log10(SV2[x][0] / N2Q2[x]) for x in _pm]
_pxb, _pyb = sum(_px) / len(_px), sum(_py) / len(_py); _sxx = sum((x - _pxb) ** 2 for x in _px)
_pb = sum((x - _pxb) * (y - _pyb) for x, y in zip(_px, _py)) / _sxx
_res = [y - (_pyb + _pb * (x - _pxb)) for x, y in zip(_px, _py)]
SV_S = math.sqrt(sum(e * e for e in _res) / max(1, len(_px) - 2)); SV_DOF = max(1, len(_px) - 2)
def sv_pi(x): return SV_S * math.sqrt(1 + 1 / len(_px) + (x - _pxb) ** 2 / _sxx)
def sv_point(r):
    own = sv_of(r)
    if own: return dict(J=own[0], status='measured', eng=own[1])
    if r['nq'] in SV2 and r.get('n2q_logical') and r['nq'] in N2Q2:
        return dict(J=SV2[r['nq']][0] * r['n2q_logical'] / N2Q2[r['nq']], status='scaled', eng=SV2[r['nq']][1])
    if r.get('n2q_logical'):
        return dict(J=10 ** (SV_ICPT + SV_SLOPE * r['nq']) * r['n2q_logical'], status='extrapolated', eng='fit')
    return None

def point(r):
    attach_iqm(r)
    c = classical(r)
    if c is None or not r.get('ibm_n2q'): return None
    q = qpu_counts(r)
    st = 'interpolated' if r.get('interp') else c['status']
    if MC and r.get('interp'):
        f = 10 ** MC['rng'].gauss(0, MC['sig_interp']); c = dict(c, best=c['best'] * f)
    p = dict(nq=r['nq'], L=r['layers'], n=r['n'], classical=c['best'], cls_by=c['by'], cls_status=st,
             methods={k: dict(J=v['J'], status='interpolated' if r.get('interp') else v['status']) for k, v in c['methods'].items()},
             iqm_src=q['iqm_src'], qpu={}, be={})
    p['sv'] = sv_point(r); p['be_sv'] = {}
    if MC and p['sv']:
        if p['sv']['status'] == 'extrapolated': p['sv'] = dict(p['sv'], J=p['sv']['J'] * 10 ** (MC['sv_t']() * sv_pi(r['nq'])))
        elif p['sv']['status'] in ('measured', 'scaled'): p['sv'] = dict(p['sv'], J=p['sv']['J'] * MC['meas_f'])
    p['shot_J'] = dict(IBM=(q['ibm_dur'] + IBM['rep']) * IBM['P'], IonQ=(q['ionq_n1q'] * IONQ['t1'] + q['ionq_n2q'] * IONQ['t2'] + IONQ['shot']) * IONQ['P'],
                       IQM=(IQM['shot'] + q['iqm_dur']) * IQM['P'])
    p['shots'] = {b: {str(m): N0 / F2(b, q, m) for m in NOISE} for b in EF}
    for b, f in EF.items():
        p['qpu'][b] = {str(m): f(q, m) for m in NOISE}
        m = breakeven(q, c['best'], f)
        p['be'][b] = None if m is None else TODAY[b] * m
        m = breakeven(q, p['sv']['J'], f) if p['sv'] else None
        p['be_sv'][b] = None if m is None else TODAY[b] * m
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

# ================= uncertainty: Monte Carlo over the real sources, band = 16th-84th percentile of the draws =================
NDRAW = int(os.environ.get('EFIG_DRAWS', '2000'))
def build_series():
    out = {}
    for name, rule in (('log2', L_of), ('L4', lambda n: 4), ('L8', lambda n: 8)):
        pts = []
        for nq in SIZES:
            r = row_at(nq, rule(nq))
            if r is None: continue
            p = point(r)
            if p: pts.append(p)
        out[name] = pts
    return out
# measurement convention: idle-subtracted (net) / gross energy per decision of the GPU runs
_ratios = []
for fn in glob.glob(os.path.join(H, 'gpu_engines*.csv')):
    for r in csv.DictReader(open(fn)):
        try:
            g, n = float(r['J_gross_per_decision']), float(r['J_net_per_decision'])
            if g > 0 and 0 < n <= g: _ratios.append(n / g)
        except (KeyError, ValueError): pass
NET_GROSS = statistics.median(_ratios) if _ratios else 1.0
# interpolation error: the real 5-layer circuits against what the 4/8-layer interpolation predicted for them
_ie = []
for nq in SIZES:
    real = by(5, nq)
    if real and real.get('paths_ps'):
        a_, b_ = classical(attach_iqm(dict(real))), classical(attach_iqm(lerp_row(nq, 5) or {}) ) if lerp_row(nq, 5) else None
        if a_ and b_: _ie.append(math.log10(b_['best'] / a_['best']))
SIG_INTERP = math.sqrt(sum(e * e for e in _ie) / len(_ie)) if _ie else 0.5
# noise-fit parameters: bootstrap of the calibration emulations (until each backend's own calibration lands: the Kingston post-selected
# runs, ln F_eff against mean qubits, whose relative slope and intercept spread is applied to every backend)
_cal = [r for r in csv.DictReader(open(os.path.join(H, 'noise_calib.csv'))) if r['method'] == 'ps']
def _fit(rows):
    xs = [float(r['nq_mean']) for r in rows]; ys = [math.log(float(r['F_eff'])) for r in rows]
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys); sxx = sum((x - xb) ** 2 for x in xs)
    b = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return yb - b * xb, b
CAL_A0, CAL_B0 = _fit(_cal)
RANGES = dict(IBM_P=(25e3, 30e3), IBM_rep=(100e-6, 500e-6), IonQ_P=(3.5e3, 6.5e3), IonQ_t=(0.8, 1.2), IQM_P=(24e3, 30e3), IQM_shot=(0.96e-3, 1.92e-3))
SOURCES = [
    ('workload', 'QPU: routing overhead of individual circuits at that depth (placement on the coupling map); classical: one typical-width '
                 'network per size was searched, so the per-sentence spread of contraction cost is NOT covered'),
    ('measured GPU energy', f'idle-subtracted to gross energy convention (median net/gross {NET_GROSS:.2f}; gross, the central value, is the '
                            'upper edge); run-to-run repeats on an L4 are queued and will be added'),
    ('classical upper bounds', f'one-sided (downward only; the best path found is the upper edge): log10 FLOP lowered by U(0,1) x g, where '
                               f'g = {GAIN_A:.2f} + {GAIN_B:.3f} log10 FLOP (fit over {len(_G)} networks of the gain of the 600 s cotengra search '
                               f'over random-greedy-128, residual {GAIN_RMS:.2f} decades): one path-quality model for all sizes'),
    ('interpolated depths', f'log-normal, sigma = {SIG_INTERP:.2f} decades: error of the 4/8-layer interpolation measured on the real 5-layer circuits'),
    ('statevector extrapolation', 'Student-t prediction interval of the regression of log10(J per gate) on qubits over the measured 22-31 q points'),
    ('QPU noise fit', f'bootstrap of the {len(_cal)} post-selected Kingston calibration emulations (slope and intercept of ln F_eff), applied to all '
                      'backends until their own calibrations land'),
    ('QPU power', 'IBM 25-30 kW (26 kW System Two refrigerator, arXiv:2304.14344; ~30 kW System One installations); IonQ 3.5-6.5 kW '
                  '(5 kW +-30%, assumed); IQM 24-30 kW (typical mean, IQM Radiance spec; 30 kW peak, Mansfield et al. 2025)'),
    ('QPU time per shot', 'IBM repetition delay 100-500 us (assumed; IBM docs example uses 500 us); IonQ gate and shot times +-20% (assumed); '
                          'IQM 0.96-1.92 ms (CLOPS-derived upper value, half of it as lower, assumed)'),
]
NOT_COVERED = ['better classical algorithms than those tried (paths are upper bounds only within the searched optimisers)',
               'hardware roadmaps (gate speed, power, connectivity changes)', 'run-to-run GPU energy spread (no repeats)',
               'error mitigation or error correction overheads', 'queueing and job-scheduling time']
def draw_all(seed=12345):
    global MC
    rng = random.Random(seed)
    base = (dict(CAL), dict(IBM), dict(IONQ), dict(IQM), dict(TODAY))
    draws = []
    try:
        for i in range(NDRAW):
            bs = [rng.choice(_cal) for _ in _cal]; a_, b_ = _fit(bs)
            fac = b_ / CAL_B0 if CAL_B0 else 1.0; da = a_ - CAL_A0
            CAL.update(base[0]); IBM.update(base[1]); IONQ.update(base[2]); IQM.update(base[3])
            for k in ('eIBM', 'eIonQ', 'e1IonQ'): CAL[k] = base[0][k] * fac
            CAL['a'] = base[0]['a'] + da
            IQM['eff'] = base[3]['eff'] * fac; IQM['a'] = (base[0]['a'] + da) * base[3]['a'] / base[0]['a']
            IBM['P'] = rng.uniform(*RANGES['IBM_P']); IBM['rep'] = rng.uniform(*RANGES['IBM_rep'])
            IONQ['P'] = rng.uniform(*RANGES['IonQ_P']); ft = rng.uniform(*RANGES['IonQ_t'])
            for k in ('t1', 't2', 'shot'): IONQ[k] = base[2][k] * ft
            IQM['P'] = rng.uniform(*RANGES['IQM_P']); IQM['shot'] = rng.uniform(*RANGES['IQM_shot'])
            TODAY.update(IBM=CAL['eIBM'], IonQ=CAL['eIonQ'], IQM=IQM['eff'])
            dof = SV_DOF
            MC = dict(rng=rng, meas_f=rng.uniform(NET_GROSS, 1.0), sig_interp=SIG_INTERP,
                      sv_t=lambda: rng.gauss(0, 1) / math.sqrt(rng.gammavariate(dof / 2, 2) / dof))
            draws.append(build_series())
    finally:
        MC = None
        CAL.update(base[0]); IBM.update(base[1]); IONQ.update(base[2]); IQM.update(base[3]); TODAY.update(base[4])
    return draws
def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = math.floor(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)
def crossing(pts, b, m, base):
    B = (lambda p: p['sv']['J'] if p.get('sv') else None) if base == 'sv' else (lambda p: p['classical'])
    for i, p in enumerate(pts):
        q, c = p['qpu'][b].get(m), B(p)
        if q is None or c is None: continue
        if q <= c:
            if i == 0: return p['nq']
            o = pts[i - 1]; q0, c0 = o['qpu'][b][m], B(o)
            if c0 is None or q0 is None: return p['nq']
            d0, d1 = math.log10(q0) - math.log10(c0), math.log10(q) - math.log10(c)
            return o['nq'] + (p['nq'] - o['nq']) * d0 / (d0 - d1) if d0 != d1 else p['nq']
    return None
def attach_bands(series, draws):
    cross = {}
    for name, pts in series.items():
        D_ = [d[name] for d in draws if len(d[name]) == len(pts)]
        lg = lambda v: [math.log10(x) for x in v if x and x > 0]
        band = lambda v: None if len(v) < 10 else [10 ** pct(v, 0.16), 10 ** pct(v, 0.84)]
        for i, p in enumerate(pts):
            up = lambda bnd, c: None if bnd is None else [min(bnd[0], c), max(bnd[1], c)]
            p['band'] = dict(classical=up(band(lg([d[i]['classical'] for d in D_])), p['classical']),
                             sv=up(band(lg([d[i]['sv']['J'] for d in D_ if d[i].get('sv')])), p['sv']['J']) if p.get('sv') else None,
                             qpu={b: {m: band(lg([d[i]['qpu'][b][m] for d in D_ if d[i]['qpu'][b].get(m)])) for m in p['qpu'][b]} for b in EF},
                             be={b: band(lg([d[i]['be'][b] for d in D_ if d[i]['be'][b]])) for b in EF},
                             be_sv={b: band(lg([d[i]['be_sv'][b] for d in D_ if d[i]['be_sv'].get(b)])) for b in EF},
                             be_frac={b: sum(1 for d in D_ if d[i]['be'][b]) / len(D_) for b in EF})
        cr = {}
        for b in EF:
            for m in (str(x) for x in NOISE):
                for base_ in ('cls', 'sv'):
                    xs = [crossing(d, b, m, base_) for d in D_]
                    got = [x for x in xs if x is not None]
                    cr[f'{b}|{m}|{base_}'] = dict(x=crossing(pts, b, m, base_), lo=pct(got, 0.16) if len(got) >= 10 else None,
                                                  hi=pct(got, 0.84) if len(got) >= 10 else None, frac=len(got) / len(xs))
        cross[name] = cr
    return cross
_t0 = datetime.datetime.now()
DRAWS = draw_all() if NDRAW else []
CROSS = attach_bands(series, DRAWS) if DRAWS else {}
MC_S = (datetime.datetime.now() - _t0).total_seconds()

meta = dict(
    updated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    N0=N0, K=round(K, 1), PCPU=PCPU, ETA=ETA, RATE=RATE, route={str(k): round(v, 2) for k, v in ROUTE.items()},
    iqm_route={str(k): round(v, 2) for k, v in IQM_ROUTE.items()}, today=TODAY, noise=NOISE,
    consts=dict(IBM=IBM, IonQ=IONQ, IQM={k: v for k, v in IQM.items()}, CAL=CAL, KING_MED2Q=KING_MED2Q, KING_MEDRO=KING_MEDRO),
    sv_ext=sv_ext, sv_fit=dict(slope=SV_SLOPE, icpt=SV_ICPT, max_measured_nq=SV_MAXQ, per='J per logical 2q gate', engines=SV_ENG),
    pending=[], cross=CROSS,
    unc=dict(draws=NDRAW, band='16th-84th percentile of the Monte Carlo draws; central line = current estimate', sources=SOURCES,
             not_covered=NOT_COVERED, sig_interp=SIG_INTERP, net_gross=NET_GROSS, cal_rows=len(_cal), ranges=RANGES, seconds=MC_S))
json.dump(dict(meta=meta, series=series), open(os.path.join(H, 'efig.json'), 'w'), indent=0)
if __name__ == '__main__':
    for name, pts in series.items():
        print('==', name)
        for p in pts:
            print(f"{p['nq']:4d}q L{p['L']} cls {p['classical']:.3g} J ({p['cls_by']}, {p['cls_status']})  " +
                  '  '.join(f"{b} {p['qpu'][b]['1']:.3g}/{p['qpu'][b]['0.01']:.3g} be {('%.2g' % p['be'][b]) if p['be'][b] else '-'}" for b in EF))
    print('ETA %.3g FLOP/J RATE %.3g FLOP/s route %s IQM eff %.3g a %.3g shot %.4g s' % (ETA, RATE, ROUTE, IQM['eff'], IQM['a'], IQM['shot']))
