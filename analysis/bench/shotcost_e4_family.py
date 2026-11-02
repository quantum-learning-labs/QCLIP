# E4 family summary (test std). Per pair, shots per decision for unit SNR: cost = circuits * V / D^2.
#   cu : native compute-uncompute (2 circuits; PS ratio estimator charged 1/a)    O : 1 circuit, eigenbasis of O, native weights
# Paired (same seed, pairs both rank correctly): geo-mean cost ratio discard/PS for (disc cu / ps cu), (disc O / ps O), (disc O / ps cu).
# Also the per-family median shots-to-target from shotcost_e4_report.py's JSON.  usage: shotcost_e4_family.py <e4 dir> <e4 report json>
import glob, json, os, re, sys, collections
import numpy as np
NAMES = {'hp_hp2_nodiscard_tl3_qlr3e-3': 'PS m4', 'hp_hp2_discard_tl3_pw0': 'plain m4', 'hp_do1_discard_tl3_pw0_initnd': 'warm m4',
         'hp_kd1_discard_kd_ws': 'distill m4', 'hp_xo1_nodiscard_n2_tl3_qlr0.003': 'PS m8', 'hp_xo1_discard_n2_tl3_pw0': 'plain m8',
         'hp_xo1_discard_n2_tl3_pw0_ws': 'warm m8', 'hp_sc3_nodiscard_n3_tl3_qlr3e-3': 'PS m10', 'hp_sc3_discard_n3_tl3_pw0': 'plain m10',
         'hp_sc3_discard_n3_tl3_pw0_ws': 'warm m10',
         'hp_sc2_nodiscard_n2_tl3_qlr3e-3': 'PS m8 sc2', 'hp_sc2_discard_n2_tl3_pw0': 'plain m8 sc2',
         'hp_sc2_discard_n2_tl3_pw0_ws': 'warm m8 sc2', 'hp_sc2_discard_n2_tl3_pw0_kd': 'distill m8 sc2'}
ORDER = list(NAMES.values())
d, rj = sys.argv[1], json.load(open(sys.argv[2]))


def costs(f, discard):
    b = np.load(f); P, U, G, Q = b['P'], b['U'], b['G'], b['Q']
    if discard: m = U.sum(-1); a = np.ones_like(m); w = np.ones(P.shape[-1])
    else: a = P[..., 0]; m = U[..., 0] / a; w = np.eye(P.shape[-1])[0]
    D = m[:, 0] - m[:, 1]
    Vcu = m[:, 0] * (1 - m[:, 0]) / a[:, 0] + m[:, 1] * (1 - m[:, 1]) / a[:, 1]
    DO = G @ w; VO = Q @ (w * w) - DO ** 2
    if not discard:   # PS on O with rejected shots charged: ratio estimator, per accepted shot var (Q0/a - (G0/a)^2), charge 1/a
        DO = G[:, 0] / a[:, 0]; VO = (Q[:, 0] / a[:, 0] - DO ** 2) / a[:, 0]
    return {'idx': b['idx'] if 'idx' in b.files else np.arange(len(D)), 'D': D, 'cu': 2 * Vcu / np.maximum(D, 1e-300) ** 2, 'O': VO / np.maximum(DO, 1e-300) ** 2, 'ok': D > 0, 'okO': DO > 0}


ms = lambda v: f'{np.mean(v):6.2f}+-{np.std(v, ddof=1) if len(v) > 1 else 0:.2f}'
fams = collections.defaultdict(list)
for f in sorted(glob.glob(f'{d}/*__std.npz')):
    ck = os.path.basename(f)[:-len('__std.npz')]
    meta = json.load(open(f[:-4] + '.json'))
    fams[NAMES.get(re.sub(r'_s\d+$', '', ck), ck)].append((ck, meta['discard'], meta['text_n'], re.search(r'_s(\d+)$', ck).group(1), f))
ps = {(n, ('sc2' if ck.startswith('hp_sc2_') else 'older'), s): f for fam, xs in fams.items() for ck, disc, n, s, f in xs if not disc}   # within one sweep tag
print('== paired discard/PS cost ratio (geo-mean over pairs both rank correctly, mean+-sd over seeds); <1 = discard cheaper')
print(f"{'model':11s} seeds  {'cu/cu':>12s} {'O/O':>12s} {'disc O/ps cu':>13s}   PS: O/cu      disc: O/cu")
for fam in ORDER:
    rows = []
    for ck, disc, n, s, f in fams.get(fam, []):
        key = (n, ('sc2' if ck.startswith('hp_sc2_') else 'older'), s)
        if not disc or key not in ps: continue
        cd, cp = costs(f, True), costs(ps[key], False)
        common = np.intersect1d(cd['idx'], cp['idx'])
        cd = {k: v[np.searchsorted(cd['idx'], common)] for k, v in cd.items()}
        cp = {k: v[np.searchsorted(cp['idx'], common)] for k, v in cp.items()}
        both = cd['ok'] & cp['ok']
        g = lambda x: float(np.exp(np.mean(np.log(x[both]))))
        rows.append((g(cd['cu']) / g(cp['cu']), g(cd['O']) / g(cp['O']), g(cd['O']) / g(cp['cu']), g(cp['O']) / g(cp['cu']), g(cd['O']) / g(cd['cu'])))
    if rows:
        r = np.array(rows)
        print(f"{fam:11s} {len(rows):5d}  {ms(r[:, 0])} {ms(r[:, 1])} {ms(r[:, 2]):>13s}   {ms(r[:, 3])}  {ms(r[:, 4])}")
cols = ['native_cu', 'native_O', 'wO@64', 'wO@512', 'wCU@64', 'wCU@512']
get = lambda r, c: r[c] if c in r else r['fits'][c]['test']
for t in ('0.75', '0.8', '0.85'):
    print(f'\n== test std: median over seeds of shots per decision to acc {t} [seeds reaching it]; mean exact acc in brackets')
    print(f"{'model':11s} " + ' '.join(f'{c:>18s}' for c in cols))
    for fam in ORDER:
        rs = [rj[ck] for ck, *_ in fams.get(fam, []) if ck in rj]
        if not rs: continue
        cell = []
        for c in cols:
            v = [get(r, c)['shots_per_decision'][t] for r in rs]; ok = [x for x in v if x is not None]
            acc = np.mean([get(r, c)['exact_acc'] for r in rs])
            cell.append(f"{(f'{np.median(ok):.0f}' if ok else '-')}[{len(ok)}/{len(rs)}]({acc:.3f})")
        print(f"{fam:11s} " + ' '.join(f'{x:>18s}' for x in cell))
print('\n== fitted weights wO@64 by Hamming weight h (mean over seeds)')
for fam in ORDER:
    rs = [rj[ck] for ck, *_ in fams.get(fam, []) if ck in rj]
    if not rs: continue
    L = min(len(r['fits']['wO@64']['w']) for r in rs)
    print(f"{fam:11s} " + ' '.join(f'{x:+.2f}' for x in np.mean([r['fits']['wO@64']['w'][:L] for r in rs], 0)))
