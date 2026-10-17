# Aggregate readout_swap.py output. usage: readout_summ.py <pkl>... -> prints tables, writes <first pkl>.summary.json
import sys, json, pickle, collections, numpy as np, pandas as pd
R = [r for f in sys.argv[1:] for r in pickle.load(open(f, 'rb'))]
def grp(r): return f"{'discard' if r['trained_discard'] else 'nodiscard'}_L{r['layers']}"
G = collections.defaultdict(list)
for r in R: G[grp(r)].append(r)
ms = lambda x: f'{np.mean(x):.3f}±{np.std(x, ddof=1) if len(x) > 1 else 0:.3f}'
summary = {}
print('sanity (max |own output - our readout|):', max(r['sanity_maxdiff'] for r in R))
print('\n== accuracy: trained with (rows) x read out by (columns), mean±sd over seeds')
for g, rs in sorted(G.items()):
    row = {}
    for sp in ('train', 'val'):
        for ro in ('ps', 'tr'):
            row[f'{sp}_{ro}'] = [float((r[sp][f'{ro}_pos'] > r[sp][f'{ro}_neg']).mean()) for r in rs]
    summary[g] = {'n': len(rs), **{k: [float(np.mean(v)), float(np.std(v))] for k, v in row.items()}}
    print(f"{g:16s} n={len(rs)}  val: postsel {ms(row['val_ps'])}  trace {ms(row['val_tr'])} | train: postsel {ms(row['train_ps'])}  trace {ms(row['train_tr'])}")
print('\n== val accuracy by negative type, own readout vs swapped readout')
for g, rs in sorted(G.items()):
    out = []
    for t in ('subj_neg', 'verb_neg', 'obj_neg'):
        for ro in ('ps', 'tr'):
            out.append(f"{t[:4]}-{ro} {np.mean([((r['val'][ro+'_pos'] > r['val'][ro+'_neg'])[r['val'][t]]).mean() for r in rs]):.3f}")
    print(f'{g:16s} ' + '  '.join(out))
print('\n== branch structure of the sentence states (val, median over sentences, mean over seeds)')
for g, rs in sorted(G.items()):
    v = lambda k: np.mean([r['val'][k].median() for r in rs])
    print(f"{g:16s} p0 {v('p0'):.3f}  purity(trace) {v('purity'):.3f}  lambda_max {v('lmax'):.3f}  <psi0|rho|psi0>/lambda_max {v('branch0_is_top'):.3f}")
    for m, d in rs[0]['val'].groupby('m'):
        pass
print('\n   p0 and purity by m (non-sentence wires), first seed of each group:')
for g, rs in sorted(G.items()):
    d = rs[0]['val'].groupby('m').agg(n=('p0', 'size'), p0=('p0', 'median'), purity=('purity', 'median'))
    print(g, ' '.join(f"m={m}:n={int(x.n)},p0={x.p0:.3f},pur={x.purity:.2f}" for m, x in d.iterrows()))
print('\n== distinguishability of val sentences that differ in ONE of subj/verb/obj (same circuit shape)')
def dists(r):
    df = r['val']; gs = r['val_global']; trip = df.pos_triplet.str.split(',').tolist(); out = collections.defaultdict(list)
    idx = collections.defaultdict(list)
    for i, (t, e) in enumerate(zip(trip, df.einsum)):
        for s in range(3): idx[(e, s, tuple(x for k, x in enumerate(t) if k != s))].append(i)
    for (e, s, _), ii in idx.items():
        ii = ii[:12]
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                if trip[i][s] == trip[j][s]: continue
                A, B = gs[i].numpy(), gs[j].numpy(); d = 2 ** 3
                pa, pb = A.reshape(d, -1), B.reshape(d, -1)
                Dg = np.sqrt(max(0.0, 1 - abs(np.vdot(A, B)) ** 2))
                ra, rb = pa @ pa.conj().T, pb @ pb.conj().T
                Dt = 0.5 * np.abs(np.linalg.eigvalsh(ra - rb)).sum()
                a0, b0 = pa[:, 0] / np.linalg.norm(pa[:, 0]), pb[:, 0] / np.linalg.norm(pb[:, 0])
                Dp = np.sqrt(max(0.0, 1 - abs(np.vdot(a0, b0)) ** 2))
                out[('subj', 'verb', 'obj')[s]].append((Dg, Dt, Dp))
    return out
for g, rs in sorted(G.items()):
    acc = collections.defaultdict(list)
    for r in rs:
        for k, v in dists(r).items(): acc[k] += v
    summary[g]['dist'] = {}
    for k in ('subj', 'verb', 'obj'):
        a = np.array(acc[k])
        if not len(a): continue
        summary[g]['dist'][k] = dict(n=len(a), global_=float(a[:, 0].mean()), trace=float(a[:, 1].mean()), postsel=float(a[:, 2].mean()),
                                     frac_postsel_gt_global=float((a[:, 2] > a[:, 0] + 1e-6).mean()), frac_trace_gt_global=float((a[:, 1] > a[:, 0] + 1e-5).mean()))
        print(f"{g:16s} {k:4s} pairs {len(a):5d}  D_global {a[:,0].mean():.3f}  D_trace {a[:,1].mean():.3f}  D_postsel {a[:,2].mean():.3f}  "
              f"postsel>global {np.mean(a[:,2] > a[:,0] + 1e-6):.2f}  trace>global {np.mean(a[:,1] > a[:,0] + 1e-5):.3f}")
print('\n== effective dimension (participation ratio of the Bloch-vector covariance, 63 dims), val')
def pr(X):
    w = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None); return float(w.sum() ** 2 / (w ** 2).sum())
for g, rs in sorted(G.items()):
    t = [pr(r['val_bloch']['bloch_tr']) for r in rs]; p = [pr(r['val_bloch']['bloch_ps']) for r in rs]; im = [pr(r['val_bloch']['img_bloch']) for r in rs]
    summary[g]['eff_dim'] = dict(trace=float(np.mean(t)), postsel=float(np.mean(p)), image=float(np.mean(im)))
    print(f'{g:16s} text(trace) {ms(t)}  text(postsel) {ms(p)}  images {ms(im)}')
print('\n== margins of the own readout (score_pos - score_neg): median, and share of pairs with |margin| < 0.02')
for g, rs in sorted(G.items()):
    ro = 'tr' if rs[0]['trained_discard'] else 'ps'
    o = []
    for sp in ('train', 'val'):
        mg = np.concatenate([(r[sp][ro + '_pos'] - r[sp][ro + '_neg']).values for r in rs])
        sc = np.concatenate([np.r_[r[sp][ro + '_pos'].values, r[sp][ro + '_neg'].values] for r in rs])
        o.append(f"{sp}: median {np.median(mg):.3f} (score sd {sc.std():.3f}), |m|<0.02 {np.mean(np.abs(mg) < 0.02):.3f}")
    print(f'{g:16s} ' + ' | '.join(o))
json.dump(summary, open(sys.argv[1] + '.summary.json', 'x'), indent=1)
