# Per-epoch training dynamics from readout_swap.py run over epoch_*.pt snapshots.
# usage: dyn_summ.py <out.csv> <pkl>...   prints one block per run, writes all rows to a CSV.
import sys, pickle, collections, csv, numpy as np
out, files = sys.argv[1], sys.argv[2:]
R = [r for f in files for r in pickle.load(open(f, 'rb'))]
def pr(X):
    w = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None); return float(w.sum() ** 2 / (w ** 2).sum())
def dists(r, cap=2000):
    df = r['val']; gs = r['val_global']; trip = df.pos_triplet.str.split(',').tolist(); idx = collections.defaultdict(list)
    for i, (t, e) in enumerate(zip(trip, df.einsum)):
        for s_ in range(3): idx[(e, s_, tuple(x for k, x in enumerate(t) if k != s_))].append(i)
    acc = []
    for (e, s_, _), ii in idx.items():
        ii = ii[:6]
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                if trip[i][s_] == trip[j][s_]: continue
                A, B = gs[i].numpy(), gs[j].numpy(); pa, pb = A.reshape(8, -1), B.reshape(8, -1)
                Dg = np.sqrt(max(0.0, 1 - abs(np.vdot(A, B)) ** 2))
                Dt = 0.5 * np.abs(np.linalg.eigvalsh(pa @ pa.conj().T - pb @ pb.conj().T)).sum()
                a0, b0 = pa[:, 0] / np.linalg.norm(pa[:, 0]), pb[:, 0] / np.linalg.norm(pb[:, 0])
                acc.append((Dg, Dt, np.sqrt(max(0.0, 1 - abs(np.vdot(a0, b0)) ** 2))))
                if len(acc) >= cap: break
    a = np.array(acc); return a.mean(0) if len(a) else (np.nan,) * 3
rows = []
for r in R:
    v = r['val']; own = 'tr' if r['trained_discard'] else 'ps'
    ep = r['epoch'] if r['epoch'] is not None else -1
    Dg, Dt, Dp = dists(r)
    row = dict(run=r['ckpt'].rstrip('/').split('/')[-1], method='discard' if r['trained_discard'] else 'nodiscard', layers=r['layers'],
               seed=r['seed'], epoch=ep, acc_ps=float((v.ps_pos > v.ps_neg).mean()), acc_tr=float((v.tr_pos > v.tr_neg).mean()),
               purity=float(v.purity.median()), p0=float(v.p0.median()), lmax=float(v.lmax.median()),
               margin_own=float((v[own + '_pos'] - v[own + '_neg']).median()), score_sd=float(np.r_[v[own + '_pos'], v[own + '_neg']].std()),
               D_global=float(Dg), D_trace=float(Dt), D_postsel=float(Dp),
               effdim_tr=pr(r['val_bloch']['bloch_tr']), effdim_ps=pr(r['val_bloch']['bloch_ps']), effdim_img=pr(r['val_bloch']['img_bloch']))
    if 'train' in r: row['train_acc_own'] = float((r['train'][own + '_pos'] > r['train'][own + '_neg']).mean())
    rows.append(row)
rows.sort(key=lambda x: (x['run'], x['epoch']))
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out, 'x', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
cur = None
for x in rows:
    if x['run'] != cur:
        cur = x['run']; print(f"\n== {cur} ({x['method']}, L{x['layers']})\nepoch acc_ps acc_tr purity  p0    margin score_sd D_glob D_tr  D_ps  effdim_tr/ps/img")
    print(f"{x['epoch']:5d} {x['acc_ps']:.3f}  {x['acc_tr']:.3f}  {x['purity']:.3f}  {x['p0']:.3f} {x['margin_own']:.3f}  {x['score_sd']:.3f}   "
          f"{x['D_global']:.3f}  {x['D_trace']:.3f} {x['D_postsel']:.3f} {x['effdim_tr']:.1f}/{x['effdim_ps']:.1f}/{x['effdim_img']:.1f}")
