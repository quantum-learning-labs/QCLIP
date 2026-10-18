# C5 / crossover / distillation follow-ups (2026-10-18), report sections 15-17. Run on the MSI copies of the emulation JSONs (stdlib only).
# C5 test, part 1: does honest noise hurt discard more because its margins are smaller?
# Per pair: noiseless margin d0 = F(pos) - F(neg) (mean of 3 shot runs, 2048 shots) and honest-noise margin dh (1 run).
# Reports the margin contraction k (least squares dh ~ k*d0), the flip rate (correct -> wrong) binned by |d0|,
# and the median |d0|. usage: c5_margin.py <noiseless dir> <noise dir> [out.json]
import json, glob, os, re, sys, statistics as st, collections
nl, nz = sys.argv[1], sys.argv[2]; N = 300; S = '2048'
BINS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
rows, pairs = {}, collections.defaultdict(list)
for f in sorted(glob.glob(f'{nz}/honest_*.json')):
    name = os.path.basename(f)[len('honest_'):-5]
    g = f'{nl}/{name}.json'
    if not os.path.exists(g): continue
    h = json.load(open(f))['results']['by_shots'][S]; z = json.load(open(g))['results']['by_shots'][S]
    P0 = [st.mean(r[i] for r in z['per_pair_pos_fidelities']) for i in range(N)]
    Q0 = [st.mean(r[i] for r in z['per_pair_neg_fidelities']) for i in range(N)]
    Ph, Qh = h['per_pair_pos_fidelities'][0][:N], h['per_pair_neg_fidelities'][0][:N]
    model = re.sub(r'_s\d+$', '', name)
    for i in range(N): pairs[model].append((P0[i] - Q0[i], Ph[i] - Qh[i], P0[i], Q0[i], Ph[i], Qh[i]))
out = {}
print(f"{'model':16s} pairs  med|d0|   k(dh~d0)  mean F0(pos,neg)   mean Fh(pos,neg)   acc0  acch  flips c->w  w->c")
for m, xs in pairs.items():
    d0 = [x[0] for x in xs]; dh = [x[1] for x in xs]
    k = sum(a * b for a, b in zip(d0, dh)) / sum(a * a for a in d0)
    acc0 = sum(a > 0 for a in d0) / len(d0); acch = sum(b > 0 for b in dh) / len(dh)
    cw = sum(a > 0 >= b for a, b in zip(d0, dh)) / len(d0); wc = sum(b > 0 >= a for a, b in zip(d0, dh)) / len(d0)
    f0 = (st.mean(x[2] for x in xs), st.mean(x[3] for x in xs)); fh = (st.mean(x[4] for x in xs), st.mean(x[5] for x in xs))
    binned = []
    for lo, hi in zip(BINS, BINS[1:]):
        sel = [(a, b) for a, b in zip(d0, dh) if a > 0 and lo <= a < hi]
        binned.append(dict(lo=lo, hi=hi, n=len(sel), flip=(sum(b <= 0 for a, b in sel) / len(sel)) if sel else None))
    out[m] = dict(pairs=len(xs), median_abs_d0=st.median(abs(a) for a in d0), k=k, acc0=acc0, acch=acch, flip_cw=cw, flip_wc=wc,
                  F0=f0, Fh=fh, flip_by_margin=binned)
    print(f"{m:16s} {len(xs):5d}  {out[m]['median_abs_d0']:.4f}   {k:.3f}     {f0[0]:.3f},{f0[1]:.3f}        {fh[0]:.3f},{fh[1]:.3f}        {acc0:.3f} {acch:.3f}  {cw:.3f}      {wc:.3f}")
print('\nflip rate of correctly-ranked pairs by noiseless margin d0 (n in brackets)')
print(f"{'model':16s} " + ' '.join(f'[{lo},{hi})'.rjust(14) for lo, hi in zip(BINS, BINS[1:])))
for m, o in out.items():
    print(f"{m:16s} " + ' '.join((f"{b['flip']:.2f} ({b['n']})" if b['n'] else '-').rjust(14) for b in o['flip_by_margin']))
if len(sys.argv) > 3: json.dump(out, open(sys.argv[3], 'w'), indent=1)
