# C5 test, part 2: predict honest-noise flips from margin contraction k and binomial shot noise on the KEPT shots.
# P(flip) = Phi(-k d0 / sigma_h), sigma_h^2 = Fp(1-Fp)/n + Fq(1-Fq)/n with n = 2048 x acceptance (post-selection) or 2048.
import json, glob, os, re, sys, math, statistics as st, collections
nl, nz = sys.argv[1], sys.argv[2]; N = 300; S = '2048'
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
data = collections.defaultdict(list); acc = {}
for f in sorted(glob.glob(f'{nz}/honest_*.json')):
    name = os.path.basename(f)[len('honest_'):-5]; g = f'{nl}/{name}.json'
    if not os.path.exists(g): continue
    h = json.load(open(f))['results']['by_shots'][S]; z = json.load(open(g))['results']['by_shots'][S]
    m = re.sub(r'_s\d+$', '', name); acc.setdefault(m, []).append(h['accept_frac_mean'])
    for i in range(N):
        d0 = st.mean(r[i] for r in z['per_pair_pos_fidelities']) - st.mean(r[i] for r in z['per_pair_neg_fidelities'])
        data[m].append((d0, h['per_pair_pos_fidelities'][0][i], h['per_pair_neg_fidelities'][0][i]))
print(f"{'model':16s} kept shots   k     median z_h   pred acc_h  obs acc_h   noiseless")
for m, xs in data.items():
    n = 2048 * st.mean(acc[m])
    d0 = [x[0] for x in xs]; dh = [x[1] - x[2] for x in xs]
    k = sum(a * b for a, b in zip(d0, dh)) / sum(a * a for a in d0)
    sig = [math.sqrt(max(p * (1 - p), 1e-4) / n + max(q * (1 - q), 1e-4) / n) for _, p, q in xs]
    zs = [k * a / s for a, s in zip(d0, sig)]
    pred = st.mean(Phi(zz) for zz in zs)      # P(noisy margin > 0) given the contracted mean margin k*d0
    print(f"{m:16s} {n:8.0f}   {k:.3f}   {st.median(abs(zz) for zz in zs):6.2f}      {pred:.3f}      {sum(b > 0 for b in dh) / len(dh):.3f}     {sum(a > 0 for a in d0) / len(d0):.3f}")

# excess scatter: residual r = dh - k d0 against the shot-noise floor; with the fitted extra variance, re-predict
print(f"\n{'model':16s} resid sd   shot sd   excess sd   excess/median|k d0|   pred acc_h (shots+excess)")
for m, xs in data.items():
    n = 2048 * st.mean(acc[m]); d0 = [x[0] for x in xs]; dh = [x[1] - x[2] for x in xs]
    k = sum(a * b for a, b in zip(d0, dh)) / sum(a * a for a in d0)
    sig = [math.sqrt(max(p * (1 - p), 1e-4) / n + max(q * (1 - q), 1e-4) / n) for _, p, q in xs]
    r = [b - k * a for a, b in zip(d0, dh)]; rs = st.pstdev(r); ss = math.sqrt(st.mean(s * s for s in sig))
    ex = math.sqrt(max(rs ** 2 - ss ** 2, 0))
    pred = st.mean(Phi(k * a / math.sqrt(s * s + ex * ex)) for a, s in zip(d0, sig))
    print(f"{m:16s} {rs:.4f}    {ss:.4f}    {ex:.4f}      {ex / st.median(abs(k * a) for a in d0):.2f}                  {pred:.3f} (obs {sum(b > 0 for b in dh) / len(dh):.3f})")
