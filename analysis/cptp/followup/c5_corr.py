# Item: correlate each pair's noiseless margin d0 with its drop under honest noise (d0 - dh), both readouts.
# Also Spearman, the correlation of d0 with the noisy margin dh, and the flip rate of correct pairs vs |d0| tercile.
import json, glob, os, re, sys, math, statistics as st, collections
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(o): r[i] = k
    return r
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b); sa, sb = st.pstdev(a), st.pstdev(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) * sa * sb)
def pairs(noiseless, noisy, N=300):
    z = json.load(open(noiseless))['results']['by_shots']['2048']; h = json.load(open(noisy))['results']['by_shots']['2048']
    d0 = [st.mean(r[i] for r in z['per_pair_pos_fidelities']) - st.mean(r[i] for r in z['per_pair_neg_fidelities']) for i in range(N)]
    dh = [a - b for a, b in zip(h['per_pair_pos_fidelities'][0][:N], h['per_pair_neg_fidelities'][0][:N])]
    return d0, dh
SETS = {  # model: list of (noiseless json, honest json)
 'm=4 post-selection L3': [(f'emu/out/nodiscard_L3_s{s}.json', f'emu/out_noise_all/honest_nodiscard_L3_s{s}.json') for s in (0, 1)],
 'm=4 discard L3 pw0':    [(f'emu/out/discard_L3pw0_s{s}.json', f'emu/out_noise_all/honest_discard_L3pw0_s{s}.json') for s in (0, 1)],
 'm=4 discard warm start':[(f'emu/out/discardWS_L3_s{s}.json', f'emu/out_noise_all/honest_discardWS_L3_s{s}.json') for s in (0, 1)],
 'm=4 post-selection L2': [(f'emu/out/nodiscard_L2_s{s}.json', f'emu/out_noise_all/honest_nodiscard_L2_s{s}.json') for s in (0, 1)],
 'm=4 discard L2':        [(f'emu/out/discard_L2_s{s}.json', f'emu/out_noise_all/honest_discard_L2_s{s}.json') for s in (0, 1)],
 'm=8 post-selection':    [(f'xo/emu/nodiscard_n2_tl3_qlr0.003_s{s}.json', f'emu/m8/out/honest_nodiscard_m8_s{s}.json') for s in (0, 1)],
 'm=8 discard':           [(f'xo/emu/discard_n2_tl3_pw0_s{s}.json', f'emu/m8/out/honest_discard_m8_s{s}.json') for s in (0, 1)],
 'm=8 discard warm start':[(f'xo/emu/discard_n2_tl3_pw0_ws_s{s}.json', f'emu/m8/out/honest_discardWS_m8_s{s}.json') for s in (0, 1)],
}
out = {}
print(f"{'model':24s} pairs  r(d0,drop)  rho(d0,drop)  r(d0,dh)  relative drop  flip rate of correct pairs by |d0| tercile (low/mid/high)")
for m, fl in SETS.items():
    D0, DH = [], []
    for a, b in fl:
        if os.path.exists(a) and os.path.exists(b): d0, dh = pairs(a, b); D0 += d0; DH += dh
    if not D0: print(m, 'missing'); continue
    drop = [a - b for a, b in zip(D0, DH)]
    r1 = pearson(D0, drop); r2 = pearson(rank(D0), rank(drop)); r3 = pearson(D0, DH)
    rel = st.mean(drop) / st.mean([abs(x) for x in D0])
    pos = sorted([(a, b) for a, b in zip(D0, DH) if a > 0]); t = len(pos) // 3
    flips = [sum(b <= 0 for _, b in g) / len(g) for g in (pos[:t], pos[t:2 * t], pos[2 * t:])]
    out[m] = dict(pairs=len(D0), r_d0_drop=r1, spearman_d0_drop=r2, r_d0_dh=r3, flips_by_tercile=flips)
    print(f"{m:24s} {len(D0):5d}   {r1:+.2f}        {r2:+.2f}        {r3:+.2f}     {rel:+.2f}           {flips[0]:.2f} / {flips[1]:.2f} / {flips[2]:.2f}")
json.dump(out, open('gap/c5_corr.json', 'w'), indent=1)
