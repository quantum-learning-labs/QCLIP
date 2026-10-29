# ps_swap honest: unmitigated vs finite-variant debias vs twirl limit. Per rep: share of pairs where both scores clamp to 0
# (tie = counted wrong), mean signal; pooled: mean |m - m'| between methods.
import pickle, sys, json
import numpy as np
import bench_lib as B
d = sys.argv[1]; meta = json.load(open(d + '/summary.json'))['meta']; n = meta['pairs']
raw = pickle.load(open(d + '/raw.pkl', 'rb')); infos = raw['infos']
src = {'unmitigated': raw['honest_miami'], 'finite': pickle.load(open(d + '/raw_debias_honest_miami.pkl', 'rb'))['honest_miami_debias'],
       'twirl': pickle.load(open(d + '/raw_debiasT_honest_miami.pkl', 'rb'))['honest_miami_debias_twirl']}
ms = {}
for k, v in src.items():
    st = [B.readout_stats(c, i, meta['meas'], meta['discard']) for c, i in zip(v['counts_rep0'], infos)]
    m = np.array([x[0] for x in st]); s = np.array([x[2] for x in st]); ms[k] = np.array(v['extra']['m'])
    tie = np.mean((s[:n] == 0) & (s[n:] == 0)); acc = np.mean(s[:n] > s[n:])
    print(f'{k:12s} rep0 acc {acc:.3f} both-clamped ties {tie:.3f} neg signals {np.mean(m < 0):.3f} | pooled mean m {ms[k].mean():+.4f} acceptance {np.mean(v["extra"]["a"]):.3f}')
for a, b in [('unmitigated', 'twirl'), ('unmitigated', 'finite'), ('twirl', 'finite')]:
    dm = ms[b] - ms[a]
    print(f'{a} -> {b}: mean dm {dm.mean():+.4f}, mean |dm| {np.abs(dm).mean():.4f}, pooled-signal accuracy {np.mean(ms[a][:n] > ms[a][n:]):.3f} -> {np.mean(ms[b][:n] > ms[b][n:]):.3f}')
