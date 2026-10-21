# Side by side: noiseless vs baseline-noise vs honest-noise emulation, same 300 std pairs, 2048 shots.
# usage: noise_summ.py <noiseless json dir> <noise json dir>
import json, glob, os, re, sys, statistics as st, collections
nl_dir, nz_dir = sys.argv[1], sys.argv[2]; N = 300; S = '2048'
rows = collections.defaultdict(lambda: collections.defaultdict(list))
def key(name):
    m = re.match(r'(?:(backend|honest)_)?(nodiscard|discard|discardWS)_(L\w+)_s(\d+)$', name); return m.groups() if m else None
for f in glob.glob(os.path.join(nl_dir, '*.json')):
    k = key(os.path.basename(f)[:-5])
    if not k or k[0]: continue
    r = json.load(open(f))['results']['by_shots'][S]
    accs = [sum(p > n for p, n in zip(P[:N], Q[:N])) / N for P, Q in zip(r['per_pair_pos_fidelities'], r['per_pair_neg_fidelities'])]
    rows[(k[1], k[2])]['noiseless'].append((int(k[3]), st.mean(accs), 1.0))
for f in glob.glob(os.path.join(nz_dir, '*.json')):
    k = key(os.path.basename(f)[:-5])
    if not k or not k[0]: continue
    r = json.load(open(f))['results']['by_shots'][S]
    rows[(k[1], k[2])][k[0]].append((int(k[3]), r['accuracy'], r['accept_frac_mean']))
print(f"{'model':22s} {'noiseless':>16s} {'baseline noise':>16s} {'honest noise':>16s}  accepted shots (noiseless/base/honest)")
for k in sorted(rows):
    d = rows[k]; seeds = sorted(set.intersection(*[set(s for s, _, _ in d[c]) for c in ('noiseless', 'backend', 'honest') if d[c]]) or {-1})
    cell = lambda c: (lambda v: f"{st.mean(v):.3f}±{(st.stdev(v) if len(v) > 1 else 0):.3f}")([a for s, a, _ in d[c] if s in seeds]) if d[c] else 'n/a'
    acc = lambda c: f"{st.mean([x for s, _, x in d[c] if s in seeds]):.3f}" if d[c] else 'n/a'
    print(f"{k[0] + ' ' + k[1]:22s} {cell('noiseless'):>16s} {cell('backend'):>16s} {cell('honest'):>16s}  {acc('noiseless')}/{acc('backend')}/{acc('honest')}  seeds {seeds}")
