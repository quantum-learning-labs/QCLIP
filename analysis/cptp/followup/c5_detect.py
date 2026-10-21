# C5 test, part 3: error detection. Same post-selected circuits, same shots, scored two ways:
# post-selected swap test (accepted shots only) and the unconditioned swap test over all shots (= the traced readout of
# that circuit). If rejecting shots detects errors, the post-selected margins should contract less under honest noise.
# usage: c5_detect.py <dir with noiseless_*.json and honest_*.json>
import json, glob, os, sys, statistics as st
D = sys.argv[1]
def load(f):
    r = json.load(open(f))['results']['by_shots']['2048']; L = r['per_circuit_log']; n = len(L) // 2
    pos, neg = L[:n], L[n:]
    ps = lambda c: c[1] / c[0] if c[0] else 0.0          # accepted signal / accepted fraction
    un = lambda c: c[2]                                  # unconditioned signal over all shots
    return dict(acc=[(p[0] + q[0]) / 2 for p, q in zip(pos, neg)],
                d_ps=[ps(p) - ps(q) for p, q in zip(pos, neg)], d_un=[un(p) - un(q) for p, q in zip(pos, neg)],
                f_ps=[ps(p) for p in pos], f_un=[un(p) for p in pos])
k = lambda a, b: sum(x * y for x, y in zip(a, b)) / sum(x * x for x in a)
print(f"{'model':18s} readout          F_pos 0 -> h      acc 0 -> h       k (margin contraction)")
for f in sorted(glob.glob(f'{D}/noiseless_*.json')):
    g = f.replace('noiseless_', 'honest_')
    if not os.path.exists(g): continue
    a, b = load(f), load(g); m = os.path.basename(f)[len('noiseless_'):-5]
    for ro in ('ps', 'un'):
        d0, dh = a[f'd_{ro}'], b[f'd_{ro}']
        print(f"{m:18s} {'post-selected' if ro == 'ps' else 'unconditioned':15s}  {st.mean(a[f'f_{ro}']):.3f} -> {st.mean(b[f'f_{ro}']):.3f}    "
              f"{sum(x > 0 for x in d0) / len(d0):.3f} -> {sum(x > 0 for x in dh) / len(dh):.3f}    {k(d0, dh):.3f}")
    print(f"{'':18s} acceptance {st.mean(a['acc']):.4f} -> {st.mean(b['acc']):.4f}")
