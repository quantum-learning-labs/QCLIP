# Exact rejection accounting for post-selected circuits (density matrix, 2^20 shots, full counts).
# Per circuit and noise setting: acceptance, and coupling-free LOWER BOUNDS on
#   lost  = ideal-accepted probability that noise moved out of the accepted set  (sum over accepted x of max(0, p - q))
#   gained = noisy accepted probability that is not in the ideal accepted distribution (sum over accepted x of max(0, q - p))
# plus the share of the noisy accepted shots that is "foreign" (gained / noisy acceptance).
import json, sys, statistics as st, collections
D = sys.argv[1]
def load(name, s):
    r = json.load(open(f'{D}/{name}_s{s}.json'))['results']['by_shots']['1048576']; return r['per_circuit_counts']
def accepted_dist(c):
    n_out = c['n_out']; n_txt = c['num_clbits'] - n_out; out = set(c['out_qs'])
    aux = [q for q in range(n_txt) if q not in out]
    tot = sum(c['counts'].values()); acc = {}
    for b, k in c['counts'].items():
        rb = b[::-1]
        if all(rb[q] == '0' for q in aux): acc[b] = k / tot
    return acc, len(aux)
res = collections.defaultdict(list)
for s in (0, 1):
    ideal = load('ideal', s)
    for cfg in ('honest', 'readout_only', 'gates_only'):
        noisy = load(cfg, s)
        for ci, cn in zip(ideal, noisy):
            p, m = accepted_dist(ci); q, _ = accepted_dist(cn)
            pa, qa = sum(p.values()), sum(q.values())
            keys = set(p) | set(q)
            lost = sum(max(0, p.get(k, 0) - q.get(k, 0)) for k in keys)
            gained = sum(max(0, q.get(k, 0) - p.get(k, 0)) for k in keys)
            res[cfg].append(dict(m=m, pa=pa, qa=qa, lost=lost, gained=gained, foreign=gained / qa if qa else 0))
print(f"{'noise':14s} circuits  m   accept ideal -> noisy   lost (>=)  gained (>=)  foreign share of kept shots (>=)")
out = {}
for cfg, rows in res.items():
    g = lambda k: st.mean(r[k] for r in rows)
    out[cfg] = {k: g(k) for k in ('pa', 'qa', 'lost', 'gained', 'foreign')}
    print(f"{cfg:14s} {len(rows):7d}  {st.mean(r['m'] for r in rows):.1f}   {g('pa'):.4f} -> {g('qa'):.4f}      {g('lost'):.4f}     {g('gained'):.4f}      {g('foreign'):.3f}")
json.dump(out, open('gap/rej_account.json', 'w'), indent=1)
