"""E2 rule check on the training wiring (user, 2026-11-11: the parse-derived diagrams are enough, no re-parse). For each word
occurrence in a sample of COCO training captions, every leg is followed along its wire to the other endpoint (word j, leg k);
a free wire is the caption output. Per signature, the distribution of wiring patterns:
  coarse: which legs share one neighbour word (A, B, C... in leg order; '*' = free/output wire)
  fine:   coarse + the neighbour's signature and which of its legs the wire lands on
  python tools/tn/s5_e2_wiring.py <out.json> [n_captions=100000] [top_signatures=30]   (CPU only, run from qclip/)"""
import sys, json, collections
from pathlib import Path
import pandas as pd

out_path = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000; TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
if out_path.exists(): raise FileExistsError(out_path)
tr = pd.read_pickle('data/mscoco/processed/tn/train.pkl')
dcol = [c for c in tr.columns if c.endswith('_diagram')][0]
coarse, fine, tok, n = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), collections.Counter(), 0
examples = collections.defaultdict(dict)
for caps in tr[dcol]:
    for d in (caps or []):
        if n >= N: break
        n += 1
        ends = collections.defaultdict(list)   # wire id -> [(word index, leg index)]
        for wi, (_, wires, _) in enumerate(d):
            for li, w in enumerate(wires): ends[w].append((wi, li))
        for wi, (word, wires, types) in enumerate(d):
            sig = '@'.join(types); tok[sig] += 1
            letters, cs, fs = {}, [], []
            for li, w in enumerate(wires):
                other = [(j, k) for j, k in ends[w] if (j, k) != (wi, li)]
                if not other:
                    cs.append('*'); fs.append('*'); continue
                j, k = other[0]
                g = letters.setdefault(j, 'ABCDEFGHIJ'[len(letters)])
                cs.append(g); fs.append(f"{g}:{'@'.join(d[j][2])}[{k}]")
            c, f = ''.join(cs), ' '.join(fs)
            coarse[sig][c] += 1; fine[sig][f] += 1
            if f not in examples[sig] and len(examples[sig]) < 40:
                examples[sig][f] = ' '.join(x[0] for x in d) + f'   <- word "{word}"'
    if n >= N: break
res = {'n_captions': n, 'signatures': {}}
for sig, c in tok.most_common(TOP):
    res['signatures'][sig] = {'tokens': c,
                              'coarse': [(p, k / c) for p, k in coarse[sig].most_common(6)],
                              'fine': [(p, k / c, examples[sig].get(p, '')) for p, k in fine[sig].most_common(8)]}
json.dump(res, open(out_path, 'x'), indent=1)
for sig, v in res['signatures'].items():
    print(f"== {sig}  tokens {v['tokens']}  coarse {[(p, round(x, 3)) for p, x in v['coarse'][:4]]}")
    for p, x, e in v['fine'][:5]: print(f"     {x:.3f}  {p:60s} e.g. {e[:110]}")
