"""E2 rule R1 table (root independence). unify_codomain writes 'out' into the root word's own type, replacing the atomic type
of its free wire, which the stored diagrams no longer record. For each (word, root signature) in the COCO training captions,
the free wire's type X in {n, s, p} is the one whose non-root signature (root signature with out -> X) that SAME word uses most
often; a word never seen non-root falls back to the most frequent candidate signature overall. No candidate seen -> no entry
(the root stays as it is). Writes {"by_word": {"<word>|<root sig>": X}, "global": {"<root sig>": X}, "stats": ...}.
  python tools/tn/s5_e2_rootmap.py <out.json>   (CPU only, run from qclip/)"""
import sys, json, collections
from pathlib import Path
import pandas as pd

out_path = Path(sys.argv[1])
if out_path.exists(): raise FileExistsError(out_path)
tr = pd.read_pickle('data/mscoco/processed/tn/train.pkl')
dcol = [c for c in tr.columns if c.endswith('_diagram')][0]
tok, wsig, roots = collections.Counter(), collections.Counter(), collections.Counter()
for caps in tr[dcol]:
    for d in (caps or []):
        for w, _, t in d:
            s = '@'.join(t); w = w.lower(); tok[s] += 1
            if 'out' in t: roots[(w, s)] += 1
            else: wsig[(w, s)] += 1
def cands(s):
    i = s.split('@').index('out'); legs = s.split('@')
    return {x: '@'.join(legs[:i] + [x] + legs[i + 1:]) for x in 'nsp'}
glob, by_word, st = {}, {}, collections.Counter()
for s in {s for _, s in roots}:
    c = {x: tok[cs] for x, cs in cands(s).items()}
    if max(c.values()) > 0: glob[s] = max(c, key=c.get)
for (w, s), n in roots.items():
    c = {x: wsig[(w, cs)] for x, cs in cands(s).items()}
    if max(c.values()) > 0:
        by_word[f'{w}|{s}'] = max(c, key=c.get); st['by_word_tokens'] += n
    elif s in glob:
        st['global_tokens'] += n
    else:
        st['unmapped_tokens'] += n
st['root_tokens'] = sum(roots.values())
res = {'by_word': by_word, 'global': glob, 'stats': dict(st)}
json.dump(res, open(out_path, 'x'), indent=0, sort_keys=True)
print(json.dumps(res['stats']), len(by_word), 'word entries', len(glob), 'global entries')
for k in ('with|n@out@n', 'cutting|n@out@n', 'is|n@out@n', 'on|s@n@n@out@n', 'is|n@out@s@n', 'filled|n@out@p', 'is|n@out@p'):
    print(' ', k, by_word.get(k))
