"""E1 (step 5, user-approved 2026-11-12): which (relation word, type symbol) occurrences get their own CP tensor. Closed-class
relation words (prepositions/particles, conjunctions, relative pronouns, copula) = the fixed list REL below; a (word, symbol)
pair qualifies when its symbol is relational (>= 3 legs, or a p leg) and it occurs >= MIN times in the COCO training captions,
counted AFTER the lifting the run uses (symbol = the compiler's type symbol: 't:<sig>' or a lifted 't.aux:'/'t.adv:'/...).
Writes {"lift": ..., "min_count": MIN, "pairs": {"<word>|<symbol>": count}, "stats": ...}.
  PYTHONPATH=. python tools/tn/s5_e1_pairs.py <out.json> <lift: none|R1|frobenius> [MIN=500]   (CPU only, run from qclip/)"""
import sys, json, collections
from itertools import count
from pathlib import Path
import pandas as pd
from modules.compilation.classical.lift import RULES, load_rootmap, lift

REL = set("""on in under above below behind near next to of with at by from into onto over beside besides between across along
alongside through around inside outside against up down beneath underneath atop toward towards past out off for about among beyond
during without via like top front while and or but that which who where whose is are was were be being""".split())
out_path, L = Path(sys.argv[1]), sys.argv[2]
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 500
if out_path.exists(): raise FileExistsError(out_path)
rules = None if L == 'none' else tuple(RULES) if L == 'frobenius' else tuple(L.split(','))
rm = load_rootmap('modules/compilation/classical/e2_rootmap.json') if rules else None
tr = pd.read_pickle('data/mscoco/processed/tn/train.pkl')
cnt, st = collections.Counter(), collections.Counter()
for caps in tr['captions_diagram']:
    for tn in (caps or []):
        if rules:
            mx = max((i for _, ix, _ in tn for i in ix), default=0)
            items = lift(tn, rules, rm, count(mx + 1))[0]
        else:
            items = [(w, i, t, None) for w, i, t in tn]
        for w, _, t, ts in items:
            st['tokens'] += 1
            if w not in REL: continue
            st['rel_tokens'] += 1
            if len(t) >= 3 or 'p' in t:
                cnt[f"{w}|{ts or 't:' + '@'.join(t)}"] += 1
pairs = {k: n for k, n in cnt.most_common() if n >= MIN}
st['rel_relational_tokens'] = sum(cnt.values()); st['own_tokens'] = sum(pairs.values())
st['own_words'] = len({k.split('|')[0] for k in pairs}); st['own_pairs'] = len(pairs)
res = {'lift': L, 'min_count': MIN, 'pairs': pairs, 'stats': dict(st)}
json.dump(res, open(out_path, 'x'), indent=0)
print(json.dumps(res['stats']))
for k, n in list(pairs.items())[:40]: print(' ', k, n)
