"""Step 5c: diagram-level hard negatives for COCO training captions (no re-parse, no new type tensors).
For every caption one negative with the same wiring and types and a different word assignment:
  relation  replace a preposition (>= 3 legs) by its contrast (on -> under, with -> without, ...);
  noun      exchange two different bare nouns (type ['n']);
  modifier  exchange two different adjective/compound modifiers (type ['n','n'], not determiners or numbers) attached to
            different nouns.
The kind is drawn uniformly among the kinds available for the caption (seeded by the caption's position); captions with no
available kind keep their own diagram and captions_neg_valid = False (dropped from the loss). Writes a NEW pickle.
Step 5d: policy prefer_rel takes the first available kind in the order relation > modifier > noun (binding over nouns).
  PYTHONPATH=. python tools/tn/s5_make_negs.py <in.pkl> <out.pkl> [seed=0] [policy=uniform|prefer_rel]"""
import sys, os, random, collections
import pandas as pd
REL = {'on': 'under', 'under': 'on', 'above': 'below', 'below': 'above', 'over': 'under', 'in': 'on', 'inside': 'outside',
       'outside': 'inside', 'with': 'without', 'without': 'with', 'behind': 'beside', 'into': 'from', 'from': 'into',
       'onto': 'off', 'off': 'onto', 'up': 'down', 'down': 'up', 'before': 'after', 'after': 'before', 'towards': 'from'}
DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'each', 'every', 'his', 'her', 'its', 'their', 'my', 'your',
       'our', 'another', 'any', 'no', 'several', 'many', 'few', 'all', 'both', 'other', 'one', 'two', 'three', 'four', 'five', 'six',
       'seven', 'eight', 'nine', 'ten', 'dozen', 'couple', 'lots', 'lot', 'plenty', 'much', 'more', 'most'}

def noun_of(d, i):   # follow a modifier's input wire down to the bare noun it modifies
    out_of = {b[1][0]: k for k, b in enumerate(d)}
    seen = set()
    while list(d[i][2]) == ['n', 'n'] and i not in seen:
        seen.add(i); j = out_of.get(d[i][1][1])
        if j is None: return None
        i = j
    return i if list(d[i][2]) == ['n'] else None

def negative(d, rng, policy='uniform'):
    words = [b[0].lower() for b in d]
    opts = []
    rel = [i for i, b in enumerate(d) if words[i] in REL and len(b[2]) >= 3]
    if rel: opts.append('relation')
    nouns = [i for i, b in enumerate(d) if list(b[2]) == ['n']]
    npairs = [(i, j) for a, i in enumerate(nouns) for j in nouns[a + 1:] if words[i] != words[j]]
    if npairs: opts.append('noun')
    mods = [(i, noun_of(d, i)) for i, b in enumerate(d) if list(b[2]) == ['n', 'n'] and words[i] not in DET]
    mpairs = [(i, j) for a, (i, ni) in enumerate(mods) for j, nj in mods[a + 1:]
              if words[i] != words[j] and ni is not None and nj is not None and ni != nj]
    if mpairs: opts.append('modifier')
    if not opts: return d, False, 'none'
    kind = rng.choice(opts) if policy == 'uniform' else next(k for k in ('relation', 'modifier', 'noun') if k in opts)
    d2 = [list(b) for b in d]
    if kind == 'relation':
        i = rng.choice(rel); d2[i][0] = REL[words[i]]
    else:
        i, j = rng.choice(npairs if kind == 'noun' else mpairs); d2[i][0], d2[j][0] = d[j][0], d[i][0]
    return [tuple(b) for b in d2], True, kind

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    policy = sys.argv[4] if len(sys.argv) > 4 else 'uniform'; assert policy in ('uniform', 'prefer_rel')
    if os.path.exists(out): sys.exit(f'{out} exists; refusing to overwrite')
    df = pd.read_pickle(src)
    negs, valid, kinds, cnt, k = [], [], [], collections.Counter(), 0
    ex = collections.defaultdict(list)
    for caps in df['captions_diagram']:
        rn, rv, rk = [], [], []
        for d in caps:
            rng = random.Random(seed * 10_000_019 + k); k += 1
            n, v, kind = negative(d, rng, policy); rn.append(n); rv.append(v); rk.append(kind); cnt[kind] += 1
            if v and len(ex[kind]) < 4: ex[kind].append((' '.join(b[0] for b in d), ' '.join(b[0] for b in n)))
        negs.append(rn); valid.append(rv); kinds.append(rk)
    df = df.copy(); df['captions_neg_diagram'] = negs; df['captions_neg_valid'] = valid; df['captions_neg_kind'] = kinds
    df.to_pickle(out)
    print(f'captions {k}:', dict(cnt), f'valid {1 - cnt["none"] / k:.3f}')
    for kind, xs in ex.items():
        for a, b in xs: print(f'  [{kind}] {a}  ->  {b}')
