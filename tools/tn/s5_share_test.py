"""Step-5 core sharing (text.share_args) in the MPS decomposer (CPU, no data): share_args=None is the step-4c decomposer; with
'leg_dist' / 'leg' the cores keep their indices and shapes (only names change), the word-leg core keeps its per-type name,
output cores follow share_out, and every shared name has one shape.
   PYTHONPATH=. python tools/tn/s5_share_test.py"""
import itertools, random
from modules.compilation.classical.decompositions import MPS

D = {'n': 128, 's': 128, 'p': 128, 'conj': 128, 'out': 512}
rng = random.Random(0); ok = True
types = []
for _ in range(2000):
    L = rng.randint(2, 12); t = [rng.choice(['n', 's', 'p', 'conj']) for _ in range(L)]
    if rng.random() < 0.6: t[rng.randrange(L)] = 'out'
    types.append(t)
def legs(t):
    mid = len(t) // 2
    return 't:' + '@'.join(t), [D[x] for x in t[:mid]] + [128] + [D[x] for x in t[mid:]], t[:mid] + ['W'] + t[mid:]
for ob, so in ((None, False), (16, True)):
    names = {}
    for mode in ('leg_dist', 'leg'):
        shapes = {}
        for t in types:
            sym, dims, lg = legs(t); idx = list(range(len(dims)))
            a = MPS(8, 2, None, ob, so).decompose(sym, idx, dims, itertools.count(100))
            b = MPS(8, 2, None, ob, so, mode).decompose(sym, idx, dims, itertools.count(100))
            ok &= [(i, d) for _, i, d in a] == [(i, d) for _, i, d in b]          # same network, only names differ
            mid = (len(dims) - 1) // 2
            for k, ((na, _, _), (nb, _, d)) in enumerate(zip(a, b)):
                if k == mid or lg[k] == 'out':
                    ok &= na == nb                                                  # word core / output core unchanged
                else:
                    ok &= nb.startswith('a:' + lg[k] + '.')
                    shapes.setdefault(nb, set()).add(tuple(d))
        ok &= all(len(v) == 1 for v in shapes.values())
        names[mode] = len(shapes)
    print(f'out_bond={ob} share_out={so}: shared names leg_dist {names["leg_dist"]}, leg {names["leg"]}')
print('ALL OK' if ok else 'FAIL')
