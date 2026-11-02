"""Step-4c output-span options of the MPS decomposer (CPU, no data): defaults reproduce the old decomposer exactly; out_bond widens
only the two bonds next to the output core; share_out names that core 'o:out_*' with one shape per position.
   PYTHONPATH=. python tools/tn/s4c_decomp_test.py"""
import itertools, random
from modules.compilation.classical.decompositions import MPS

def old_decompose(bond, max_order, root_bond, symbol, idx_arr, dims, ctr):   # decompositions.py at c9d030a, verbatim logic
    order = len(idx_arr)
    if order <= max_order: return [(symbol, idx_arr, dims)]
    b = root_bond if root_bond and 'out' in symbol.split(':')[-1].split('@') else bond
    vb = [next(ctr) for _ in range(order - 1)]; cores = []
    for k in range(order):
        if k == 0: cores.append((f"{symbol}_core{k}", [idx_arr[0], vb[0]], [dims[0], b]))
        elif k == order - 1: cores.append((f"{symbol}_core{k}", [vb[-1], idx_arr[-1]], [b, dims[-1]]))
        else: cores.append((f"{symbol}_core{k}", [vb[k-1], idx_arr[k], vb[k]], [b, dims[k], b]))
    return cores

D = {'n': 128, 's': 128, 'p': 128, 'conj': 128, 'out': 512}
rng = random.Random(0); ok = True
types = []
for _ in range(500):
    L = rng.randint(1, 9); t = [rng.choice(['n', 's', 'p', 'conj']) for _ in range(L)]
    if rng.random() < 0.6: t[rng.randrange(L)] = 'out'
    types.append(t)
def legs(t):
    mid = len(t) // 2; sym = 't:' + '@'.join(t)
    return sym, [D[x] for x in t[:mid]] + [128] + [D[x] for x in t[mid:]], t[:mid] + ['W'] + t[mid:]
for rb in (None, 32):
    for t in types:
        sym, dims, _ = legs(t); idx = list(range(len(dims)))
        a = MPS(8, 2, rb).decompose(sym, idx, dims, itertools.count(100))
        b = old_decompose(8, 2, rb, sym, idx, dims, itertools.count(100))
        ok &= a == b
print('defaults == old decomposer:', ok)
shapes = {}
for ob, share in ((16, False), (16, True), (None, True), (32, True)):
    m = MPS(8, 2, None, ob, share); bad = 0
    for t in types:
        sym, dims, lg = legs(t)
        cores = m.decompose(sym, list(range(len(dims))), dims, itertools.count(100))
        if len(dims) <= 2: continue
        if 'out' not in t:
            bad += any(c[0].startswith('o:') or max(c[2]) > 128 and 8 not in c[2] for c in cores); continue
        k = lg.index('out')
        core = cores[k]
        bad += core[2][[0, 1][0] if k == 0 else 1] != 512 if k < len(dims) - 1 else core[2][-1] != 512
        if share:
            bad += not core[0].startswith('o:out_')
            shapes.setdefault((ob, core[0]), set()).add(tuple(core[2]))
        nb = [x for c in cores for x in (c[2][:1] if c is cores[-1] else c[2][-1:])][:len(dims) - 1]
        widened = sum(x == (ob or 8) for x in nb)
        if ob: bad += widened < (1 if k in (0, len(dims) - 1) else 2)
    print(f'out_bond={ob} share={share}: bad={bad}'); ok &= bad == 0
for k, v in sorted(shapes.items(), key=str):
    print('  shared', k, v); ok &= len(v) == 1
print('ALL OK' if ok else 'FAILED')
