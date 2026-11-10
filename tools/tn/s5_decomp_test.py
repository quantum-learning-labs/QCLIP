"""Step-5 decomposers (CPU, small dims): MPS / TR / CP of random complex types contract to a tensor with exactly the type's legs in
order (open indices = idx_arr), shared names (share_args, share_out) always have one shape, the word-leg factor keeps its per-type
name, and CP at full rank represents a dense tensor exactly (explicit construction).
   PYTHONPATH=. python tools/tn/s5_decomp_test.py"""
import itertools, random
import numpy as np, opt_einsum as oe
from modules.compilation.classical.decompositions import MPS, TR, CP

D = {'n': 3, 's': 2, 'p': 2, 'conj': 2, 'out': 4}
rng = random.Random(0); nrng = np.random.default_rng(0); ok = True
types = []
for _ in range(300):
    L = rng.randint(2, 6); t = [rng.choice(['n', 's', 'p', 'conj']) for _ in range(L)]
    if rng.random() < 0.6: t[rng.randrange(L)] = 'out'
    types.append(t)
def legs(t):
    mid = len(t) // 2
    return 't:' + '@'.join(t), [D[x] for x in t[:mid]] + [3] + [D[x] for x in t[mid:]]
def contract(cores):
    subs = ','.join(''.join(oe.get_symbol(i) for i in idx) for _, idx, _ in cores)
    free = [i for i in sorted({i for _, idx, _ in cores for i in idx}) if sum(i in idx for _, idx, _ in cores) == 1]
    return subs, free
decs = {'mps': lambda s, a: MPS(2, 2, None, 3 if s else None, s, a), 'tr': lambda s, a: TR(2, 2, None, 3 if s else None, s, a),
        'cp': lambda s, a: CP(3, 2, s, a)}
for kind, mk in decs.items():
    for so, sa in ((False, None), (True, 'leg_dist'), (True, 'leg')):
        shapes = {}
        for t in types:
            sym, dims = legs(t); idx = list(range(100, 100 + len(dims)))
            cores = mk(so, sa).decompose(sym, idx, dims, itertools.count(1000))
            subs, free = contract(cores)
            ok &= free == idx                                                   # open legs are the type's legs
            mid = (len(dims) - 1) // 2
            ok &= cores[mid][0] == f'{sym}_core{mid}'                            # word factor per type
            for n, _, s in cores:
                if not n.startswith('t:'): shapes.setdefault(n, set()).add(tuple(s))
            arrs = [nrng.standard_normal(s) for _, _, s in cores]
            ok &= oe.contract(subs + '->' + ''.join(oe.get_symbol(i) for i in idx), *arrs).shape == tuple(dims)
        ok &= all(len(v) == 1 for v in shapes.values())
        print(f'{kind:3s} share_out={so!s:5s} share_args={sa!s:8s}: shared names {len(shapes)}', 'ok' if ok else 'FAIL')
# semantics: CP at rank prod(dims[:-1]) represents any dense tensor (one-hot factors select a multi-index, the last factor holds T)
T = nrng.standard_normal((3, 3, 2)); dims = [3, 3, 2]; R = 9
cores = CP(R, 2).decompose('t:n@s', [100, 101, 102], dims, itertools.count(1000))
digits = np.array(list(np.ndindex(3, 3)))                                    # r -> (i0, i1)
arrs = [np.eye(3)[:, digits[:, 0]], np.eye(3)[:, digits[:, 1]], T.reshape(R, 2).T]
subs, _ = contract(cores)
err = np.abs(oe.contract(subs + '->' + ''.join(oe.get_symbol(i) for i in [100, 101, 102]), *arrs) - T).max()
print(f'CP full-rank construction: max err {err:.1e}'); ok &= err < 1e-12
print('ALL OK' if ok else 'FAIL')
