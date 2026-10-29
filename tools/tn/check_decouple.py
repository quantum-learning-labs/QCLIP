"""Step-3 checks of the decoupled compiler (CPU, no data): einsum vs a hand-built contraction, sharing of word vectors
and type tensors, MPS only on the type tensor.   PYTHONPATH=. python tools/tn/check_decouple.py"""
import torch, opt_einsum as oe
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS

ob = {'n': 3, 's': 4, 'p': 2, 'out': 5, 'sem': 6}
# "dogs chase cats": dog n (idx 1), chase n.r@out@n.l (1, 0, 2), cats n (2); "chase" also used as a noun below
tn = [('dogs', [1], ['n']), ('chase', [1, 0, 2], ['n.r', 'out', 'n.l']), ('cats', [2], ['n'])]
dense = TNCompiler(ob, None, decouple=True)
eq, syms = dense.compile_tn(tn)
names = [s['name'] for s in syms]
print(eq, names)
assert names == ['w:dogs', 't:n', 'w:chase', 't:n.r@out@n.l', 'w:cats', 't:n']
assert [s['shape'] for s in syms] == [[6], [6, 3], [6], [1, 6, 5, 1], [6], [6, 3]]   # n.r/n.l dims default 1 here
g = torch.Generator().manual_seed(0)
P = {n: torch.randn(*s['shape'], generator=g, dtype=torch.float64) for n, s in zip(names, syms)}
out = oe.contract(eq, *[P[n] for n in names])
ref = torch.einsum('a,ai,b,ibkj,c,cj->k', P['w:dogs'], P['t:n'], P['w:chase'], P['t:n.r@out@n.l'], P['w:cats'], P['t:n'])
assert torch.allclose(out, ref), (out, ref)
# a word used with two types shares one vector
_, s2 = dense.compile_tn([('chase', [0], ['out'])])
assert s2[0]['name'] == 'w:chase' and s2[1]['name'] == 't:out'
# thresholds: max_order 1 decomposes the order-2 noun type, max_order 2 keeps it dense; words never decomposed
ob2 = dict(ob, n=128, sem=128)
for mo, want in ((1, ['w:dogs', 't:n_core0', 't:n_core1']), (2, ['w:dogs', 't:n'])):
    _, s = TNCompiler(ob2, MPS(bond_dim=8, max_order=mo), decouple=True).compile_tn([('dogs', [0], ['n'])])
    assert [x['name'] for x in s] == want, (mo, s)
_, s = TNCompiler(ob2, MPS(bond_dim=8, max_order=2), decouple=True).compile_tn(tn)
print([(x['name'], x['shape']) for x in s])
assert all(len(x['shape']) == 1 for x in s if x['name'].startswith('w:'))
# decouple off: unchanged symbols
_, s = TNCompiler(ob2, MPS(bond_dim=8, max_order=1)).compile_tn([('dogs', [0], ['n'])])
assert [x['name'] for x in s] == ['dogs_n']
print('check_decouple: all OK')
