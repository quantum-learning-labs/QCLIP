"""Step 3: MMEvaluator._calculate_recall (expected metrics under random tie-breaking) vs the old argsort metric averaged
over many random tie orders, on score matrices with forced exact ties.   PYTHONPATH=. python tools/tn/check_recall.py"""
import torch
from modules.models.fusion.engine import MMEvaluator

def old_metric(scores, mask):   # the pre-step-3 argsort code
    rankings = scores.argsort(dim=1, descending=True)
    positions = torch.argmax(mask.gather(1, rankings).float(), dim=1)
    return {f"R{k}": (positions < k).float().mean().item() for k in (1, 5, 10)} | {"mrr": (1.0 / (positions.float() + 1)).mean().item()}

g = torch.Generator().manual_seed(0)
n, m = 200, 1000
scores = torch.randint(0, 40, (n, m), generator=g).double()   # few distinct values: many exact ties
mask = torch.zeros(n, m, dtype=torch.bool)
mask[torch.arange(n), torch.randint(0, m, (n,), generator=g)] = True
mask[torch.arange(n), torch.randint(0, m, (n,), generator=g)] = True
new = MMEvaluator._calculate_recall(None, scores, mask, '')
new['mrr'] = new['_mrr']
acc = {k: 0.0 for k in ('R1', 'R5', 'R10', 'mrr')}; reps = 3000
for _ in range(reps):   # random tie order: a random column permutation, then the old (sort-based) metric
    perm = torch.randperm(m, generator=g)
    o = old_metric(scores[:, perm], mask[:, perm])
    for k in acc: acc[k] += o[k] / reps
print({k: (round(new[k], 4), round(acc[k], 4)) for k in acc}, 'tie_frac', new['_tie_frac'])
assert all(abs(new[k] - acc[k]) < 4e-3 for k in acc), 'mismatch'
s2 = torch.randn(n, m, dtype=torch.float64, generator=g)   # no ties: exactly the old metric
o, nn = old_metric(s2, mask), MMEvaluator._calculate_recall(None, s2, mask, '')
nn['mrr'] = nn['_mrr']
assert all(abs(nn[k] - o[k]) < 1e-6 for k in o), (nn, o)   # old MRR mean in float32
print('check_recall OK')
nanm = MMEvaluator._calculate_recall(None, torch.full((n, m), float('nan'), dtype=torch.float64), mask, '')   # diverged model
assert abs(nanm['R1'] - 1 / (m - 1)) < 1e-9 and nanm['R10'] < 0.02, nanm   # chance (uniform rank), not 1.0
print('NaN scores -> chance', {k: round(v, 5) for k, v in nanm.items()})
