# compare a two-bin npz (shotcost_branches_fast.py) to a full-bin reference npz (shotcost_branches.py): by idx
import sys, numpy as np
ref, new = np.load(sys.argv[1]), np.load(sys.argv[2])
pos = {int(i): n for n, i in enumerate(ref['idx'])}; sel = [pos[int(i)] for i in new['idx']]
def two(x): return np.stack([x[..., 0], x[..., 1:].sum(-1)], -1)
rows = []
for key in ('P', 'U', 'G', 'Q'):
    r = two(ref[key][sel]); n = new[key][..., :2]
    err = np.abs(r - n); scale = np.abs(r).max()
    rows.append((key, float(err.max()), float(scale), float((err / np.maximum(np.abs(r), 1e-12)).max())))
    print(f'{key}: max abs err {err.max():.3e} (max |value| {scale:.3e}); max rel err {rows[-1][3]:.3e}')
print('pairs compared', len(sel), '| k equal', bool((ref['k'][sel] == new['k']).all()))
