"""Compare the optimiser state (save_state) of two last.pt files: RowAdam row_step equality, exp_avg_sq ratios, weights.
  s4_state_cmp.py a/last.pt b/last.pt"""
import sys, torch
a, b = (torch.load(p, map_location='cpu', weights_only=False) for p in sys.argv[1:3])
print('epochs', a['epoch'], b['epoch'])
sa, sb = a['state']['optimizer']['state'], b['state']['optimizer']['state']
for k in sorted(sa):
    x, y = sa[k], sb.get(k, {})
    if 'row_step' in x:
        same = torch.equal(x['row_step'], y['row_step'])
        r = (y['exp_avg_sq'].sum() / x['exp_avg_sq'].sum()).item()
        d = ((y['exp_avg'] - x['exp_avg']).norm() / x['exp_avg'].norm()).item()
        print(f"bank state {k}: row_step equal {same} (max {x['row_step'].max().item():.0f}), sum v ratio {r:.4f}, rel diff m {d:.3f}")
    elif 'step' in x:
        print(f"dense state {k}: step {x['step'].item()} vs {y['step'].item()}")
ta, tb = a['text'], b['text']
rel = max(((tb[k] - ta[k]).norm() / ta[k].norm().clamp_min(1e-30)).item() for k in ta if k.startswith('params.') and ta[k].norm() > 0)
print('max rel weight diff over tensors', f'{rel:.3g}')
print('sched', a['state']['scheduler'] == b['state']['scheduler'] if not isinstance(a['state']['scheduler'], dict) else
      {k: (a['state']['scheduler'][k] == b['state']['scheduler'][k]) for k in ('epoch',)})
