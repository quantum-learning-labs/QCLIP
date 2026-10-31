"""Stable rank (||M||_F^2/sigma_max^2) of the 3 most used root types' output cores (unfolded along the 512 leg) in checkpoints.
  s4_rootsr.py <ckpt.pt>..."""
import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location='cpu', weights_only=False); sd = ck['text']; row = []
    for r in ['t:n@out@n', 't:s@n@n@out@n', 't:n@out@s@n']:
        core = next((v for k, v in sd.items() if k.startswith('params.') and k[7:].split('_core')[0] == r and 512 in v.shape), None)
        if core is None: row.append(None); continue
        m = core.movedim(list(core.shape).index(512), 0).reshape(512, -1).double(); s = torch.linalg.svdvals(m)
        row.append(round(float((s ** 2).sum() / s[0] ** 2), 1))
    print(p.split('/')[-2], p.split('/')[-1], 'epoch', ck.get('epoch'), 'loss', ck.get('train_loss'), 'stable ranks', row)
