"""Step 7: learned predicate matrices R_p = A diag(m_p) B of an image graph TN checkpoint: distance from identity, spread across
predicates, and how strongly each R_p couples different object indices (off-diagonal mass).
  PYTHONPATH=. python tools/tn/s7_relstats.py <ckpt.pt>"""
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = sd.get('image_model', sd.get('image_state_dict', sd)) if isinstance(sd, dict) else sd
keys = [k for k in sd if 'params.' in k] if isinstance(sd, dict) else []
if not keys:   # find the image model state dict inside the checkpoint
    for k, v in sd.items():
        if isinstance(v, dict) and any('params.a:rel.in' in kk for kk in v): sd = v; break
P = {k.split('params.')[1]: v.float() for k, v in sd.items() if 'params.' in k}
A, B = P['a:rel.in'], P['a:rel.out']; D = A.shape[0]; I = torch.eye(D)
ms = {k: v for k, v in P.items() if k.startswith('m:')}
R = {k: A @ torch.diag(v) @ B for k, v in ms.items()}
Rm = torch.stack(list(R.values())); mean = Rm.mean(0)
rel = lambda X, Y: float((X - Y).norm() / Y.norm())
print(f'|A-I|/|I| {rel(A, I):.3f}  |B-I|/|I| {rel(B, I):.3f}  m: mean {torch.stack(list(ms.values())).mean():.3f} std over all {torch.stack(list(ms.values())).std():.3f}')
print(f'R_p: mean |R_p - mean R|/|mean R| {sum(rel(r, mean) for r in R.values()) / len(R):.3f}; |mean R - c I| rel {rel(mean, I * torch.trace(mean) / D):.3f}')
off = [float((r - torch.diag(torch.diag(r))).norm() / r.norm()) for r in R.values()]
print(f'off-diagonal share of |R_p|: mean {sum(off) / len(off):.3f} min {min(off):.3f} max {max(off):.3f}')
S = [v for k, v in P.items() if k == 's:apex'][0]; print('S apex norm', float(S.norm()))
for k in ('E.weight', 'P.weight', 'base', 'P_g.weight', 'base_g'):
    for kk, v in sd.items():
        if kk.endswith(k): print(f'{kk} norm {float(v.float().norm()):.3f}')
