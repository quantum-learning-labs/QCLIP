"""Unit checks for modules/models/text/tn_constraints.py on a mock TNModel (random banks, MPS cores, skewed spectra).
Verification by CPU SVD (CUDA svd is broken in our envs). Usage: s4_constraints_test.py [cuda|cpu]"""
import sys
from collections import Counter
import torch
from modules.models.text.tn_constraints import TNConstraints, polar, unfold

dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
torch.manual_seed(0)


class Mock(torch.nn.Module):
    # banks: words (128,), dense type (128,128), MPS of 3 cores (128,8),(8,128,8),(8,128); spiky spectra on some rows
    def __init__(self, n_sym=40):
        super().__init__()
        self.bank_shapes = [(128,), (128, 128), (128, 8), (8, 128, 8), (8, 128)]
        rows = [n_sym] * 5
        self.sym_loc, self.out_votes = {}, {}
        for b, (s, n) in enumerate(zip(self.bank_shapes, rows)):
            t = torch.randn(n, *s) / s[0] ** 0.5
            if b == 1:   # rank-1 spikes
                t[:10] += 3 * torch.randn(10, 128, 1) * torch.randn(10, 1, 128) / 128 ** 0.5
            self.register_parameter(f'bank{b}', torch.nn.Parameter(t))
        for i in range(n_sym):
            self.sym_loc[f'w:{i}'] = (0, i)
            self.sym_loc[f't:d{i}'] = (1, i); self.out_votes[f't:d{i}'] = Counter({i % 2: 1})
            for k in range(3):
                self.sym_loc[f't:m{i}_core{k}'] = (2 + k, i); self.out_votes[f't:m{i}_core{k}'] = Counter({(i + k) % len(self.bank_shapes[2 + k]): 1})

    def _bank(self, b):
        return getattr(self, f'bank{b}')

    def touch(self, frac=0.5):
        # perturb a random subset of rows (as an optimiser step would) and record them as touched
        self._last_row_grads = []
        with torch.no_grad():
            for b in range(5):
                bank = self._bank(b)
                rows = torch.randperm(bank.shape[0], device=bank.device)[: int(frac * bank.shape[0])].sort().values
                bank.index_add_(0, rows, 0.3 * torch.randn(len(rows), *bank.shape[1:], device=bank.device) / bank.shape[1] ** 0.5)
                self._last_row_grads.append((bank, rows, None))


def mps_full(m, i):
    c0, c1, c2 = (m._bank(2 + k)[i].double().cpu() for k in range(3))
    return torch.einsum('ab,bcd,de->ace', c0, c1, c2)


def sv(x):
    return torch.linalg.svdvals(x.double().cpu())


ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")

# polar
for shape in [(64, 128, 8), (64, 8, 1024), (64, 128, 128)]:
    m = torch.randn(*shape, device=dev)
    q = polar(m, 60)   # random square Gaussians are ill-conditioned: the init setting (per-step use starts near-isometric)
    s = sv(q)
    check(f'polar {shape}', (s - 1).abs().max() < 1e-4, f'max |sigma-1| {(s - 1).abs().max():.2e}')

# canonical gauge keeps the represented tensors, cores 0,1 isometric
m = Mock().to(dev)
full0 = [mps_full(m, i) for i in range(40)]
c = TNConstraints(m, {'mps_gauge': 'canonical', 'norm': 'type'})
err = max(((mps_full(m, i) - full0[i]).norm() / full0[i].norm()).item() for i in range(40))
check('canonical keeps tensors', err < 1e-4, f'max rel err {err:.2e}')
iso = max((sv(m._bank(2)[i]) - 1).abs().max().item() for i in range(40))
iso1 = max((sv(m._bank(3)[i].reshape(-1, 8)) - 1).abs().max().item() for i in range(40))
check('canonical cores isometric', max(iso, iso1) < 1e-3, f'core0 {iso:.1e} core1 {iso1:.1e}')
for _ in range(5):
    full_b = [mps_full(m, i) for i in range(40)]
    m.touch(); c.apply()
iso = max((sv(m._bank(3)[i].reshape(-1, 8)) - 1).abs().max().item() for i in range(40))
check('canonical after steps', iso < 1e-3, f'core1 max |sigma-1| {iso:.1e}')
rel = (torch.stack([mps_full(m, i).norm() for i in range(40)]) / torch.stack([f.norm() for f in full0])).sub(1).abs().max()
check('canonical+norm: full norm fixed', rel < 1e-3, f'max |norm/norm0 - 1| {rel:.1e}')

# norm
m = Mock().to(dev)
c = TNConstraints(m, {'norm': 'all'})
for _ in range(5): m.touch(); c.apply()
err = max((m._bank(b).flatten(1).norm(dim=1) / c.norm0[b] - 1).abs().max().item() for b in range(5))
check('norm all fixed', err < 1e-5, f'max rel {err:.1e}')

# spectral cap
m = Mock().to(dev)
def ratio(m):
    r = []
    for i in range(40):
        u = unfold(m._bank(1)[i:i + 1], i % 2)[0]
        s = sv(u); r.append((s[0] / s.pow(2).mean().sqrt()).item())
    return max(r)
r0 = ratio(m)
c = TNConstraints(m, {'spectral_kappa': 2.0, 'norm': 'type'})
for _ in range(20):
    m.touch(1.0); c.apply()
check('spectral cap kappa 2', ratio(m) < 2.1, f'max sigma_max/sigma_rms {r0:.2f} -> {ratio(m):.2f}')

# isometric + init_orth
m = Mock().to(dev)
n0 = m._bank(1).flatten(1).norm(dim=1).clone()
c = TNConstraints(m, {'isometric': True, 'init_orth': True})
flat = max((lambda s: (s / s.mean() - 1).abs().max().item())(sv(unfold(m._bank(1)[i:i + 1], i % 2)[0])) for i in range(40))
check('init_orth flat spectrum', flat < 1e-3, f'max |sigma/mean-1| {flat:.1e}; norm kept {(m._bank(1).flatten(1).norm(dim=1) / n0 - 1).abs().max():.1e}')
for _ in range(5): m.touch(); c.apply()
flat = max((lambda s: (s / s.mean() - 1).abs().max().item())(sv(unfold(m._bank(1)[i:i + 1], i % 2)[0])) for i in range(40))
check('isometric after steps', flat < 1e-3, f'max |sigma/mean-1| {flat:.1e}')
print('ALL PASS' if ok else 'SOME FAILED')
