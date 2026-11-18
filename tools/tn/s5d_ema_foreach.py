"""Step 5d: EMA update as torch._foreach_lerp_ vs the per-tensor lerp_ loop: bit-identical results (torch.equal) on many small
CUDA tensors over 50 updates, and the time per update at the MLP control's tensor count.
  PYTHONPATH=. python tools/tn/s5d_ema_foreach.py"""
import time, torch
torch.manual_seed(0); dev = 'cuda'
shapes = [(128, 64), (128,), (128,), (128,), (64, 128), (64,)] * 4000 + [(64,)] * 20000   # ~44k tensors
P = [torch.randn(s, device=dev) for s in shapes]
A = [p.clone() * 0.5 for p in P]; B = [a.clone() for a in A]
for t in range(1, 51):
    d = min(0.999, (1 + t) / (10 + t))
    for p in P: p.add_(torch.randn_like(p) * 1e-2)
    for s, p in zip(A, P): s.lerp_(p, 1 - d)
    torch._foreach_lerp_(B, P, 1 - d)
print('bit-identical:', all(torch.equal(a, b) for a, b in zip(A, B)), 'tensors', len(P))
for name, f in (('loop', lambda: [s.lerp_(p, 0.001) for s, p in zip(A, P)]), ('foreach', lambda: torch._foreach_lerp_(B, P, 0.001))):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): f()
    torch.cuda.synchronize(); print(f'{name}: {(time.time() - t0) / 5 * 1000:.1f} ms per EMA update')
