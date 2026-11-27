"""Step 7: memory and time of the image graph TN per training batch (forward + backward), GPU.
  PYTHONPATH=. python tools/tn/s7_probe.py D A max_tw|none [relations 1|0]"""
import sys, time, random, torch
from modules.models.vision.graph_tn import ImageGraphTN
D, A = int(sys.argv[1]), int(sys.argv[2]); mt = None if sys.argv[3] == 'none' else int(sys.argv[3])
rel = len(sys.argv) < 5 or sys.argv[4] == '1'
t0 = time.time(); torch.manual_seed(0)
m = ImageGraphTN(512, graph_dim=D, apex_dim=A, relations=rel, max_tw=mt).cuda()
print(f'build {time.time() - t0:.0f} s | params {sum(p.numel() for p in m.parameters()):,}', flush=True)
n = len(m.graphs); rng = random.Random(0)
worst = sorted(range(n), key=lambda r: -m.flops[__import__("modules.models.text.einsum_classical", fromlist=["_batched"])._batched(m.graphs[r][0])])[:64]
batches = [rng.sample(range(n), 512) for _ in range(8)] + [worst + rng.sample(range(n), 448)]
for k, rows in enumerate(batches):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    out = m(torch.tensor([[float(r)] for r in rows], device='cuda'))
    (out * torch.randn_like(out)).sum().backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"batch {k}{' (64 costliest graphs)' if k == 8 else ''}: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, finite {bool(torch.isfinite(out).all())}", flush=True)
