"""Step 5d: MLPModel.forward_batched vs forward (the per-node loop) on real COCO captions: sentence vectors and parameter
gradients agree up to float rounding; timing of forward + backward per 512-caption batch. Test captions (validation.pkl)
include words unseen in training (zero leaves) and diagrams with loops (dead wires read as zeros in both).
  PYTHONPATH=. python tools/tn/s5d_mlp_eq.py"""
import time, random, torch
import pandas as pd
from modules.compilation.classical.neural import MLPCompiler
from modules.models.text.neural_model import MLPModel
torch.manual_seed(0); random.seed(0)
dev = 'cuda'
comp = MLPCompiler()
tr = comp.compile_dataset(pd.read_pickle('data/mscoco/processed/tn/train.pkl').head(20000).copy())
te = comp.compile_dataset(pd.read_pickle('data/mscoco/processed/validation.pkl').head(2000).copy())
m = MLPModel(hdim=64, out_dim=512)
m.from_plans(tr['captions_einsum'].tolist()); m.to(dev)
caps = lambda df: [(p, s) for ps, ss in zip(df['captions_einsum'], df['captions_symbols']) for p, s in zip(ps, ss)]
pools = {'train': caps(tr), 'test(OOV)': caps(te)}
def run(batch, batched, R):
    m.batched = batched; m.zero_grad(set_to_none=True)
    out = m(batch); (out * R).sum().backward()
    return out.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
worst = 0.0
for name, pool in pools.items():
    for t in range(3):
        batch = random.sample(pool, 512); R = torch.randn(512, 512, device=dev)
        o1, g1 = run(batch, False, R); o2, g2 = run(batch, True, R)
        # nodes that never reach the root (dead branches of loop diagrams) get grad None in the loop, zeros when batched
        assert g1.keys() == g2.keys(), f'gradient sets differ: {sorted(g1.keys() ^ g2.keys())[:5]}'
        do = (o1 - o2).abs().max().item()
        dg = max(((g1[k] - g2[k]).abs().max() / (g1[k].abs().max() + 1e-12)).item() for k in g1.keys() & g2.keys() if g1[k].abs().max() > 0)
        print(f'  params with grad: loop {len(g1)}, batched {len(g2)}, all-zero extras {len(g1.keys() ^ g2.keys())}')
        worst = max(worst, do); print(f'{name} batch {t}: max |dout| {do:.2e}, max rel |dgrad| {dg:.2e}, params with grad {len(g1)}', flush=True)
for batched in (False, True):
    batch = random.sample(pools['train'], 512); R = torch.randn(512, 512, device=dev); run(batch, batched, R)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): run(random.sample(pools['train'], 512), batched, R)
    torch.cuda.synchronize(); print(f'batched={batched}: {(time.time() - t0) / 5:.3f} s per forward+backward (512 captions)', flush=True)
print('PASS' if worst < 1e-4 else 'FAIL', worst)
