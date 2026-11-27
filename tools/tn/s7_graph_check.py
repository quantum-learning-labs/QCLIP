"""Step 7 checks of the image graph TN (modules/models/vision/graph_tn.py), GPU:
 1. engine output == per-graph opt_einsum reference on random parameters (scale_free off, fp64-free: cosine and rel. error),
    on random graphs and on every graph of treewidth >= 2 among them (cycles)
 2. relations off at any parameters: output == closed form g * prod_j (S^T o_j)
 3. gradients reach P, E, base, P_g, base_g, S, A, B, m_p
 4. contraction cost (opt_einsum flops, largest intermediate) of every graph at the run dims, by treewidth with the apex
  PYTHONPATH=. python tools/tn/s7_graph_check.py [D A]"""
import sys, random, collections, multiprocessing, os
import numpy as np, torch, opt_einsum as oe
from modules.models.vision.graph_tn import ImageGraphTN, compile_graph, elim_path

dev = 'cuda'
G = torch.load('data/psg/graphs.pt'); start = G['obj_start'].tolist(); tw = G['tw'].tolist()
rng = random.Random(0)
rows = sorted(set(rng.sample(range(len(tw)), 300)) | set(rng.sample([r for r, t in enumerate(tw) if t >= 2], 100)))

def leaves(m, syms, eng):
    out = []
    for d in syms:
        k, rest = d['name'].split(':', 1)
        if k in ('x', 'y'): out.append(m._data(k, torch.tensor([int(rest)], device=dev))[0])
        else:
            b, r = eng.sym_loc[d['name']]; out.append(eng._bank(b)[r])
    return out

ok = True
for relations in (True, False):
    torch.manual_seed(1)
    m = ImageGraphTN(512, graph_dim=8, apex_dim=16, relations=relations, scale_free=False, rows=rows).to(dev)
    eng = m.engine
    with torch.no_grad():   # random parameters everywhere (identity init would hide relation bugs)
        for p in eng.parameters():
            if p.numel(): p.normal_(0, 0.4)
        m.base.normal_(0, 0.5); m.base_g.normal_(0, 0.5)
    batch = [m.graphs[r] for r in rows]
    out = m.contract(batch)
    ref = torch.stack([oe.contract(e, *leaves(m, s, eng)) for e, s in batch])
    cos = (torch.nn.functional.normalize(out.double(), dim=1) * torch.nn.functional.normalize(ref.double(), dim=1)).sum(1)
    rel = ((out - ref).norm(dim=1) / ref.norm(dim=1)).max().item()
    cyc = [i for i, r in enumerate(rows) if tw[r] >= 2]
    print(f'[1] relations={relations}: {len(rows)} graphs ({len(cyc)} with tw>=2): min cos {cos.min().item():.6f}, max rel err {rel:.2e}')
    ok &= rel < 1e-3
    if not relations:
        b, r = eng.sym_loc['s:apex']; S = eng._bank(b)[r]
        cf = []
        for row in rows:
            o = m._data('x', torch.arange(start[row], start[row + 1], device=dev)); g = m._data('y', torch.tensor([row], device=dev))[0]
            cf.append(g * torch.prod(o @ S, dim=0) if len(o) else g)
        e2 = ((out - torch.stack(cf)).norm(dim=1) / torch.stack(cf).norm(dim=1)).max().item()
        print(f'[2] closed form g*prod(S^T o): max rel err {e2:.2e}'); ok &= e2 < 1e-3
    else:
        m.zero_grad(); m(torch.tensor([[float(r)] for r in rows[:64]], device=dev)).sum().backward()
        named = {'P': m.P.weight, 'E': m.E.weight, 'base': m.base, 'P_g': m.P_g.weight, 'base_g': m.base_g}
        for name in ('s:apex', 'a:rel.in', 'a:rel.out'):
            b, _ = eng.sym_loc[name]; named[name + ' (bank)'] = eng._bank(b)
        named['m: (bank)'] = eng._bank(eng.sym_loc[next(n for n in eng.symbols if n.startswith('m:'))][0])
        g = {k: (0.0 if v.grad is None else float(v.grad.norm())) for k, v in named.items()}
        print('[3] grad norms', {k: f'{v:.2e}' for k, v in g.items()}); ok &= all(v > 0 and np.isfinite(v) for v in g.values())

D, A = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (32, 128)
def cost(r):
    e, s = compile_graph(r, start[r + 1] - start[r], start[r], G['rel'][r], D, A)
    _, info = oe.contract_path(e, *[d['shape'] for d in s], optimize=elim_path(start[r + 1] - start[r], G['rel'][r]), shapes=True)
    return float(info.opt_cost), float(info.largest_intermediate)
with multiprocessing.get_context('fork').Pool(int(os.environ.get('TN_PATH_WORKERS', 8))) as pool:
    C = np.array(pool.map(cost, range(len(tw)), chunksize=256))
twa = np.array(G['tw_apex'].tolist())
print(f'[4] D={D} A={A}: flops per image p50 {np.median(C[:, 0]):.2e} p99 {np.quantile(C[:, 0], .99):.2e} max {C[:, 0].max():.2e} total {C[:, 0].sum():.2e}; largest intermediate max {C[:, 1].max():.2e}')
for t in sorted(set(twa)):
    s = twa == t; print(f'    tw_apex {t}: n {s.sum():6d}  flops p50 {np.median(C[s, 0]):.2e} max {C[s, 0].max():.2e}  share of total {C[s, 0].sum() / C[:, 0].sum():.1%}')
print('ALL OK' if ok else 'CHECK FAILED')
