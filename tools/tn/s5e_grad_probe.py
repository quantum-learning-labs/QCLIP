"""Step 5e: where do text.affine's step-0 gradients go non-finite? One forward/backward (zero biases, as training starts) on
COCO train captions; counts levelwise einsum outputs whose max |entry| is 0 (scale_free then divides by its clamp) and the
non-finite / largest row gradients per bank, with the symbols of the worst rows.
  PYTHONPATH=. python tools/tn/s5e_grad_probe.py <config.yaml> [n_captions] [--set k=v ...]"""
import sys, copy, torch
import pandas as pd
import modules.models.text.einsum_classical as ec
from modules.utils.general import setup_exp
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

args = sys.argv[2:]; n = int(args.pop(0)) if args and not args[0].startswith('--') else 512
ov = [args[i + 1] for i in range(len(args)) if args[i] == '--set']
cfg, DEV, _ = setup_exp(sys.argv[1], seed=0, overrides=ov)
df = pd.read_pickle(cfg['dataset']['train']['text_path']).head(n // 5 + 1)
zeros = {'ops': 0, 'zero': 0, 'tiny': 0}

class _T:   # torch proxy for einsum_classical: count zero-max / tiny-max einsum results
    def __getattr__(self, k): return getattr(torch, k)
    @staticmethod
    def einsum(eq, ops):
        r = torch.einsum(eq, ops)
        m = r.detach().abs().flatten(1).amax(1)
        zeros['ops'] += 1; zeros['zero'] += int((m == 0).sum()); zeros['tiny'] += int(((m > 0) & (m < 1e-20)).sum())
        return r
ec.torch = _T()

torch.manual_seed(0)
ansatz, _, tm, _ = build_experiment(cfg, DEV)
comp = ansatz.compile_dataset(df.copy())
de = DataEngine(cfg, ansatz, DEV); de.text_init(tm, comp); tm.to(DEV).train()
if cfg['text'].get('affine'): tm.affine_reset_bias()
caps = [(e, s) for es, ss in zip(comp['captions_einsum'], comp['captions_symbols']) for e, s in zip(es, ss)][:n]
out = tm(caps)
g = torch.Generator(device='cpu').manual_seed(1)
(out * torch.randn(out.shape, generator=g).to(out.device)).sum().backward()
print('captions', len(caps), 'out finite', bool(torch.isfinite(out).all()), '| einsum results', zeros)
names = {loc: s for s, loc in tm.sym_loc.items()}
for b, rows, leaf in tm._active:
    gr = leaf.grad
    if gr is None: continue
    bad = ~torch.isfinite(gr).flatten(1).all(1); mx = gr.nan_to_num(0, 0, 0).abs().flatten(1).amax(1)
    worst = [names.get((b, int(r)), (b, int(r))) for r in rows[bad.nonzero().flatten()[:4].cpu()]] if bad.any() else \
            [names.get((b, int(rows[i])), None) for i in mx.argsort(descending=True)[:2].cpu()]
    print(f'bank {b!s:>14} shape {tuple(leaf.shape[1:])!s:>16} rows {leaf.shape[0]:5d} nonfinite rows {int(bad.sum()):5d} '
          f'max|g| {mx.max().item():.3e} worst {worst}')
