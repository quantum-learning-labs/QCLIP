"""Debug (step 3): one batch through the TN text model, pre-LayerNorm output scale and leaf gradient norms per bank.
  PYTHONPATH=. python tools/tn/dbg_grad.py <config> [--set k=v ...]   (CPU, 2000 train rows, batch 512)"""
import sys, yaml, torch, collections
from modules.utils.factory import build_experiment
from modules.utils.general import load_pkl
cfg = yaml.safe_load(open(sys.argv[1]))
for kv in sys.argv[2:]:
    if kv == '--set': continue
    k, v = kv.split('='); d = cfg
    for p in k.split('.')[:-1]: d = d[p]
    d[k.split('.')[-1]] = yaml.safe_load(v)
torch.manual_seed(0)
ansatz, image_model, text_model, loss_fn = build_experiment(cfg, 'cpu')
df = ansatz.compile_dataset(load_pkl(cfg['dataset']['train']['text_path']).iloc[:2000].copy())
text_model.from_symbols(df['captions_symbols'])
sents = [(e, s) for es, ss in zip(df['captions_einsum'], df['captions_symbols']) for e, s in zip(es, ss)][:512]
pre = {}
text_model.norm_head.register_forward_hook(lambda m, i, o: pre.setdefault('x', i[0].detach()))
torch.set_grad_enabled(True)
out = text_model(sents)
print('grad enabled', torch.is_grad_enabled(), 'out requires grad', out.requires_grad, 'active leaves', len(text_model._active))
x = pre['x']
nrm = x.norm(dim=-1)
print('pre-LN output norm: median %.3e, min %.3e, max %.3e, frac<1e-3 %.3f' % (nrm.median(), nrm.min(), nrm.max(), (nrm < 1e-3).float().mean()))
img = torch.nn.functional.normalize(torch.randn(len(sents), out.shape[1]), dim=-1)
loss = loss_fn(img, out) if not isinstance(loss_fn(img, out), tuple) else loss_fn(img, out)[0]
loss.backward()
g = collections.defaultdict(float); n = collections.Counter()
for b, rows, leaf in text_model._active:
    g[tuple(text_model.bank_shapes[b])] += float(leaf.grad.pow(2).sum()) if leaf.grad is not None else float('nan'); n[tuple(text_model.bank_shapes[b])] += len(rows)
print('loss', float(loss), '| LN weight grad', float(text_model.norm_head.weight.grad.norm()))
for s in sorted(g, key=lambda s: (len(s), s)): print('bank', s, 'rows', n[s], 'grad norm %.3e' % g[s] ** 0.5)
