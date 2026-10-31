"""Step-4b recall diagnosis on the coco5k test split for one checkpoint (A3a, A4): text/image embedding geometry and where
retrieval fails. Writes <out>.json.
  s4_embed.py <checkpoint.pt> <out.json>
Reports: i2t/t2i R@K (plain ranks, random tie order ignored); effective rank (participation ratio of the covariance spectrum)
of text embeddings (after the head), of pre-LayerNorm text vectors (row-normalised) and of the image embeddings; per root type
(the type tensor carrying the 512-d 'out' leg): #captions, numerical rank of its pre-LN vectors, t2i R@10; t2i R@10 by #words
and by OOV; hubness of the t2i top-10 (k-occurrence of images)."""
import sys, json, collections
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ck_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
if out_path.exists(): raise FileExistsError(out_path)
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV)
de.image_init(image_model)
ck = CheckpointManager.load_model(ck_path, image_model, text_model, DEV)
text_model.to(DEV); text_model.eval(); image_model.eval()
comp = de.compile_text('coco5k')
oov = text_model.extend_symbols([v for c in comp.columns if c.endswith('_symbols') for v in comp[c].tolist() if not isinstance(v, float)], seed=SEED)
de.precompute_paths(text_model, comp)
trained = text_model.trained_symbols
loader = de.get_loader(comp, split='coco5k')

imgs, caps, cap_img, meta = [], [], [], []
for batch in loader:
    for k, (img, cl) in enumerate(zip(batch['image'], batch['caption'])):
        i = len(imgs); imgs.append(img)
        for c in cl:
            caps.append(c); cap_img.append(i)
            syms = [d['name'] for d in c[1]]
            root = next((s.split('_core')[0] for s in syms if 'out' in s.split(':')[-1].split('_core')[0].split('@')), '?')
            meta.append({'n_words': sum(s.startswith('w:') for s in syms), 'n_tensors': len(syms), 'root': root,
                         'oov': any(s not in trained for s in syms)})
with torch.no_grad():
    img = image_model(torch.stack(imgs).to(DEV)).float()
    txt, pre = [], []
    text_model.capture = pre
    for j in range(0, len(caps), 512):
        txt.append(text_model(caps[j:j + 512]).float())
    text_model.capture = None
    txt, pre = torch.cat(txt), torch.cat(pre).float()
cap_img = torch.tensor(cap_img, device=DEV)
S_ = txt @ img.T                                   # (captions, images)
t2i_rank = (S_ > S_.gather(1, cap_img[:, None])).sum(1)          # images scored above the caption's own image
pos = S_.T                                         # (images, captions)
own = torch.zeros_like(pos, dtype=torch.bool); own[cap_img, torch.arange(len(caps), device=DEV)] = True
best_own = pos.masked_fill(~own, -1e9).max(1).values
i2t_rank = (pos.masked_fill(own, -1e9) > best_own[:, None]).sum(1)
R = lambda r, k: float((r < k).float().mean())

def eff_rank(x):   # participation ratio of the covariance spectrum (CPU eigvalsh: CUDA eigh is broken here)
    x = x - x.mean(0); ev = torch.linalg.eigvalsh((x.T @ x).double().cpu()).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())
def num_rank(x, tol=1e-3):
    s = torch.linalg.svdvals(x.double().cpu()); return int((s > tol * s[0]).sum()) if len(s) else 0

res = {'checkpoint': str(ck_path), 'epoch': ck.get('epoch'), 'n_images': len(imgs), 'n_captions': len(caps), 'oov': oov,
       'i2t': {f'R{k}': R(i2t_rank, k) for k in (1, 5, 10)}, 't2i': {f'R{k}': R(t2i_rank, k) for k in (1, 5, 10)},
       'eff_rank': {'text': eff_rank(txt), 'text_preLN_rownorm': eff_rank(torch.nn.functional.normalize(pre, dim=1)),
                    'image': eff_rank(img)}}
hit10 = (t2i_rank < 10).cpu().numpy()
by_root = collections.defaultdict(list)
for j, m in enumerate(meta): by_root[m['root']].append(j)
roots = sorted(by_root.items(), key=lambda kv: -len(kv[1]))
res['roots'] = [{'root': r, 'n': len(ix), 'frac': len(ix) / len(caps), 'rank_preLN': num_rank(pre[ix]), 't2i_R10': float(hit10[ix].mean())}
                for r, ix in roots[:25]]
res['n_root_types'] = len(roots)
nw = np.array([m['n_words'] for m in meta]); bins = [(0, 8), (9, 10), (11, 12), (13, 99)]
res['t2i_R10_by_words'] = {f'{a}-{b}': {'n': int(((nw >= a) & (nw <= b)).sum()), 'R10': float(hit10[(nw >= a) & (nw <= b)].mean())} for a, b in bins}
ov = np.array([m['oov'] for m in meta])
res['t2i_R10_by_oov'] = {'in_vocab': float(hit10[~ov].mean()), 'oov': float(hit10[ov].mean()), 'oov_frac': float(ov.mean())}
top10 = S_.topk(10, dim=1).indices.flatten()
kocc = torch.bincount(top10, minlength=len(imgs)).float()
res['hubness'] = {'max_k_occ': float(kocc.max()), 'frac_images_never_in_top10': float((kocc == 0).float().mean()),
                  'top1pct_images_share_of_top10': float(kocc.sort(descending=True).values[:len(imgs) // 100].sum() / kocc.sum()),
                  'skew': float(((kocc - kocc.mean()) ** 3).mean() / kocc.std() ** 3)}
json.dump(res, open(out_path, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'roots'}, indent=1))
print('roots (top 10):'); [print('  ', r) for r in res['roots'][:10]]
