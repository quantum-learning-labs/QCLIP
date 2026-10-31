"""Step-4b jump diagnosis: geometry of a run's snapshots (epoch_XXX.pt, checkpoint_every) on the coco5k test split, compiled once.
Per snapshot: i2t/t2i R@10, effective rank (participation ratio) of text embeddings and of row-normalised pre-LN vectors,
hubness (max k-occurrence in t2i top-10), mean cosine of captions of different images (collapse), and for the 3 most used root
types the stable rank (||M||_F^2 / sigma_max^2) of the output core unfolded along its 512 leg.
  s4_traj.py <run_dir> <out.json>"""
import sys, json, glob, random
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine
run, out_path = Path(sys.argv[1]), Path(sys.argv[2])
if out_path.exists(): raise FileExistsError(out_path)
snaps = sorted(glob.glob(str(run / 'epoch_*.pt')))
config, DEV, SEED = setup_exp(str(run / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
CheckpointManager.load_model(snaps[0], image_model, text_model, DEV); text_model.to(DEV).eval()
comp = de.compile_text('coco5k')
text_model.extend_symbols([v for c in comp.columns if c.endswith('_symbols') for v in comp[c].tolist() if not isinstance(v, float)], seed=SEED)
de.precompute_paths(text_model, comp)
caps, img_of = [], []
for i, (es, ss) in enumerate(zip(comp['captions_einsum'], comp['captions_symbols'])):
    for c in zip(es, ss): caps.append(c); img_of.append(i)
loader = de.get_loader(comp, split='coco5k')
img = torch.cat([image_model(b['image'].to(DEV)).float() for b in loader])
img_of_t = torch.tensor(img_of, device=DEV)
rng = random.Random(0); pairs = [(rng.randrange(len(caps)), rng.randrange(len(caps))) for _ in range(20000)]
pairs = torch.tensor([p for p in pairs if img_of[p[0]] != img_of[p[1]]], device=DEV)
def eff(x):
    x = x - x.mean(0); ev = torch.linalg.eigvalsh((x.T @ x).double().cpu()).clamp_min(0); return float(ev.sum() ** 2 / (ev ** 2).sum())
roots = ['t:n@out@n', 't:s@n@n@out@n', 't:n@out@s@n']
res = []
for sp in snaps:
    ck = torch.load(sp, map_location=DEV, weights_only=False)   # non-strict: the OOV rows added above keep their seeded init
    text_model.load_state_dict(ck['text'], strict=False); text_model.to(DEV).eval()
    pre = []; text_model.capture = pre
    with torch.no_grad():
        E = torch.cat([text_model(caps[j:j + 512]).float() for j in range(0, len(caps), 512)])
    text_model.capture = None; pre = torch.nn.functional.normalize(torch.cat(pre).float(), dim=1)
    S_ = E @ img.T
    t2i = (S_ > S_.gather(1, img_of_t[:, None])).sum(1)
    own = torch.zeros_like(S_.T, dtype=torch.bool); own[img_of_t, torch.arange(len(caps), device=DEV)] = True
    best = S_.T.masked_fill(~own, -1e9).max(1).values; i2t = (S_.T.masked_fill(own, -1e9) > best[:, None]).sum(1)
    kocc = torch.bincount(S_.topk(10, dim=1).indices.flatten(), minlength=len(img))
    row = {'epoch': ck.get('epoch'), 'train_loss': ck.get('train_loss'), 'i2t_R10': float((i2t < 10).float().mean()),
           't2i_R10': float((t2i < 10).float().mean()), 'eff_rank_text': eff(E), 'eff_rank_preLN': eff(pre),
           'max_k_occ': int(kocc.max()), 'cos_diff_images': float((E[pairs[:, 0]] * E[pairs[:, 1]]).sum(1).mean())}
    sd = {k[len('params.'):]: v for k, v in text_model.state_dict().items() if k.startswith('params.t:')}
    for r in roots:
        core = next((v for k, v in sd.items() if k.split('_core')[0] == r and 512 in v.shape), None)
        if core is not None:
            leg = list(core.shape).index(512); m = core.movedim(leg, 0).reshape(512, -1).double().cpu()
            s = torch.linalg.svdvals(m); row[f'stable_rank_{r}'] = float((s ** 2).sum() / s[0] ** 2)
    res.append(row); print(json.dumps(row))
json.dump(res, open(out_path, 'w'), indent=1)
