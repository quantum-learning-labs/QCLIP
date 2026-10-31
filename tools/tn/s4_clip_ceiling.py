"""Step-4b A1: CLIP ViT-B/32's own text encoder on the coco5k test split (data/mscoco/processed/validation.pkl), scored
against the same stored image embeddings as the TN (unit-normalised, cosine): the ceiling of a good text encoder in this frozen
image space. Plain ranks.   s4_clip_ceiling.py <out.json>"""
import sys, json
from pathlib import Path
import torch, pandas as pd, clip
out = Path(sys.argv[1])
if out.exists(): raise FileExistsError(out)
dev = 'cuda'
df = pd.read_pickle('data/mscoco/processed/validation.pkl')
emb = torch.load('data/mscoco/processed/image_embeddings.pt', weights_only=False)
print(type(emb), len(df), list(df.columns))
img = torch.stack([emb[int(Path(str(f)).stem)] for f in df['file_name']]).float().to(dev)
caps, cap_img = [], []
for i, cl in enumerate(df['captions']):
    for c in cl: caps.append(str(c)); cap_img.append(i)
model, _ = clip.load('ViT-B/32', device=dev); model.eval()
with torch.no_grad():
    txt = torch.cat([model.encode_text(clip.tokenize(caps[j:j + 1024], truncate=True).to(dev)).float() for j in range(0, len(caps), 1024)])
img = torch.nn.functional.normalize(img, dim=1); txt = torch.nn.functional.normalize(txt, dim=1)
cap_img = torch.tensor(cap_img, device=dev)
S_ = txt @ img.T
t2i = (S_ > S_.gather(1, cap_img[:, None])).sum(1)
own = torch.zeros_like(S_.T, dtype=torch.bool); own[cap_img, torch.arange(len(caps), device=dev)] = True
best = S_.T.masked_fill(~own, -1e9).max(1).values
i2t = (S_.T.masked_fill(own, -1e9) > best[:, None]).sum(1)
x = txt - txt.mean(0); ev = torch.linalg.eigvalsh((x.T @ x).double().cpu()).clamp_min(0)
res = {'n_images': len(df), 'n_captions': len(caps), 'i2t': {f'R{k}': float((i2t < k).float().mean()) for k in (1, 5, 10)},
       't2i': {f'R{k}': float((t2i < k).float().mean()) for k in (1, 5, 10)}, 'eff_rank_text': float(ev.sum() ** 2 / (ev ** 2).sum())}
json.dump(res, open(out, 'w'), indent=1); print(res)
