"""Step-4b recall diagnosis: agreement between captions of the SAME image (paraphrases) on the coco5k test split. Mean cosine of
text embeddings for same-image caption pairs, split by same vs different root type (the type tensor carrying the 'out' leg),
vs pairs from different images; and how many of an image's other captions are nearer to a caption than its own image is.
  s4_paraphrase.py <checkpoint.pt> <out.json>"""
import sys, json, itertools, random
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine
ck_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
if out_path.exists(): raise FileExistsError(out_path)
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
CheckpointManager.load_model(ck_path, image_model, text_model, DEV); text_model.to(DEV).eval()
comp = de.compile_text('coco5k')
text_model.extend_symbols([v for c in comp.columns if c.endswith('_symbols') for v in comp[c].tolist() if not isinstance(v, float)], seed=SEED)
de.precompute_paths(text_model, comp)
caps, img_of, root = [], [], []
for i, (es, ss) in enumerate(zip(comp['captions_einsum'], comp['captions_symbols'])):
    for c in zip(es, ss):
        caps.append(c); img_of.append(i)
        names = [d['name'].split('_core')[0] for d in c[1]]
        root.append(next((n for n in names if 'out' in n.split(':')[-1].split('@')), '?'))
with torch.no_grad():
    E = torch.cat([text_model(caps[j:j + 512]).float() for j in range(0, len(caps), 512)])
img_of = np.array(img_of); root = np.array(root)
same_r, diff_r = [], []
by_img = {}
for j, i in enumerate(img_of): by_img.setdefault(i, []).append(j)
for i, js in by_img.items():
    for a, b in itertools.combinations(js, 2):
        (same_r if root[a] == root[b] else diff_r).append(float(E[a] @ E[b]))
rng = random.Random(0); other = []
for _ in range(50000):
    a, b = rng.randrange(len(caps)), rng.randrange(len(caps))
    if img_of[a] != img_of[b]: other.append(float(E[a] @ E[b]))
res = {'checkpoint': str(ck_path), 'same_image_same_root': {'n': len(same_r), 'mean_cos': float(np.mean(same_r))},
       'same_image_diff_root': {'n': len(diff_r), 'mean_cos': float(np.mean(diff_r))},
       'different_images': {'n': len(other), 'mean_cos': float(np.mean(other))},
       'frac_pairs_same_root': len(same_r) / max(len(same_r) + len(diff_r), 1)}
json.dump(res, open(out_path, 'w'), indent=1); print(json.dumps(res, indent=1))
