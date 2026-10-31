"""Step-4b recall diagnosis: how much each word of a caption moves its embedding. For N coco5k test captions, each word vector is
swapped for a random other trained word (fixed seed) and the influence 1 - cos(e, e_swapped) recorded. Aggregates: by word
position, by the word's type (its type tensor, cores merged), the per-caption share of the most influential word, and the
fraction of words with influence < 0.01 ("ignored").   s4_influence.py <checkpoint.pt> <out.json> [N=2000]"""
import sys, json, random, collections
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ck_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
if out_path.exists(): raise FileExistsError(out_path)
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
ck = CheckpointManager.load_model(ck_path, image_model, text_model, DEV); text_model.to(DEV).eval()
comp = de.compile_text('coco5k')
text_model.extend_symbols([v for c in comp.columns if c.endswith('_symbols') for v in comp[c].tolist() if not isinstance(v, float)], seed=SEED)
de.precompute_paths(text_model, comp)
trained = text_model.trained_symbols
words = sorted(s for s in trained if s.startswith('w:'))
rng = random.Random(0)
caps = [c for e, s in zip(comp['captions_einsum'], comp['captions_symbols']) for c in zip(e, s)]
rng.shuffle(caps); caps = [c for c in caps if all(d['name'] in trained for d in c[1])][:N]
variants, meta = [], []
for ci, (eq, syms) in enumerate(caps):
    widx = [k for k, d in enumerate(syms) if d['name'].startswith('w:')]
    for pos, k in enumerate(widx):
        nxt = syms[k + 1]['name'] if k + 1 < len(syms) else '?'
        typ = nxt.split('_core')[0] if nxt.startswith('t:') else '?'
        new = [dict(d) for d in syms]; new[k]['name'] = rng.choice(words)
        variants.append((eq, new)); meta.append((ci, pos, len(widx), typ))
def embed(cs):
    with torch.no_grad():
        return torch.cat([text_model(cs[j:j + 512]).float() for j in range(0, len(cs), 512)])
base = embed(caps); var = embed(variants)
ci = torch.tensor([m[0] for m in meta], device=DEV)
infl = (1 - (base[ci] * var).sum(1)).cpu().numpy()
res = {'checkpoint': str(ck_path), 'n_captions': len(caps), 'n_words': len(meta),
       'mean_influence': float(infl.mean()), 'frac_ignored_lt_0.01': float((infl < 0.01).mean())}
by_pos = collections.defaultdict(list); by_rel = collections.defaultdict(list); by_typ = collections.defaultdict(list); per_cap = collections.defaultdict(list)
for (c, p, n, t), v in zip(meta, infl):
    by_pos[min(p, 12)].append(v); by_rel[min(n - 1 - p, 12)].append(v); by_typ[t].append(v); per_cap[c].append(v)
res['by_position_from_start'] = {k: float(np.mean(v)) for k, v in sorted(by_pos.items())}
res['by_position_from_end'] = {k: float(np.mean(v)) for k, v in sorted(by_rel.items())}
res['by_type_top'] = {t: {'n': len(v), 'mean': float(np.mean(v))} for t, v in sorted(by_typ.items(), key=lambda kv: -len(kv[1]))[:15]}
shares = [max(v) / max(sum(v), 1e-12) for v in per_cap.values()]
res['top_word_share_mean'] = float(np.mean(shares))
res['top_word_share_uniform_ref'] = float(np.mean([1 / len(v) for v in per_cap.values()]))
json.dump(res, open(out_path, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'by_type_top'}, indent=1))
for t, v in res['by_type_top'].items(): print('  ', t[:45], v)
