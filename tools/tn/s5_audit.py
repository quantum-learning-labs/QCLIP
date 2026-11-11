"""Benchmark artefact audit (step 5, user-approved 2026-11-11): per-item scores on every compositional test set, then accuracy split
by ties, OOV, structure (positive and negative with the same einsum = only word positions differ), training frequency of the
rarest type tensor / word in the item, and dataset tags (ARO relation, Winoground tag). Checks itself: the overall accuracy must
equal the benchmark's (same scoring code paths).
  PYTHONPATH=. python tools/tn/s5_audit.py <checkpoint.pt> <out.json>"""
import sys, json, re, collections
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import MMEvaluator

ck_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
if out_path.exists(): raise FileExistsError(out_path)
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
ck = CheckpointManager.load_model(ck_path, image_model, text_model, DEV)
text_model.to(DEV).eval(); image_model.to(DEV).eval()
ev = MMEvaluator(image_model, text_model, DEV)
trained = text_model.trained_symbols
base = lambda n: re.sub(r'_core\d+$', '', n)

# training frequency of every word and type tensor (counted once per caption occurrence)
tr = de.compile_text('train')
freq = collections.Counter()
for col in [c for c in tr.columns if c.endswith('_symbols')]:
    for v in tr[col]:
        if isinstance(v, float): continue
        caps = v if v and isinstance(v[0], list) else [v]
        for syms in caps: freq.update({base(d['name']) for d in syms})
del tr

def bins(x):
    return '0' if x == 0 else '1-10' if x <= 10 else '11-100' if x <= 100 else '101-1k' if x <= 1000 else '>1k'

def group_acc(keys, corr, top=30):
    g = collections.defaultdict(list)
    for k, c in zip(keys, corr): g[k].append(c)
    return {str(k): {'n': len(v), 'acc': float(np.mean(v))} for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:top]}

META = {'aro_rel': 'relation_name', 'winoground': 'tag'}
res = {'checkpoint': str(ck_path), 'epoch': ck.get('epoch')}
for split, info in config['dataset']['test'].items():
    task = info['task']
    if task == 'global_retrieval': continue
    df = de.compile_text(split)
    scols = [c for c in df.columns if c.endswith('_symbols')]
    rows = [[r[c] if not isinstance(r[c], float) else [] for c in scols] for _, r in df.iterrows()]
    oov = [any(d['name'] not in trained for s in r for d in s) for r in rows]
    rare_t = [min([freq[base(d['name'])] for s in r for d in s if not d['name'].startswith('w:')] or [10**9]) for r in rows]
    rare_w = [min([freq[base(d['name'])] for s in r for d in s if d['name'].startswith('w:')] or [10**9]) for r in rows]
    ecols = [c for c in df.columns if c.endswith('_einsum')]
    same = [len({str(r[c]) for c in ecols}) == 1 for _, r in df.iterrows()] if len(ecols) > 1 else [None] * len(df)
    text_model.extend_symbols([v for c in scols for v in df[c].tolist() if not isinstance(v, float)], seed=SEED)
    de.precompute_paths(text_model, df)
    loader = de.get_loader(df, split=split)
    out = {'n': len(df), 'task': task}
    with torch.no_grad():
        if task in ('evaluate_text_choice', 'evaluate_image_choice'):
            p, n, _, _ = ev._pair_sims(loader, 'text' if task == 'evaluate_text_choice' else 'image')
            p, n = p.double().numpy(), n.double().numpy()
            corr = (p > n).astype(float); tie = np.isclose(p, n, rtol=1e-6, atol=1e-9)
            scores = {'acc': corr}
        elif task == 'evaluate_sugarcrepe_pp':
            S = []
            for b in loader:
                i = ev._encode_img(b['image'])
                S.append(torch.stack([ev._score(i, ev._encode_txt(b[k])) for k in ('pos_caption1', 'pos_caption2', 'neg_caption')], 1).cpu())
            S = torch.cat(S).double().numpy()
            scores = {'acc': ((S[:, 0] > S[:, 2]) & (S[:, 1] > S[:, 2])).astype(float), 'acc_pos1': (S[:, 0] > S[:, 2]).astype(float),
                      'acc_pos2': (S[:, 1] > S[:, 2]).astype(float)}
            corr = scores['acc']; tie = np.isclose(S[:, 0], S[:, 2], rtol=1e-6, atol=1e-9) | np.isclose(S[:, 1], S[:, 2], rtol=1e-6, atol=1e-9)
        elif task == 'evaluate_winoground':
            S = []
            for b in loader:
                i0, i1 = ev._encode_img(b['image_0']), ev._encode_img(b['image_1'])
                c0, c1 = ev._encode_txt(b['caption_0']), ev._encode_txt(b['caption_1'])
                S.append(torch.stack([ev._score(i0, c0), ev._score(i0, c1), ev._score(i1, c0), ev._score(i1, c1)], 1).cpu())
            S = torch.cat(S).double().numpy()   # columns: i0c0, i0c1, i1c0, i1c1
            t = (S[:, 0] > S[:, 1]) & (S[:, 3] > S[:, 2]); im = (S[:, 0] > S[:, 2]) & (S[:, 3] > S[:, 1])
            scores = {'txt': t.astype(float), 'img': im.astype(float), 'grp': (t & im).astype(float)}
            corr = scores['txt']
            tie = np.isclose(S[:, 0], S[:, 1], rtol=1e-6, atol=1e-9) | np.isclose(S[:, 3], S[:, 2], rtol=1e-6, atol=1e-9)
            tie_img = np.isclose(S[:, 0], S[:, 2], rtol=1e-6, atol=1e-9) | np.isclose(S[:, 3], S[:, 1], rtol=1e-6, atol=1e-9)
            out['tie_frac_img'] = float(tie_img.mean())
        else:
            continue
    for k, v in scores.items(): out[k] = float(v.mean())
    out['tie_frac'] = float(tie.mean())
    oov = np.array(oov); same_a = np.array([bool(s) for s in same])
    for k, v in scores.items():
        out[f'{k}_no_oov'] = float(v[~oov].mean()) if (~oov).any() else None
        out[f'{k}_oov'] = float(v[oov].mean()) if oov.any() else None
        out[f'{k}_excl_ties'] = float(v[~tie].mean()) if (~tie).any() else None
    out['oov_item_frac'] = float(oov.mean())
    if same[0] is not None:
        out['same_structure_frac'] = float(same_a.mean())
        out['acc_same_structure'] = float(corr[same_a].mean()) if same_a.any() else None
        out['acc_diff_structure'] = float(corr[~same_a].mean()) if (~same_a).any() else None
    out['acc_by_rarest_type_freq'] = group_acc([bins(x) for x in rare_t], corr)
    out['acc_by_rarest_word_freq'] = group_acc([bins(x) for x in rare_w], corr)
    if split in META and META[split] in df.columns:
        keys = df[META[split]].tolist()
        out[f'by_{META[split]}'] = {k: dict(v, **({'img': float(np.mean([s for kk, s in zip(keys, scores['img']) if str(kk) == k]))} if 'img' in scores else {}))
                                    for k, v in group_acc(keys, corr, top=60).items()}
    out['per_item'] = {k: v.astype(int).tolist() for k, v in scores.items()}
    out['per_item_tie'] = tie.astype(int).tolist()
    res[split] = out
    print(split, {k: v for k, v in out.items() if not k.startswith(('per_item', 'acc_by', 'by_'))})
json.dump(res, open(out_path, 'x'), indent=1)
