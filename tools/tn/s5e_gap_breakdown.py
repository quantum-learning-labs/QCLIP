"""Step 5e D3: per-caption breakdown of retrieval on the 5k test split and on 5k TRAINING images (as s5e_train_gap.py).
For every caption: expected t2i R@1/5/10 of its image (tie-neutral, as MMEvaluator._calculate_recall) and keys from the
training compile (all train captions): lexical (min / median train frequency of its words, any unseen word), structural
(train frequency of its diagram shape = einsum string + non-word symbol names, word identity ignored; min train frequency
of its type tensors), size (#words). Writes per-caption arrays and bucket tables (t2i R@10 per bucket).
  PYTHONPATH=. python tools/tn/s5e_gap_breakdown.py <out.json> <ckpt.pt>"""
import sys, json, torch
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from modules.utils.factory import build_experiment, load_obj
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import MMEvaluator

def encode(ev, loader, mapper):
    imgs, txts, img_of, caps, gi = [], [], [], [], 0
    for batch in loader:
        img_batch, txt_batch = mapper(batch)
        imgs.append(ev._encode_img(img_batch)); pay = []
        for cs in txt_batch:
            if isinstance(cs[0], str) or isinstance(cs[0][0], dict): cs = [cs]
            for c in cs: img_of.append(gi); pay.append((c[0], c[1])); caps.append((c[0], c[1]))
            gi += 1
        txts.append(ev._encode_txt(pay))
    return torch.cat(imgs), torch.cat(txts), np.array(img_of), caps

def per_caption(ev, I, C, img_of):
    S = ev._readout(C @ I.conj().T)   # captions x images
    S = S.nan_to_num(nan=float('-inf'))
    io = torch.from_numpy(img_of).to(S.device)
    pos = S.gather(1, io.view(-1, 1)); neg = torch.ones_like(S, dtype=torch.bool); neg[torch.arange(len(S)), io] = False
    g = ((S > pos) & neg).sum(1).double(); t = ((S == pos) & neg).sum(1).double()
    return {f'R{k}': ((k - g) / (t + 1)).clamp(0, 1).cpu().numpy() for k in (1, 5, 10)}

def keys(caps, wfreq, sfreq, tfreq):
    out = {k: [] for k in ('w_min', 'w_med', 'oov', 'shape', 't_min', 'n_words')}
    for e, syms in caps:
        names = [s['name'] for s in syms]; words = [n for n in names if n.startswith('w:')]; types = [n for n in names if not n.startswith('w:')]
        wf = [wfreq.get(w, 0) for w in words] or [0]
        out['w_min'].append(min(wf)); out['w_med'].append(float(np.median(wf))); out['oov'].append(int(min(wf) == 0))
        out['shape'].append(sfreq.get((e, tuple(types)), 0)); out['t_min'].append(min([tfreq.get(x, 0) for x in types] or [0]))
        out['n_words'].append(len(words))
    return {k: np.array(v) for k, v in out.items()}

EDGES = {'w_min': [0, 1, 10, 100, 1000], 'shape': [0, 1, 10, 100, 1000], 't_min': [0, 1, 10, 100, 1000], 'n_words': [0, 9, 11, 13, 16]}
def buckets(K, R):
    tab = {}
    for k, ed in EDGES.items():
        b = np.digitize(K[k], ed[1:])   # bucket i: ed[i] <= x < ed[i+1] (last open)
        tab[k] = [{'lo': ed[i], 'hi': ed[i + 1] if i + 1 < len(ed) else None, 'n': int((b == i).sum()),
                   'R10': float(R['R10'][b == i].mean()) if (b == i).any() else None} for i in range(len(ed))]
    tab['oov'] = [{'oov': v, 'n': int((K['oov'] == v).sum()), 'R10': float(R['R10'][K['oov'] == v].mean()) if (K['oov'] == v).any() else None} for v in (0, 1)]
    return tab

ck = sys.argv[2]
config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
for kv in sys.argv[3:]:   # config overrides for evaluation, e.g. text.oov_init=neutral
    k, v = kv.split('=', 1); d = config
    for part in k.split('.')[:-1]: d = d[part]
    d[k.split('.')[-1]] = v
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV)
tr, va = de.compile_text('train'), de.compile_text('val')
de.text_init(text_model, pd.concat([tr, va], ignore_index=True))
CheckpointManager.load_model(ck, image_model, text_model, DEV)
text_model.to(DEV).eval(); image_model.to(DEV).eval()
wfreq, sfreq, tfreq = Counter(), Counter(), Counter()
for es, ss in zip(tr['captions_einsum'], tr['captions_symbols']):
    for e, syms in zip(es, ss):
        names = [s['name'] for s in syms]; types = tuple(n for n in names if not n.startswith('w:'))
        wfreq.update(n for n in names if n.startswith('w:')); tfreq.update(types); sfreq[(e, types)] += 1
ev = MMEvaluator(image_model, text_model, DEV); mapper = load_obj(config['dataset']['eval_mapper'])
res = {'ckpt': ck, 'train_vocab': len(wfreq), 'train_shapes': len(sfreq), 'train_types': len(tfreq)}
with torch.no_grad():
    te = de.compile_text('coco5k')
    text_model.extend_symbols([v for c in te.columns if c.endswith('_symbols') for v in te[c].tolist() if not isinstance(v, float)], seed=SEED)
    keep = [c for c in tr.columns if not c.startswith('captions_neg')]
    for name, df, split in (('train5k', tr[keep].head(5000).reset_index(drop=True), 'val'), ('test5k', te, 'coco5k')):
        I, C, img_of, caps = encode(ev, de.get_loader(df, split=split), mapper)
        R = per_caption(ev, I, C, img_of); K = keys(caps, wfreq, sfreq, tfreq)
        res[name] = {'n_cap': len(caps), 'n_img': len(I), **{f't2i_{k}': float(v.mean()) for k, v in R.items()}, 'buckets': buckets(K, R),
                     'per_caption': {**{k: v.tolist() for k, v in R.items()}, **{k: v.tolist() for k, v in K.items()}}}
        print(name, {k: round(v, 4) for k, v in res[name].items() if k.startswith('t2i')}, 'n_cap', len(caps), flush=True)
        for k in EDGES: print(' ', k, [(b['lo'], b['n'], None if b['R10'] is None else round(b['R10'], 3)) for b in res[name]['buckets'][k]], flush=True)
        print('  oov', [(b['oov'], b['n'], None if b['R10'] is None else round(b['R10'], 3)) for b in res[name]['buckets']['oov']], flush=True)
json.dump(res, open(sys.argv[1], 'x'))
