"""Word influence by part of speech (step 5, user-approved 2026-11-11). As s4_influence.py (each word vector swapped for a random
other trained word, influence = 1 - cos(e, e_swapped)), on any test split's captions, aggregated by a part-of-speech role read off
the word's pregroup type (nouns are bare word vectors; n@n words split into determiners / numbers / adjectives-and-modifiers by a
closed word list; types carrying the 'out' leg are the caption root). Also reports per-role 'ignored' fractions (< 0.01) and the
most and least influential frequent words.
  PYTHONPATH=. python tools/tn/s5_pos_influence.py <checkpoint.pt> <out.json> <split>[,<split>...] [N=2000]"""
import sys, json, random, collections
from pathlib import Path
import numpy as np, torch
from modules.utils.general import setup_exp, CheckpointManager
from modules.utils.factory import build_experiment
from modules.data_pipeline.engine import DataEngine

ck_path, out_path, splits = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(',')
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
if out_path.exists(): raise FileExistsError(out_path)
config, DEV, SEED = setup_exp(str(ck_path.parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV); de.image_init(image_model)
ck = CheckpointManager.load_model(ck_path, image_model, text_model, DEV); text_model.to(DEV).eval()
trained = text_model.trained_symbols
words = sorted(s for s in trained if s.startswith('w:'))
DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'each', 'every', 'his', 'her', 'its', 'their', 'my', 'your', 'our',
       'another', 'any', 'no', 'several', 'many', 'few', 'all', 'both'}
NUM = {'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'one', 'dozen', 'couple'}

def role(word, sig):
    if sig is None: return 'noun (bare vector)'
    legs = sig.split('@')
    if 'out' in legs: return f'root ({len(legs)}-leg)'
    if legs == ['n', 'n']: return 'determiner' if word in DET else 'number' if word in NUM else 'adjective/modifier'
    if 'p' in legs: return 'preposition-like (p leg)'
    if 's' in legs: return f'verb/clause ({len(legs)}-leg, s leg)'
    return f'other n-only ({len(legs)}-leg)'

def embed(cs):
    with torch.no_grad():
        return torch.cat([text_model(cs[j:j + 512]).float() for j in range(0, len(cs), 512)])

res = {'checkpoint': str(ck_path), 'epoch': ck.get('epoch')}
for split in splits:
    comp = de.compile_text(split)
    scol = [c for c in comp.columns if c.endswith('_symbols')]
    text_model.extend_symbols([v for c in scol for v in comp[c].tolist() if not isinstance(v, float)], seed=SEED)
    de.precompute_paths(text_model, comp)
    ecol = scol[0].replace('_symbols', '_einsum')   # first caption column = the positive / reference caption
    caps = []
    for e, s in zip(comp[ecol], comp[scol[0]]):
        if isinstance(s, float): continue
        caps += list(zip(e, s)) if s and isinstance(s[0], list) else [(e, s)]
    rng = random.Random(0); rng.shuffle(caps)
    caps = [c for c in caps if all(d['name'] in trained for d in c[1])][:N]
    variants, meta = [], []
    for ci, (eq, syms) in enumerate(caps):
        for k, d in enumerate(syms):
            if not d['name'].startswith('w:'): continue
            nxt = syms[k + 1]['name'] if k + 1 < len(syms) else ''
            sig = nxt.split('_core')[0][2:] if nxt.startswith('t:') else None
            w = d['name'][2:]
            new = [dict(x) for x in syms]; new[k]['name'] = rng.choice(words)
            variants.append((eq, new)); meta.append((ci, w, role(w, sig)))
    b, v = embed(caps), embed(variants)
    ci = torch.tensor([m[0] for m in meta], device=DEV)
    infl = (1 - (b[ci] * v).sum(1) / (b[ci].norm(dim=1) * v.norm(dim=1)).clamp(min=1e-30)).cpu().numpy()
    by_role, by_word, per_cap = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for (c, w, r), x in zip(meta, infl):
        by_role[r].append(x); by_word[(w, r)].append(x); per_cap[c].append((x, r))
    top_role = collections.Counter(max(v)[1] for v in per_cap.values())
    out = {'n_captions': len(caps), 'n_words': len(meta), 'mean_influence': float(infl.mean()),
           'frac_ignored_lt_0.01': float((infl < 0.01).mean()),
           'by_role': {r: {'n': len(x), 'share_of_words': len(x) / len(meta), 'mean': float(np.mean(x)), 'median': float(np.median(x)),
                           'ignored_lt_0.01': float(np.mean(np.array(x) < 0.01)),
                           'is_top_word_frac': top_role[r] / len(per_cap)} for r, x in sorted(by_role.items(), key=lambda kv: -len(kv[1]))}}
    freq_words = [(k, x) for k, x in by_word.items() if len(x) >= 20]
    srt = sorted(freq_words, key=lambda kv: np.mean(kv[1]))
    out['least_influential_words'] = [{'word': k[0], 'role': k[1], 'n': len(x), 'mean': float(np.mean(x))} for k, x in srt[:25]]
    out['most_influential_words'] = [{'word': k[0], 'role': k[1], 'n': len(x), 'mean': float(np.mean(x))} for k, x in srt[::-1][:25]]
    res[split] = out
    print(split, json.dumps({k: v for k, v in out.items() if k in ('n_captions', 'mean_influence', 'frac_ignored_lt_0.01')}))
    for r, x in out['by_role'].items(): print('   ', r, {k: round(v, 3) for k, v in x.items()})
json.dump(res, open(out_path, 'x'), indent=1)
