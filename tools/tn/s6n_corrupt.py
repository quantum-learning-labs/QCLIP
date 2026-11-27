"""Step 6n: robustness to ONE bad leaf. On the 5k test split, each caption gets one word leaf ('w:') chosen uniformly
(seeded) and replaced by (a) the neutral leaf (word_base alone: the word's information removed) or (b) a trained word drawn
by training frequency (a wrong but valid word). Reports tie-neutral t2i R@1/5/10 clean / neutral / swap, so TN and BoW can
be compared by how much one bad leaf costs. Runs with text.oov_init=neutral (true OOV words neutral too).
  PYTHONPATH=. python tools/tn/s6n_corrupt.py <out.json> <ckpt.pt> [seed]"""
import sys, json, copy, random, torch
import numpy as np
from collections import Counter
from pathlib import Path
from modules.utils.factory import build_experiment, load_obj
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import MMEvaluator
sys.path.insert(0, str(Path(__file__).parent))
from s5e_gap_breakdown_lib import encode, per_caption

ck, rseed = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
config['text']['oov_init'] = 'neutral'
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV)
tr, va = de.compile_text('train'), de.compile_text('val')
import pandas as pd
de.text_init(text_model, pd.concat([tr, va], ignore_index=True))
CheckpointManager.load_model(ck, image_model, text_model, DEV)
text_model.to(DEV).eval(); image_model.to(DEV).eval()
wfreq = Counter(s['name'] for ss in tr['captions_symbols'] for syms in ss for s in syms if s['name'].startswith('w:'))
words, weights = zip(*wfreq.items())
te = de.compile_text('coco5k')
rng = random.Random(rseed)
def corrupt(df, mode):
    df = df.copy(); col = []
    r = random.Random(rseed)   # same leaf chosen in both modes
    for ss in df['captions_symbols']:
        new = []
        for syms in ss:
            syms = copy.deepcopy(syms); idx = [i for i, s in enumerate(syms) if s['name'].startswith('w:')]
            if idx:
                i = r.choice(idx); old = syms[i]['name']
                if mode == 'neutral': syms[i]['name'] = 'w:__neutral__'
                else:
                    w = old
                    while w == old: w = r.choices(words, weights)[0]
                    syms[i]['name'] = w
            new.append(syms)
        col.append(new)
    df['captions_symbols'] = col
    return df
ev = MMEvaluator(image_model, text_model, DEV); mapper = load_obj(config['dataset']['eval_mapper'])
res = {'ckpt': ck, 'seed': rseed}
with torch.no_grad():
    sets = {'clean': te, 'neutral': corrupt(te, 'neutral'), 'swap': corrupt(te, 'swap')}
    for name, df in sets.items():
        text_model.extend_symbols([v for c in df.columns if c.endswith('_symbols') for v in df[c].tolist() if not isinstance(v, float)], seed=SEED)
        I, C, img_of, caps = encode(ev, de.get_loader(df, split='coco5k'), mapper)
        R = per_caption(ev, I, C, img_of)
        res[name] = {**{f't2i_{k}': float(v.mean()) for k, v in R.items()}, 'R10': R['R10'].tolist()}
        print(name, {k: round(v, 4) for k, v in res[name].items() if k.startswith('t2i')}, flush=True)
json.dump(res, open(sys.argv[1], 'x'))
