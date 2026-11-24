"""Step 5e E0: generalisation gap of a checkpoint. COCO retrieval rsum (i2t + t2i R@1/5/10, x100) on 5k TRAINING images
(the first 5000 rows of the train split, all their captions), on the val holdout and on the 5k test split, with the same
encode and recall code (as tools/tn/wino_hub.py). Large train >> test: the tensors memorise; train ~ test: capacity.
  PYTHONPATH=. python tools/tn/s5e_train_gap.py <out.json> <ckpt.pt>..."""
import sys, json, torch
import pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment, load_obj
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import MMEvaluator

def encode_retrieval(ev, loader, mapper):
    imgs, txts, coords, gi, gt = [], [], [], 0, 0
    for batch in loader:
        img_batch, txt_batch = mapper(batch)
        imgs.append(ev._encode_img(img_batch)); pay = []
        for caps in txt_batch:
            if isinstance(caps[0], str) or isinstance(caps[0][0], dict): caps = [caps]
            for c in caps: coords.append((gi, gt)); gt += 1; pay.append((c[0], c[1]))
            gi += 1
        txts.append(ev._encode_txt(pay))
    return torch.cat(imgs), torch.cat(txts), coords

def rsum(ev, I, C, coords):
    S = ev._readout(I @ C.conj().T)
    mask = torch.zeros(len(I), len(C), dtype=torch.bool, device=I.device)
    r, c = torch.tensor(coords, device=I.device).T; mask[r, c] = True
    m = {**ev._calculate_recall(S, mask, 'i2t'), **ev._calculate_recall(S.T, mask.T, 't2i')}
    return 100 * sum(m[f'{d}R{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))

out = {}
for ck in sys.argv[2:]:
    config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    tr, va = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([tr, va], ignore_index=True))
    CheckpointManager.load_model(ck, image_model, text_model, DEV)
    text_model.to(DEV).eval(); image_model.to(DEV).eval()
    ev = MMEvaluator(image_model, text_model, DEV); mapper = load_obj(config['dataset']['eval_mapper'])
    res = {}
    with torch.no_grad():
        te = de.compile_text('coco5k')
        text_model.extend_symbols([v for c in te.columns if c.endswith('_symbols') for v in te[c].tolist() if not isinstance(v, float)], seed=SEED)
        keep = [c for c in tr.columns if not c.startswith('captions_neg')]   # the train negatives are not evaluated
        for name, df, split in (('train5k', tr[keep].head(5000).reset_index(drop=True), 'val'), ('val', va, 'val'), ('test5k', te, 'coco5k')):
            I, C, coords = encode_retrieval(ev, de.get_loader(df, split=split), mapper)
            res[f'{name}_rsum'] = rsum(ev, I, C, coords); res[f'{name}_n_img'] = len(I)
    res['gap_train_minus_test'] = res['train5k_rsum'] - res['test5k_rsum']
    out[ck] = res; print(ck, {k: round(v, 1) for k, v in res.items()}, flush=True)
json.dump(out, open(sys.argv[1], 'x'), indent=1)
