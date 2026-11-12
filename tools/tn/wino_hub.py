"""Post-hoc hubness corrections (step 5, E3 follow-up): does removing the per-caption / per-image score offset make Winoground's
text and image scores significant, and what does it do to COCO 5k retrieval? Bank = the COCO val split (val_holdout), never
test. Corrections of a score s(c, i) (raw = the model's readout):
  csls{k}:  s - a*(r_c + r_i) with r_x = mean of x's top-k scores against the other modality's bank (CSLS, Conneau et al. 2018;
            alpha a in {0.5, 1}; a=1 is CSLS, a<1 NNN-like, Chowdhury et al. 2024); k=all = the bank mean
  centre:   embeddings minus their modality's bank mean, renormalised (the in-model version: centring both spaces)
Winoground also reports the 'pairwise' diagnostic [s00 + s11 > s01 + s10] (= any per-item row/column normalisation of a 2x2:
text = image = group, chance 0.5, NOT comparable with the standard scores). Significance threshold of the standard text/image
score (chance 0.25): one-sided binomial p < 0.05 at the item count.
  PYTHONPATH=. python tools/tn/wino_hub.py <out.json> <ckpt.pt>..."""
import sys, json, torch
import pandas as pd
from pathlib import Path
from scipy.stats import binom
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

def topk_mean(S, k):   # row-wise mean of the top-k entries (k=None: all)
    return S.mean(1) if k is None else S.topk(k, dim=1).values.mean(1)

def corrections(I, C, bI, bC, ev):
    # yields (name, score matrix images x captions) for each correction
    S = ev._readout(I @ C.conj().T)
    yield 'raw', S
    SIb = ev._readout(I @ bC.conj().T); SCb = ev._readout(C @ bI.conj().T)
    for k in (10, None):
        ri, rc = topk_mean(SIb, k), topk_mean(SCb, k)
        for a in (0.5, 1.0):
            yield f"csls{k or 'all'}_a{a}", S - a * (ri[:, None] + rc[None, :])
    mi, mc = bI.mean(0), bC.mean(0)
    Ic = (I - mi); Ic = Ic / Ic.norm(dim=1, keepdim=True); Cc = (C - mc); Cc = Cc / Cc.norm(dim=1, keepdim=True)
    yield 'centre', ev._readout(Ic @ Cc.conj().T)

out = {}
for ck in sys.argv[2:]:
    config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(text_model, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True))
    CheckpointManager.load_model(ck, image_model, text_model, DEV)
    text_model.to(DEV).eval(); image_model.to(DEV).eval()
    ev = MMEvaluator(image_model, text_model, DEV)
    mapper = load_obj(config['dataset']['eval_mapper'])
    res = {}
    with torch.no_grad():
        dfs = {s: de.compile_text(s) for s in ('val', 'coco5k', 'winoground')}
        for s in ('coco5k', 'winoground'):
            text_model.extend_symbols([v for c in dfs[s].columns if c.endswith('_symbols') for v in dfs[s][c].tolist() if not isinstance(v, float)], seed=SEED)
        bI, bC, _ = encode_retrieval(ev, de.get_loader(dfs['val'], split='val'), mapper)
        # COCO 5k test retrieval under each correction
        tI, tC, coords = encode_retrieval(ev, de.get_loader(dfs['coco5k'], split='coco5k'), mapper)
        mask = torch.zeros(len(tI), len(tC), dtype=torch.bool, device=tI.device)
        r, c = torch.tensor(coords, device=tI.device).T; mask[r, c] = True
        for name, S in corrections(tI, tC, bI, bC, ev):
            m = {**ev._calculate_recall(S, mask, 'i2t'), **ev._calculate_recall(S.T, mask.T, 't2i')}
            res[f'coco5k_{name}_rsum'] = 100 * sum(m[f'{d}R{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))
        # Winoground: per item the 2 captions x 2 images block of each correction (bank terms per caption / image)
        W = {'i0': [], 'i1': [], 'c0': [], 'c1': []}
        for batch in de.get_loader(dfs['winoground'], split='winoground'):
            W['i0'].append(ev._encode_img(batch['image_0'])); W['i1'].append(ev._encode_img(batch['image_1']))
            W['c0'].append(ev._encode_txt(batch['caption_0'])); W['c1'].append(ev._encode_txt(batch['caption_1']))
        W = {k: torch.cat(v) for k, v in W.items()}
        n = len(W['i0']); I = torch.cat([W['i0'], W['i1']]); C = torch.cat([W['c0'], W['c1']])
        thr = int(binom.ppf(0.95, n, 0.25)) + 1   # smallest count with one-sided p < 0.05
        res['wino_n'], res['wino_sig_threshold'] = n, thr / n
        ar = torch.arange(n, device=I.device)
        for name, S in corrections(I, C, bI, bC, ev):
            s00, s01, s10, s11 = S[ar, ar], S[ar, ar + n], S[ar + n, ar], S[ar + n, ar + n]   # s[image, caption]
            t = (s00 > s01) & (s11 > s10); i = (s00 > s10) & (s11 > s01)
            res[f'wino_{name}'] = {'txt': t.float().mean().item(), 'img': i.float().mean().item(), 'grp': (t & i).float().mean().item(),
                                   'pairwise': (s00 + s11 > s01 + s10).float().mean().item(),
                                   'txt_p': float(binom.sf(int(t.sum()) - 1, n, 0.25)), 'img_p': float(binom.sf(int(i.sum()) - 1, n, 0.25))}
    out[ck] = res
    print(ck, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v)
                          for k, v in res.items()}), flush=True)
json.dump(out, open(sys.argv[1], 'x'), indent=1)
