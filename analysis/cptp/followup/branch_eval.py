# Report section 22, experiments 1-2 (exact, no shots).
#  post-selected checkpoint: accuracy of post-selection vs Theorem A's channel Phi(rho) = K0 rho K0^+ + Tr[(I-K0^+K0)rho] I/d
#      on std (image choice: must be identical) and on the swap sets (text choice: may differ, Theorem B)
#  discard checkpoint: split each std margin Tr[rho (tau+ - tau-)] into the success-branch term (same weights, post-selected
#      compile, unnormalised) and the failure-branch remainder; decision flips caused by the failure branches, by negative type
# usage: PYTHONPATH=. python analysis/cptp/followup/branch_eval.py <out_dir> <best.pt>...
import sys, json, copy, torch
from pathlib import Path
import pandas as pd
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
EXTRA = {'swap_full': 'data/svo-probes/processed/swap_full.pkl', 'swap_animate': 'data/svo-probes/processed/swap_animate.pkl'}

def models(config, ck, DEV):
    ansatz, img, txt, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(txt, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(img)
    CheckpointManager.load_model(ck, img, txt, DEV); img.eval(); txt.eval()
    return de, img, txt

def vec(x): return x.reshape(x.shape[0], -1).to(torch.complex128)
def as_dm(x):
    if x.dim() == 3: return x.to(torch.complex128)
    v = vec(x); d = int(round(v.shape[1] ** 0.5))
    if d * d == v.shape[1] and d >= 4 and (d & (d - 1)) == 0 and v.shape[1] > 8: return v.view(-1, d, d)
    return v[:, :, None] * v.conj()[:, None, :]
def tr(r, t): return torch.einsum('bij,bji->b', r, t).real

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
for ck in sys.argv[2:]:
    ck = Path(ck); out = out_dir / f'{ck.parent.name}.json'
    if out.exists(): print('exists', out); continue
    config, DEV, _ = setup_exp(resolve_eval_config(None, ck))
    for n, p in EXTRA.items(): config['dataset']['test'][n] = {**copy.deepcopy(config['dataset']['test']['swap']), 'text_path': p}
    discard = bool(config['text'].get('discard'))
    de, img, txt = models(config, ck, DEV)
    if discard:   # same weights through the post-selected compile = unnormalised success branch
        cfg_p = copy.deepcopy(config); cfg_p['text']['discard'] = False; cfg_p['text']['postselect_types'] = []; cfg_p['loss'] = 'fs'
        de_p, _, txt_p = models(cfg_p, ck, DEV)
    res = {'checkpoint': str(ck), 'discard': discard}
    with torch.no_grad():
        for split in ['std', 'swap', *EXTRA]:
            choice = 'image' if config['dataset']['test'][split]['task'] == 'evaluate_image_choice' else 'text'
            df = de.compile_text(split); loader = de.get_loader(df, split=split)
            if discard: loader_p = de_p.get_loader(de_p.compile_text(split), split=split)
            acc = {k: [] for k in ('ps', 'phi', 'tr', 'b0', 'flip')}; ratio = []
            for bi, batch in enumerate(zip(loader, loader_p) if discard else ((b, None) for b in loader)):
                b, bp = batch
                if choice == 'image':
                    t_pos = t_neg = (b['caption'], bp['caption'] if bp else None)
                    i_pos, i_neg = as_dm(img(b['pos_image'].to(DEV))), as_dm(img(b['neg_image'].to(DEV)))
                    caps = [(b['caption'], bp['caption'] if bp else None)] * 2; ims = [i_pos, i_neg]
                else:
                    i = as_dm(img(b['image'].to(DEV))); ims = [i, i]
                    caps = [(b['pos_caption'], bp['pos_caption'] if bp else None), (b['neg_caption'], bp['neg_caption'] if bp else None)]
                if not discard:
                    S = []
                    for (c, _), im in zip(caps, ims):
                        psi = vec(txt(c)); p0 = (psi.abs() ** 2).sum(1); d = psi.shape[1]
                        n = torch.einsum('bi,bij,bj->b', psi.conj(), im, psi).real
                        S.append((n / p0.clamp(min=1e-30), n + (1 - p0) / d))
                    acc['ps'] += (S[0][0] > S[1][0]).tolist(); acc['phi'] += (S[0][1] > S[1][1]).tolist()
                else:
                    T, B0 = [], []
                    for (c, cp), im in zip(caps, ims):
                        rho = as_dm(txt(c)); psi = vec(txt_p(cp))
                        T.append(tr(rho, im)); B0.append(torch.einsum('bi,bij,bj->b', psi.conj(), im, psi).real)
                    dt, d0 = T[0] - T[1], B0[0] - B0[1]
                    acc['tr'] += (dt > 0).tolist(); acc['b0'] += (d0 > 0).tolist(); acc['flip'] += ((dt > 0) != (d0 > 0)).tolist()
                    ratio += ((dt - d0).abs() / d0.abs().clamp(min=1e-12)).tolist()
            r = {k: sum(v) / len(v) for k, v in acc.items() if v}; r['n'] = len(next(v for v in acc.values() if v))
            if discard:
                r['median_fail_over_success_margin'] = float(torch.tensor(ratio).median())
                if split == 'std':
                    for t in ('subj_neg', 'verb_neg', 'obj_neg'):
                        m = df[t].astype(bool).values
                        r[t] = {k: sum(x for x, y in zip(acc[k], m) if y) / max(m.sum(), 1) for k in ('tr', 'b0', 'flip')}
            res[split] = r
    json.dump(res, open(out, 'w'), indent=1); print(ck.parent.name, json.dumps({k: v for k, v in res.items() if k != 'checkpoint'})[:600])
