# Report section 22, future direction 1 (objective test), exact, no shots.
# Hypothesis: the discard objective (Pauli_InfoNCE) rewards something other than accuracy, so Theorem A's channel
#   Phi(rho) = K0 rho K0^+ + Tr[(I-K0^+K0)rho] I/d, which has post-selection's decisions, scores a WORSE loss than the
#   trained discard optimum. Note: Pauli_InfoNCE's similarity is the cosine of the identity-free Pauli vectors, and the
#   Pauli vector of Phi(rho) is that of K0 rho K0^+, so Phi's logits equal the post-selected state's (p0 cancels).
# For every checkpoint and split, scores each batch's text x image matrix (and each caption's hard-negative image)
#   post-selected checkpoint: with Phi (and the normalised post-selected state, as a check that they agree)
#   discard checkpoint: with the trained density matrices
# then evaluates the same criterion (Pauli_InfoNCE, temperature 0.07, label smoothing 0.1, purity 0), plus the loss at
# the best fixed logit scale (grid 1..100) and the accuracies (Tr(rho tau) as the engine measures, and Pauli cosine).
# usage: PYTHONPATH=. python analysis/cptp/followup/loss_at_phi.py <out_dir> <best.pt>...
import sys, json, math, torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from modules.utils.general import setup_exp, resolve_eval_config
from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.criteria import Pauli_InfoNCE

def models(config, ck, DEV):   # as branch_eval.py
    ansatz, img, txt, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(txt, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(img)
    CheckpointManager.load_model(ck, img, txt, DEV); img.to(DEV).eval(); txt.to(DEV).eval()   # params stay on CPU after init/load
    return de, img, txt

def vec(x): return x.reshape(x.shape[0], -1).to(torch.complex128)
def as_dm(x):
    if x.dim() == 3: return x.to(torch.complex128)
    v = vec(x); d = int(round(v.shape[1] ** 0.5))
    if d * d == v.shape[1] and d >= 4 and (d & (d - 1)) == 0 and v.shape[1] > 8: return v.view(-1, d, d)
    return v[:, :, None] * v.conj()[:, None, :]
def tr(r, t): return torch.einsum('bij,bji->b', r, t).real

TEMP, LS = 0.07, 0.1
SCALES = torch.logspace(0, 2, 41)   # logit scale grid, 1..100 (1/0.07 = 14.3 is the trained value)

def phi(psi):   # Theorem A's channel on the unnormalised post-selected statevector
    p0 = (psi.abs() ** 2).sum(1); d = psi.shape[1]
    return psi[:, :, None] * psi.conj()[:, None, :] + ((1 - p0) / d)[:, None, None] * torch.eye(d, dtype=psi.dtype, device=psi.device)

def ce(logits, ls=LS):
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device), label_smoothing=ls)

def losses(S, Sn, scale):   # S [B,B] text x image cosines, Sn [B,B] text x hard-negative-image cosines
    sym = 0.5 * (ce(S * scale) + ce(S.T * scale))
    hn = 0.5 * (ce(torch.cat([S, Sn], 1) * scale) + ce(S.T * scale))   # trainer with hard_negatives: true
    return sym.item(), hn.item()

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
for ck in sys.argv[2:]:
    ck = Path(ck); out = out_dir / f'{ck.parent.name}.json'
    if out.exists(): print('exists', out); continue
    config, DEV, _ = setup_exp(resolve_eval_config(None, ck))
    discard = bool(config['text'].get('discard'))
    de, img, txt = models(config, ck, DEV)
    crit = Pauli_InfoNCE(temperature=TEMP, label_smoothing=LS, num_qubits=config['embedding_qubits'], purity_weight=0.0).to(DEV)
    res = {'checkpoint': str(ck), 'discard': discard}
    readouts = ['trained'] if discard else ['phi', 'ps_norm']
    with torch.no_grad():
        for split in ['val', 'std']:
            df = de.compile_text(split); loader = de.get_loader(df, split=split)
            acc = {r: {'tr': [], 'cos': [], 'cos_margin': [], 'tr_margin': []} for r in readouts}
            L = {r: {'sym': [], 'hn': [], 'grid_sym': torch.zeros(len(SCALES)), 'grid_hn': torch.zeros(len(SCALES)), 'check': []} for r in readouts}
            nb = 0
            for b in loader:
                ip, ineg = as_dm(img(b['pos_image'].to(DEV))), as_dm(img(b['neg_image'].to(DEV)))
                ip, ineg = ip / ip.diagonal(dim1=1, dim2=2).sum(-1).real[:, None, None], ineg / ineg.diagonal(dim1=1, dim2=2).sum(-1).real[:, None, None]
                out_t = txt(b['caption'])
                states = {'trained': as_dm(out_t)} if discard else {}
                if not discard:
                    psi = vec(out_t); states['phi'] = phi(psi)
                    u = psi / psi.norm(dim=1, keepdim=True).clamp(min=1e-15); states['ps_norm'] = u[:, :, None] * u.conj()[:, None, :]
                for r, rho in states.items():
                    S = crit.compute_similarity(rho, ip).double(); Sn = crit.compute_similarity(rho, ineg).double()
                    s_ref = crit(rho, ip, ineg).item()   # library forward, hard negatives, scale 1/0.07
                    sym, hn = losses(S, Sn, 1 / TEMP); L[r]['sym'].append(sym); L[r]['hn'].append(hn); L[r]['check'].append(abs(hn - s_ref))
                    for k, sc in enumerate(SCALES):
                        a, h = losses(S, Sn, sc.item()); L[r]['grid_sym'][k] += a; L[r]['grid_hn'][k] += h
                    tp, tn = tr(rho, ip), tr(rho, ineg); cp = S.diagonal()
                    acc[r]['tr'] += (tp > tn).tolist(); acc[r]['cos'] += (cp > Sn.diagonal()).tolist()
                    acc[r]['tr_margin'] += (tp - tn).tolist(); acc[r]['cos_margin'] += (cp - Sn.diagonal()).tolist()
                nb += 1
            for r in readouts:
                gs, gh = L[r]['grid_sym'] / nb, L[r]['grid_hn'] / nb
                m = lambda x: float(torch.tensor(x).mean()); md = lambda x: float(torch.tensor(x).median())
                res.setdefault(split, {})[r] = {
                    'n': len(acc[r]['tr']), 'batches': nb,
                    'loss_sym': m(L[r]['sym']), 'loss_hn': m(L[r]['hn']), 'max_abs_diff_vs_library': max(L[r]['check']),
                    'best_scale_sym': float(SCALES[gs.argmin()]), 'loss_sym_at_best': float(gs.min()),
                    'best_scale_hn': float(SCALES[gh.argmin()]), 'loss_hn_at_best': float(gh.min()),
                    'acc_tr': m(acc[r]['tr']), 'acc_cos': m(acc[r]['cos']),
                    'median_cos_margin': md(acc[r]['cos_margin']), 'median_tr_margin': md(acc[r]['tr_margin']),
                    'grid_scales': SCALES.tolist(), 'grid_sym': gs.tolist(), 'grid_hn': gh.tolist()}
    json.dump(res, open(out, 'w'), indent=1)
    print(ck.parent.name, {s: {r: {k: round(v, 4) for k, v in d.items() if isinstance(v, float)} for r, d in res[s].items()} for s in ('val', 'std')})
