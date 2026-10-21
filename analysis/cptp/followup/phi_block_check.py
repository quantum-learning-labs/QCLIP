# Check of text.phi_block (report sec 22, direction 2): a post-selected checkpoint loaded into the discard circuit with
# the Phi block must give rho == Phi(psi) = psi psi^+ + (1 - p0) I/d, with psi the post-selected (unnormalised) output,
# so its std/val decisions equal post-selection's. Also times one forward of each.
# usage: PYTHONPATH=. python analysis/cptp/followup/phi_block_check.py <post-selected best.pt>
import sys, copy, time, torch
import pandas as pd
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine

def models(config, ck, DEV):   # as branch_eval.py
    ansatz, img, txt, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(txt, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(img)
    CheckpointManager.load_model(ck, img, txt, DEV); img.to(DEV).eval(); txt.to(DEV).eval()
    return de, img, txt

ck = sys.argv[1]
config, DEV, _ = setup_exp(resolve_eval_config(None, ck))
assert not config['text'].get('discard')
de_p, img_p, txt_p = models(config, ck, DEV)
cfg = copy.deepcopy(config); cfg['text']['discard'] = True; cfg['text']['phi_block'] = True; cfg['loss'] = 'pauli'; cfg['purity_weight'] = 0.0
de_d, img_d, txt_d = models(cfg, ck, DEV)
assert txt_d.symbols == txt_p.symbols, 'symbol sets differ'
dm = lambda v: v[:, :, None] * v.conj()[:, None, :]
worst, n, agree, acc_p, acc_d, tp, td = 0.0, 0, 0, 0, 0, 0.0, 0.0
with torch.no_grad():
    for split in ('val', 'std'):
        lp = de_p.get_loader(de_p.compile_text(split), split=split); ld = de_d.get_loader(de_d.compile_text(split), split=split)
        for bp, bd in zip(lp, ld):
            t0 = time.time(); psi = txt_p(bp['caption']).reshape(len(bp['pos_image']), -1).to(torch.complex128); t1 = time.time()
            rho = txt_d(bd['caption']).to(torch.complex128); t2 = time.time(); tp += t1 - t0; td += t2 - t1
            d = psi.shape[1]; p0 = (psi.abs() ** 2).sum(1)
            phi = dm(psi) + ((1 - p0) / d)[:, None, None] * torch.eye(d, dtype=psi.dtype, device=psi.device)
            worst = max(worst, (rho - phi).abs().max().item())
            ip = img_p(bp['pos_image'].to(DEV)).reshape(len(psi), -1).to(torch.complex128); ineg = img_p(bp['neg_image'].to(DEV)).reshape(len(psi), -1).to(torch.complex128)
            sp = (psi.conj() * ip).sum(1).abs() ** 2 - (psi.conj() * ineg).sum(1).abs() ** 2
            jp = img_d(bd['pos_image'].to(DEV)); jn = img_d(bd['neg_image'].to(DEV)); dd = int(round(jp[0].numel() ** 0.5))
            jp, jn = jp.reshape(-1, dd, dd).to(torch.complex128), jn.reshape(-1, dd, dd).to(torch.complex128)
            sd = torch.einsum('bij,bji->b', rho, jp).real - torch.einsum('bij,bji->b', rho, jn).real
            agree += ((sp > 0) == (sd > 0)).sum().item(); acc_p += (sp > 0).sum().item(); acc_d += (sd > 0).sum().item(); n += len(psi)
        print(f'{split}: n {n} max|rho - Phi(psi)| {worst:.2e} decisions agree {agree}/{n} acc PS {acc_p / n:.4f} acc phi_block {acc_d / n:.4f} '
              f'forward s: PS {tp:.1f} phi_block {td:.1f}')
        worst, n, agree, acc_p, acc_d, tp, td = 0.0, 0, 0, 0, 0, 0.0, 0.0
