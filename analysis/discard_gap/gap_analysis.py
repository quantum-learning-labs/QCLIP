# Diagnose the discard vs non-discard validation gap on trained checkpoints (best.pt).
# For each checkpoint: per-pair image-choice scores on val (and a train subset) under several
# similarity functions, plus purities, negative type and the number of traced qubits per sentence.
# usage (from qclip/): PYTHONPATH=. python gap_analysis.py <out.pkl> <ckpt_dir>...
import sys, pickle, torch, numpy as np, pandas as pd, re
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
torch.set_num_threads(int(__import__('os').environ.get('NTHREADS', 8)))
out_path, ckpts = sys.argv[1], sys.argv[2:]
assert not Path(out_path).exists(), f'{out_path} exists'

def pauli_basis(n):
    P = [torch.eye(2), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, -1j], [1j, 0]]), torch.tensor([[1, 0], [0, -1]])]
    P = [p.to(torch.complex64) for p in P]
    import itertools
    B = []
    for t in itertools.product(range(4), repeat=n):
        if all(i == 0 for i in t): continue
        m = P[t[0]]
        for i in t[1:]: m = torch.kron(m, P[i])
        B.append(m)
    return torch.stack(B)

def sqrtm_psd(r):
    w, v = torch.linalg.eigh(r)
    return (v * w.clamp(min=0).sqrt().unsqueeze(-2)) @ v.conj().transpose(-1, -2)

def fidelity(r, s):   # Uhlmann fidelity (Tr sqrt(sqrt r s sqrt r))^2, batched 8x8
    sr = sqrtm_psd(r)
    w = torch.linalg.eigvalsh(sr @ s @ sr).clamp(min=0)
    return w.sqrt().sum(-1) ** 2

res = []
for ck in ckpts:
    ck = Path(ck); cfg_path = ck / 'config.yaml'
    config, DEV, _ = setup_exp(str(cfg_path))
    disc = config['text'].get('discard', False); q = config['embedding_qubits']; d = 2 ** q
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    ctrain, cval = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([ctrain, cval], ignore_index=True)); de.image_init(image_model)
    meta = CheckpointManager.load_model(ck / 'best.pt', image_model, text_model, DEV)
    image_model.eval(); text_model.eval()
    PB = pauli_basis(q)
    for split, comp in (('val', cval), ('train', ctrain.iloc[:2000])):
        raw = pd.read_pickle(config['dataset'][split]['text_path'])
        loader = de.get_loader(comp, split='val')   # no shuffle
        T, IP, IN = [], [], []
        with torch.no_grad():
            for b in loader:
                T.append(text_model(b['caption']).cpu()); IP.append(image_model(b['pos_image'].to(DEV)).cpu()); IN.append(image_model(b['neg_image'].to(DEV)).cpu())
        T, IP, IN = torch.cat(T), torch.cat(IP), torch.cat(IN)
        n = len(T)
        if disc:
            rt, rp, rn = T.reshape(n, d, d), IP.reshape(n, d, d), IN.reshape(n, d, d)
        else:   # pure states -> density matrices (unnormalised text: post-selection norm kept separately)
            t, p, m = T.reshape(n, d), IP.reshape(n, d), IN.reshape(n, d)
            tn = t.norm(dim=1)
            rt = torch.einsum('bi,bj->bij', t, t.conj()); rp = torch.einsum('bi,bj->bij', p, p.conj()); rn = torch.einsum('bi,bj->bij', m, m.conj())
        tr = lambda r: r.diagonal(dim1=1, dim2=2).sum(-1).real
        hs = lambda a, b: torch.einsum('bij,bji->b', a, b).real          # Tr(a b)
        pur = lambda r: hs(r, r) / tr(r) ** 2
        unit = lambda r: r / tr(r).clamp(min=1e-12)[:, None, None]
        ut, up, un = unit(rt), unit(rp), unit(rn)
        pv = lambda r: torch.einsum('bmn,knm->bk', r, PB).real
        vt, vp, vn = pv(ut), pv(up), pv(un)
        cos = lambda a, b: (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
        nhs = lambda a, b: hs(a, b) / (hs(a, a) * hs(b, b)).sqrt()
        scores = {
            'hs': (hs(ut, up), hs(ut, un)),                 # = the logged hard_neg_acc metric (trace-normalised)
            'fid': (fidelity(ut, up), fidelity(ut, un)),
            'pauli_cos': (cos(vt, vp), cos(vt, vn)),
            'nhs': (nhs(ut, up), nhs(ut, un)),
            'trdist': (-0.5 * torch.linalg.eigvalsh(ut - up).abs().sum(-1), -0.5 * torch.linalg.eigvalsh(ut - un).abs().sum(-1)),
        }
        df = pd.DataFrame({k + '_correct': (a > b).numpy() for k, (a, b) in scores.items()})
        df['hs_margin'] = (scores['hs'][0] - scores['hs'][1]).numpy()
        df['pur_text'], df['pur_pos'], df['pur_neg'] = pur(rt).numpy(), pur(rp).numpy(), pur(rn).numpy()
        df['img_pair_hs'] = hs(up, un).numpy()            # how similar the two images are
        df['text_norm'] = (tr(rt).sqrt().numpy() if not disc else tr(rt).numpy())
        einsum = comp['corrected_sentence_einsum'].tolist() if 'corrected_sentence_einsum' in comp else [None] * n
        df['n_wires_out'] = [len(e.split('->')[1]) if e else -1 for e in einsum]
        df['n_inputs'] = [len(re.findall(',', e.split('->')[0])) + 1 if e else -1 for e in einsum]
        rr = raw.iloc[:n] if len(raw) >= n else raw
        for c in ('subj_neg', 'verb_neg', 'obj_neg'):
            if c in rr: df[c] = rr[c].values[:n]
        df['n_words'] = [len(x) for x in rr['corrected_sentence_diagram'].values[:n]] if 'corrected_sentence_diagram' in rr else -1
        df['len_match'] = len(raw) == n
        res.append(dict(ckpt=str(ck), discard=disc, seed=config['seed'], split=split, epoch=meta.get('epoch'), df=df))
        print(ck.name, split, n, 'len(raw)=', len(raw), {k: round(float(df[k + '_correct'].mean()), 4) for k in scores}, flush=True)
pickle.dump(res, open(out_path, 'xb'))
