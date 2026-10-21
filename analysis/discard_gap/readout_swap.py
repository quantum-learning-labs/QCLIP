# Post-selection vs partial trace on the SAME trained weights (cluster GPU job; run from the code dir).
# Every curried sentence circuit is a unitary on 3 sentence wires + m other wires, started in |0>:
#   |Psi> = sum_k sqrt(p_k) |psi_k>|k>   (k = basis state of the m other wires)
#   post-selection (non-discard): |psi_0>, success probability p_0
#   partial trace  (discard):     rho = sum_k p_k |psi_k><psi_k|   (all branches, incl. the post-selected one)
# For each checkpoint (either training method) we compute both readouts, global states, and per-pair scores.
# usage: python readout_swap.py <out.pkl> <ckpt_dir>...
import sys, copy, pickle, yaml, itertools, collections, torch, numpy as np, pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import set_seed, CheckpointManager
from modules.data_pipeline.engine import DataEngine
out_path, ckpts = sys.argv[1], sys.argv[2:]
assert not Path(out_path).exists()
DEV = torch.device('cuda')

def psi_forward(tm, recipes):
    groups = collections.defaultdict(list)
    for i, (e, t) in enumerate(recipes): groups[e].append((i, t))
    thetas = torch.tanh(torch.cat([p for p in tm.params])) * 2 * torch.pi
    out = [None] * len(recipes)
    for e, items in groups.items():
        idx, tl = zip(*items); mb = len(idx)
        plan = tm._plan(e, tl[0]); ops = [None] * plan['n_ops']
        for pos, b in plan['static']:
            buf = getattr(tm, b); ops[pos] = buf.expand(mb, *buf.shape)
        if plan['param_pos']:
            pidx = torch.tensor([[tm.sym2param.get(ts[p]['name'], tm.unk_param_index) for p in plan['param_pos']] for ts in tl], device=thetas.device)
            ang = thetas[pidx].to(tm.precision)
            for op, cols in plan['by_type'].items():
                g = tm.GATE_FNS[op](ang[:, cols])
                for j, c in enumerate(cols): ops[plan['param_pos'][c]] = g[:, j]
        go = plan['expr'](*ops, backend='torch')
        assert go.numel() % (mb * 2 ** tm.out_q) == 0, e
        go = go.reshape(mb, 2 ** tm.out_q, -1)
        for j, i in enumerate(idx): out[i] = go[j]
    return out

def load(cfg, ck):
    ans, img, txt, _ = build_experiment(cfg, DEV)
    de = DataEngine(cfg, ans, DEV)
    ctr, cva = de.compile_text('train'), de.compile_text('val')
    de.text_init(txt, pd.concat([ctr, cva], ignore_index=True)); de.image_init(img)
    img.to(DEV); txt.to(DEV)   # text_init creates the parameters on the CPU
    meta = CheckpointManager.load_model(ck / 'best.pt', img, txt, DEV)
    img.to(DEV); txt.to(DEV); img.eval(); txt.eval()
    return de, img, txt, ctr, cva, meta

def bloch(r, PB): return torch.einsum('bmn,knm->bk', r, PB).real

P1 = [torch.eye(2), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, -1j], [1j, 0]]), torch.tensor([[1, 0], [0, -1]])]
def pauli_basis(n):
    B = []
    for t in itertools.product(range(4), repeat=n):
        if all(i == 0 for i in t): continue
        m = P1[t[0]].to(torch.complex64)
        for i in t[1:]: m = torch.kron(m, P1[i].to(torch.complex64))
        B.append(m)
    return torch.stack(B).to(DEV)

res = []
for ck in map(Path, ckpts):
    cfg = yaml.safe_load(open(ck / 'config.yaml')); set_seed(cfg['seed'])
    trained_disc = cfg['text'].get('discard', False); q = cfg['embedding_qubits']; PB = pauli_basis(q)
    deA, imgA, txtA, _, _, meta = load(copy.deepcopy(cfg), ck)              # as trained (for the sanity check)
    cfgB = copy.deepcopy(cfg); cfgB['text']['discard'] = True; cfgB['loss'] = 'hs'
    deB, _, txtB, ctrB, cvaB, _ = load(cfgB, ck)                              # same weights, no post-selection
    assert txtA.symbols == txtB.symbols, 'symbol order differs between readouts'
    imgA.discard = False                                                      # image circuit has only q wires: pure states
    rec = dict(ckpt=str(ck), trained_discard=trained_disc, seed=cfg['seed'], layers=cfg['text']['layers'], epoch=meta.get('epoch'))
    for split, comp in (('val', cvaB), ('train', ctrB.iloc[:2000])):
        raw = pd.read_pickle(cfg['dataset'][split]['text_path']).iloc[:len(comp)]
        loader = deB.get_loader(comp, split='val')
        S = collections.defaultdict(list); glob_states = []
        with torch.no_grad():
            for bi, b in enumerate(loader):
                psis = psi_forward(txtB, b['caption'])
                ip = imgA(b['pos_image'].to(DEV)).reshape(len(psis), -1); ineg = imgA(b['neg_image'].to(DEV)).reshape(len(psis), -1)
                ip = ip / ip.norm(dim=1, keepdim=True); ineg = ineg / ineg.norm(dim=1, keepdim=True)
                if bi == 0 and split == 'val':   # sanity: our readout reproduces the trained model's own output
                    own = txtA(b['caption']).reshape(len(psis), -1)
                    if trained_disc:
                        ref = torch.stack([(p @ p.conj().T).reshape(-1) for p in psis])
                    else:
                        ref = torch.stack([p[:, 0] for p in psis])
                    rec['sanity_maxdiff'] = float((own - ref).abs().max())
                for j, p in enumerate(psis):
                    rho = p @ p.conj().T
                    pk = (p.abs() ** 2).sum(0)                                     # branch probabilities p_k
                    p0 = float(pk[0]); psi0 = p[:, 0] / pk[0].sqrt().clamp(min=1e-12)
                    ev = torch.linalg.eigvalsh(rho.cpu())   # cusolver eigh is broken in the cluster env; 8x8 on CPU
                    for name, im in (('pos', ip[j]), ('neg', ineg[j])):
                        S[f'tr_{name}'].append(float((im.conj() @ rho @ im).real))
                        S[f'ps_{name}'].append(float((im.conj() @ psi0).abs() ** 2))
                    S['p0'].append(p0); S['purity'].append(float((rho @ rho).diagonal().sum().real)); S['lmax'].append(float(ev[-1]))
                    S['m'].append(int(round(np.log2(p.shape[1])))); S['branch0_is_top'].append(float((psi0.conj() @ rho @ psi0).real) / float(ev[-1]))
                    if split == 'val': glob_states.append(p.reshape(-1).cpu())
                    S['bloch_tr'].append(bloch(rho[None], PB)[0].cpu().numpy()); S['bloch_ps'].append(bloch(torch.outer(psi0, psi0.conj())[None], PB)[0].cpu().numpy())
                if split == 'val' and bi == 0: S['img_bloch'] = []
                if split == 'val': S['img_bloch'] += list(bloch(torch.einsum('bi,bj->bij', ip, ip.conj()), PB).cpu().numpy())
        df = pd.DataFrame({k: v for k, v in S.items() if k not in ('bloch_tr', 'bloch_ps', 'img_bloch')})
        for c in ('subj_neg', 'verb_neg', 'obj_neg', 'pos_triplet'): df[c] = raw[c].values
        df['einsum'] = comp['corrected_sentence_einsum'].values
        rec[split] = df
        rec[split + '_bloch'] = {k: np.stack(S[k]) for k in ('bloch_tr', 'bloch_ps') + (('img_bloch',) if split == 'val' else ())}
        if split == 'val': rec['val_global'] = glob_states
        print(ck.name, split, 'acc trace %.4f  postsel %.4f' % ((df.tr_pos > df.tr_neg).mean(), (df.ps_pos > df.ps_neg).mean()),
              'sanity', rec.get('sanity_maxdiff'), flush=True)
    res.append(rec)
pickle.dump(res, open(out_path, 'xb'))
