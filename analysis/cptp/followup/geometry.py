# Geometry of the trained sentence circuits (next_steps X4.1, X6.1, X6.2), exact, no shots.
#  DLA: every text gate is H, CX or a single-qubit Pauli rotation, so pushing the fixed Clifford gates to the front turns
#      each rotation into a Pauli-string generator; the Lie closure of Pauli strings is spanned by Pauli strings, found by
#      BFS over ad_g (dim g = number of strings reached). Per circuit structure and per isolated word box (N wires, L layers).
#  Fisher rank: per sentence, the quantum Fisher information of the full (3+m)-qubit pure state, of the post-selected
#      state |phi> = psi_0/sqrt(p0), and of the traced state rho (SLD QFI), w.r.t. the angles of that sentence's gates.
#      Dataset level: rank of the stacked readout-state Jacobians of all sampled sentences w.r.t. all text angles.
#  Magic / entropy: stabiliser Renyi entropy M2 of |phi>; mixed SRE M2~ = -log2(sum_P Tr(rho P)^4 / sum_P Tr(rho P)^2)
#      of rho (= M2 for pure states, 0 for I/d); Renyi-2 entropy S2 = -log2 Tr rho^2 (= S-E entanglement, Psi is pure).
# Every checkpoint is compiled in discard mode, so the circuit returns Psi (d x 2^m) and both readouts come from the same
# weights (as in readout_swap.py). 'init' = the same circuits at the training init (text_init, id_init) without a checkpoint.
# usage: PYTHONPATH=. python analysis/cptp/followup/geometry.py <out_dir> init|<ckpt_dir>...   (env GEO_N sentences, default 300)
import os; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, copy, json, yaml, itertools, numpy as np, torch, pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import set_seed, CheckpointManager
from modules.data_pipeline.engine import DataEngine
DEV, C, R = torch.device('cuda'), torch.complex128, torch.float64
N_SENT = int(os.environ.get('GEO_N', 300))
REF_CFG = '/cs/research/pplv/comp_bridge/qclip_snapshots/ckpt_from_hpc/hp_hp2_nodiscard_tl3_qlr3e-3_s0/config.yaml'

# ---------- complex128 gates (gates.py is complex64) ----------
def _cs(a): return torch.cos(a / 2).to(C), torch.sin(a / 2).to(C)
def Rx(a): c, s = _cs(a); return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
def Ry(a): c, s = _cs(a); return torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)
def Rz(a): e = torch.exp(-0.5j * a.to(C)); z = torch.zeros_like(e); return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)
GATES = {'Rx': Rx, 'Ry': Ry, 'Rz': Rz}

# ---------- DLA via Pauli strings ----------
def gate_list(e, ts):
    ins, wire, n, gates = e.split('->')[0].split(','), {}, 0, []
    for idx, t in zip(ins, ts):
        op = t['op_type']
        if op == '0': wire[idx] = n; n += 1
        elif op == '0_dag': continue
        elif len(idx) == 2: q = wire[idx[0]]; wire[idx[1]] = q; gates.append((op, (q,), t['name']))
        elif len(idx) == 4:
            c, q = wire[idx[0]], wire[idx[1]]; wire[idx[2]], wire[idx[3]] = c, q; gates.append((op, (c, q), t['name']))
        else: raise ValueError(f'gate {op} on {idx}')
    return n, gates

def pauli_generators(n, gates):
    gens, fixed = [], []
    for op, qs, name in gates:
        if name is None: fixed.append((op, qs)); continue
        if op not in ('Rx', 'Ry', 'Rz'): return None          # controlled rotations are not single Pauli strings
        x, z, q = [0] * n, [0] * n, qs[0]
        x[q], z[q] = int(op in ('Rx', 'Ry')), int(op in ('Rz', 'Ry'))
        for fop, fq in reversed(fixed):                       # F^+ P F, latest fixed gate first; signs don't affect span
            if fop == 'H': a = fq[0]; x[a], z[a] = z[a], x[a]
            elif fop == 'CX': c, t = fq; x[t] ^= x[c]; z[c] ^= z[t]
            else: return None
        gens.append((sum(b << i for i, b in enumerate(x)), sum(b << i for i, b in enumerate(z))))
    return gens

def dla_dim(n, gens):
    mask = (1 << n) - 1
    gx = np.array([g[0] for g in gens], np.int64); gz = np.array([g[1] for g in gens], np.int64)
    seen = np.zeros(4 ** n, bool); k = np.unique((gx << n) | gz); seen[k] = True
    while len(k):
        fx, fz = k >> n, k & mask; new = []
        for a, b in zip(gx, gz):
            anti = (np.bitwise_count((fx & b) ^ (fz & a)) & 1).astype(bool)
            new.append(((fx[anti] ^ a) << n) | (fz[anti] ^ b))
        k = np.unique(np.concatenate(new)); k = k[~seen[k]]; seen[k] = True
    return int(seen.sum())

def box_gates(N, L):   # CustomV5 on N wires, as in ansatz.py
    g = [('H', (i,), None) for i in range(N)]
    for l in range(L):
        g += [(op, (i,), f'l{l}_{i}_{j}') for i in range(N) for j, op in enumerate(('Rz', 'Ry', 'Rz'))]
        if N > 1: g += [('CX', (i, (i + 1) % N), None) for i in range(N)]
    return g

# ---------- single-sentence amplitude as a function of the angles ----------
def amp(tm, e, ts, th):
    plan = tm._plan(e, ts); ops = [None] * plan['n_ops']
    for pos, b in plan['static']: buf = getattr(tm, b).to(C); ops[pos] = buf.expand(1, *buf.shape)
    pidx = torch.tensor([tm.sym2param.get(ts[p]['name'], tm.unk_param_index) for p in plan['param_pos']], device=DEV)
    ang = th[pidx][None]
    for op, cols in plan['by_type'].items():
        g = GATES[op](ang[:, cols])
        for j, c in enumerate(cols): ops[plan['param_pos'][c]] = g[:, j]
    return plan['expr'](*ops, backend='torch').reshape(2 ** tm.out_q, -1), pidx

def pure_qfi(psi, J):   # psi (D,), J (D, P): 4 Re(J^+J - J^+psi psi^+J)
    a = J.conj().T @ psi
    return 4 * (J.conj().T @ J - torch.outer(a, a.conj())).real

def sld_qfi(rho, dr):   # rho (d,d), dr (P,d,d): 2 sum_{ab} Re(<a|dr_i|b><b|dr_j|a>) / (l_a + l_b)
    lam, V = torch.linalg.eigh(rho.cpu()); lam, V = lam.to(DEV), V.to(DEV)
    D = torch.einsum('ai,pij,jb->pab', V.conj().T, dr, V)
    w = lam[:, None] + lam[None, :]; w = torch.where(w > 1e-12, 2 / w.clamp(min=1e-12), torch.zeros_like(w))
    return torch.einsum('iab,jba,ab->ij', D, D, w).real

def rank(ev, rel=1e-9):
    ev = ev.clamp(min=0); return int((ev > rel * ev.max().clamp(min=1e-300)).sum())

P1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
def pauli_all(q):
    B = []
    for t in itertools.product(range(4), repeat=q):
        m = np.array([[1.0]])
        for i in t: m = np.kron(m, P1[i])
        B.append(m)
    return torch.tensor(np.stack(B), dtype=C, device=DEV)

def magic(rho, PB):    # mixed SRE; for a pure state this is M2
    t = torch.einsum('ij,kji->k', rho, PB).real
    return float(-torch.log2((t ** 4).sum() / (t ** 2).sum()))

def load(cfg, pt):
    ans, img, txt, _ = build_experiment(cfg, DEV)
    de = DataEngine(cfg, ans, DEV)
    ctr, cva = de.compile_text('train'), de.compile_text('val')
    de.text_init(txt, pd.concat([ctr, cva], ignore_index=True)); de.image_init(img)
    meta = CheckpointManager.load_model(pt, img, txt, DEV) if pt else {}
    txt.to(DEV).eval()
    return txt, cva, meta

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
for arg in sys.argv[2:]:
    name = 'init' if arg == 'init' else Path(arg).name
    out = out_dir / f'{name}.json'
    if out.exists(): print('exists', out); continue
    ck = None if arg == 'init' else Path(arg)
    cfg = yaml.safe_load(open(REF_CFG if ck is None else ck / 'config.yaml')); set_seed(cfg.get('seed', 0))
    trained_disc = bool(cfg['text'].get('discard')) if ck else None
    cfg['text']['discard'] = True; cfg['loss'] = 'hs'; cfg['text']['postselect_types'] = []
    txt, cva, meta = load(cfg, ck / 'best.pt' if ck else None)
    q = cfg['embedding_qubits']; d = 2 ** q; PB = pauli_all(q)
    th_all = (torch.tanh(torch.cat([p.detach() for p in txt.params])) * 2 * torch.pi).to(DEV, R)
    sents = cva[['sentence', 'corrected_sentence_einsum', 'corrected_sentence_symbols']] if 'sentence' in cva else \
        cva[['corrected_sentence_einsum', 'corrected_sentence_symbols']].assign(sentence=cva.index.astype(str))
    sents = sents.assign(key=sents['corrected_sentence_symbols'].map(lambda s: str([t['name'] for t in s])))
    sents = sents.drop_duplicates('key').sample(n=min(N_SENT, sents['key'].nunique()), random_state=0)
    rec = dict(name=name, ckpt=str(ck), trained_discard=trained_disc, seed=cfg.get('seed'), layers=cfg['text']['layers'],
               epoch=meta.get('epoch') if isinstance(meta, dict) else None, sentences=[])
    rows_ps, rows_tr, cols_used = [], [], set()
    for e, ts, sent in zip(sents['corrected_sentence_einsum'], sents['corrected_sentence_symbols'], sents['sentence']):
        with torch.no_grad(): Psi, pidx = amp(txt, e, ts, th_all)
        u = torch.unique(pidx[pidx != txt.unk_param_index]); P = len(u); cols_used.update(u.tolist())
        def f(sub): return torch.view_as_real(amp(txt, e, ts, th_all.index_put((u,), sub))[0])
        Jr = torch.autograd.functional.jacobian(f, th_all[u].clone())          # (d, D, 2, P)
        J = torch.complex(Jr[:, :, 0], Jr[:, :, 1])                             # dPsi/dtheta (d, D, P)
        Dm = Psi.shape[1]; n = q + int(round(np.log2(Dm)))
        ev_full = torch.linalg.eigvalsh(pure_qfi(Psi.reshape(-1), J.reshape(-1, P)).cpu())
        psi0, J0 = Psi[:, 0], J[:, 0]; p0 = float((psi0.abs() ** 2).sum()); phi = psi0 / p0 ** 0.5
        Jphi = J0 / p0 ** 0.5 - torch.outer(phi, (psi0.conj() @ J0).real) / p0
        ev_ps = torch.linalg.eigvalsh(pure_qfi(phi, Jphi).cpu())
        rho = Psi @ Psi.conj().T
        dr = torch.einsum('sep,te->pst', J, Psi.conj()); dr = dr + dr.conj().transpose(1, 2)
        ev_tr = torch.linalg.eigvalsh(sld_qfi(rho, dr).cpu())
        dphi = torch.einsum('sp,t->pst', Jphi, phi.conj()); dphi = dphi + dphi.conj().transpose(1, 2)
        for rows, dm in ((rows_ps, dphi), (rows_tr, dr)):
            rows.append((u, torch.cat([dm.real.reshape(P, -1), dm.imag.reshape(P, -1)], 1).T))   # (2 d^2, P)
        pur = float((rho @ rho).trace().real)
        gens = pauli_generators(*gate_list(e, ts))
        rec['sentences'].append(dict(
            sentence=str(sent), n=n, m=n - q, P=P, n_param_gates=sum(t['name'] is not None for t in ts),
            rank_full=rank(ev_full), rank_ps=rank(ev_ps), rank_tr=rank(ev_tr),
            max_full=min(P, 2 ** (n + 1) - 2), max_ps=min(P, 2 * d - 2), max_tr=min(P, d * d - 1),
            qfi_full=ev_full.tolist(), qfi_ps=ev_ps.tolist(), qfi_tr=ev_tr.tolist(),
            p0=p0, purity=pur, S2=-np.log2(pur), M2_ps=magic(torch.outer(phi, phi.conj()), PB), M2_tr=magic(rho, PB),
            n_gens=None if gens is None else len(gens), struct=e))
    # dataset level: rank of the stacked readout Jacobians over all sampled sentences
    cols = torch.tensor(sorted(cols_used), device=DEV)
    for tag, rows in (('ps', rows_ps), ('tr', rows_tr)):
        pos = torch.full((len(th_all),), -1, dtype=torch.long, device=DEV); pos[cols] = torch.arange(len(cols), device=DEV)
        A = torch.zeros(sum(b.shape[0] for _, b in rows), len(cols), dtype=R, device=DEV); r0 = 0
        for uu, b in rows: A[r0:r0 + b.shape[0], pos[uu]] = b; r0 += b.shape[0]
        ev = torch.linalg.eigvalsh((A.T @ A).cpu()).flip(0).clamp(min=0)   # CUDA linalg is broken in this env: Gram on GPU, eigh on CPU
        rec[f'dataset_{tag}'] = dict(n_params=len(cols), n_rows=A.shape[0], rank=rank(ev), sv=ev.sqrt().tolist())
    json.dump(rec, open(out, 'w')); s = pd.DataFrame(rec['sentences'])
    print(name, 'n', s.n.median(), 'P', s.P.median(), 'rank full/ps/tr', s.rank_full.median(), s.rank_ps.median(), s.rank_tr.median(),
          'S2', round(s.S2.median(), 3), 'M2 ps/tr', round(s.M2_ps.median(), 3), round(s.M2_tr.median(), 3),
          'dataset rank ps/tr', rec['dataset_ps']['rank'], rec['dataset_tr']['rank'], '/', len(cols), flush=True)

# ---------- Haar references: 3-qubit pure states, and (3+m)-qubit pure states traced to 3 qubits ----------
rout = out_dir / 'reference.json'
if not rout.exists():
    g = torch.Generator(device='cpu').manual_seed(0); PB = pauli_all(3); ref = {}
    for m in (0, 2, 4, 6):
        v = torch.randn(2000, 8, 2 ** m, 2, generator=g, dtype=R); v = torch.complex(v[..., 0], v[..., 1]).to(DEV)
        v = v / v.flatten(1).norm(dim=1)[:, None, None]; rh = v @ v.conj().transpose(1, 2)
        pur = (rh @ rh).diagonal(dim1=1, dim2=2).sum(-1).real
        ref[f'm{m}'] = dict(S2=(-torch.log2(pur)).tolist(), M2=[magic(r, PB) for r in rh])
        print('haar m', m, 'S2 median', float((-torch.log2(pur)).median()), 'M2 median', float(np.median(ref[f'm{m}']['M2'])), flush=True)
    json.dump(ref, open(rout, 'w'))

# ---------- DLA dimensions: per circuit structure in the sampled sentences, and isolated word boxes ----------
dout = out_dir / 'dla.json'
if not dout.exists():
    res = {'boxes': [], 'structures': []}
    for N in range(1, 8):
        for L in (1, 2, 3):
            g = pauli_generators(N, box_gates(N, L)); res['boxes'].append(dict(N=N, L=L, params=len(g), dim=dla_dim(N, g), full=4 ** N - 1))
            print('box', res['boxes'][-1], flush=True)
    cfg = yaml.safe_load(open(REF_CFG)); cfg['text']['discard'] = True; cfg['loss'] = 'hs'
    ans, _, _, _ = build_experiment(cfg, DEV); de = DataEngine(cfg, ans, DEV); cva = de.compile_text('val')
    by_struct = {}
    for e, ts in zip(cva['corrected_sentence_einsum'], cva['corrected_sentence_symbols']): by_struct.setdefault(e, ts)
    for e, ts in by_struct.items():
        n, gl = gate_list(e, ts); g = pauli_generators(n, gl)
        if g is None or n > 13: res['structures'].append(dict(n=n, params=len(gl), dim=None)); continue
        res['structures'].append(dict(n=n, m=n - 3, params=len(g), dim=dla_dim(n, g), full=4 ** n - 1,
                                      count=int((cva['corrected_sentence_einsum'] == e).sum())))
        print('struct', res['structures'][-1], flush=True)
    json.dump(res, open(dout, 'w'))
