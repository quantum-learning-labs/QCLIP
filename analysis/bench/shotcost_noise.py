# Shot-cost study E6: branch-resolved readout statistics under device noise (image choice, test std).
# Per pair, 3 circuits on the text register (all qubits measured): CU-pos, CU-neg (text, then image^dagger on the output qubits) and
# O (text, then a unitary W on the output qubits mapping the +/- eigenvectors of O = |vp><vp| - |vn><vn| to outcomes 0 / 1;
# outcome values +lam / -lam / 0, lam = sqrt(1 - |<vp|vn>|^2)). From the counts, per Hamming weight h of the aux outcome:
# P (branch prob), U (CU: P(out=0, h)), G/Q (O: first/second moment) -- the same fields as shotcost_branches.py, so
# shotcost_e4_report/family apply unchanged. Aer density matrix (exact noisy probabilities, sampled with --shots), pairs whose
# text register has <= --qmax qubits (DM size). 'ideal' = noiseless statevector of the same logical circuits (check vs E4).
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/shotcost_noise.py -cp ckpt -o out --profiles ideal,ibm_kingston,honest_miami -n 300
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import qiskit_ibm_runtime.fake_provider as fake_provider

from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager

NOISY = {'ibm_kingston': ('backend', 'FakeKingston'), 'honest_miami': ('honest', 'FakeMiami'), 'ibm_miami': ('backend', 'FakeMiami')}
K = 32


def o_unitary(vp, vn):
    O = np.outer(vp, vp.conj()) - np.outer(vn, vn.conj())
    w, V = np.linalg.eigh(O)                       # ascending: w[0] = -lam, w[-1] = +lam
    ep, em = V[:, -1], V[:, 0]
    rng = np.random.default_rng(0)
    M = np.column_stack([ep, em, rng.normal(size=(len(vp), len(vp) - 2)) + 1j * rng.normal(size=(len(vp), len(vp) - 2))])
    Qm, _ = np.linalg.qr(M)
    Qm[:, 0] *= np.vdot(Qm[:, 0], ep) / abs(np.vdot(Qm[:, 0], ep))   # keep columns 0/1 = ep/em up to phase
    Qm[:, 1] *= np.vdot(Qm[:, 1], em) / abs(np.vdot(Qm[:, 1], em))
    return Qm.conj().T, float(w[-1]), float(w[0])  # W = Q^dagger: W ep = |0>, W em = |1>


def stats(probs, out_q, n, vals=None):
    # probs: array over 2^n outcomes, index bit q = qubit q. Returns per-h arrays (P, U) or (G, Q) when vals given (value per out index).
    idx = np.arange(2 ** n)
    x = np.zeros_like(idx)
    for i, q in enumerate(out_q):
        x |= ((idx >> q) & 1) << i
    auxmask = sum(1 << q for q in range(n) if q not in out_q)
    h = np.array([bin(int(v) & auxmask).count('1') for v in idx])
    P = np.bincount(h, probs, minlength=K + 1)[:K + 1]
    if vals is None:
        return P, np.bincount(h, probs * (x == 0), minlength=K + 1)[:K + 1]
    o = vals[x]
    return np.bincount(h, probs * o, minlength=K + 1)[:K + 1], np.bincount(h, probs * o * o, minlength=K + 1)[:K + 1]


def counts_to_probs(cnt, n):
    p = np.zeros(2 ** n); tot = sum(cnt.values())
    for k, c in cnt.items():
        p[int(k.replace(' ', ''), 2)] += c / tot
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoint', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--profiles', default='ideal,ibm_kingston,honest_miami')
    ap.add_argument('-n', '--pairs', type=int, default=300)
    ap.add_argument('--qmax', type=int, default=12)
    ap.add_argument('--shots', type=int, default=2 ** 20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    ckpt = Path(a.checkpoint); name = ckpt.parent.name
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=a.seed)
    discard = config['text'].get('discard', False)
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    c_train, c_val = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([c_train, c_val], ignore_index=True))
    de.image_init(image_model)
    CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    image_model.eval(); text_model.eval()
    config['emulate'].update(set='std', measurement_method='compute_uncompute', noise=False)
    c_eval = de.compile_text('std')
    bm = BackendManager(config, a.qmax)                    # text circuits above qmax qubits are dropped by the compiler
    bm.transpile = lambda circs, **kw: circs
    cap, cur = [], [None]
    ds = de.get_dataset(c_eval, 'std')
    class Track:
        def __len__(self): return len(ds)
        def __getitem__(self, i): cur[0] = i; return ds[i]
    def cu(qt, qi, oq):
        if len(oq) == 0: raise ValueError('empty output register')
        cap.append((cur[0], bm._strip_clbits(qt), bm._strip_clbits(qi), list(oq)))
        return QuantumCircuit(1)
    bm._compute_uncompute = cu
    bm.compile_circuits(Track(), text_model, image_model, None)
    by = {}
    for c in cap: by.setdefault(c[0], []).append(c[1:])
    rows = [i for i in sorted(by) if len(by[i]) == 2][:a.pairs]
    print(f'{name}: {len(rows)} pairs with <= {a.qmax} text qubits (of {len(ds)} rows)', flush=True)
    circs, meta_c = [], []
    for i in rows:
        (qt, qip, oq), (_, qin, _) = by[i]
        n = qt.num_qubits
        vp, vn = Statevector(qip).data, Statevector(qin).data
        W, lp, lm = o_unitary(vp, vn)
        for kind, img in (('cu_pos', qip), ('cu_neg', qin), ('O', None)):
            qc = QuantumCircuit(n, n)
            qc.compose(qt, inplace=True)
            if img is not None: qc.compose(img.inverse(), qubits=oq, inplace=True)
            else: qc.unitary(W, oq, label='W_O')
            qc.measure(range(n), range(n))
            circs.append(qc)
        vals = np.zeros(2 ** len(oq)); vals[0], vals[1] = lp, lm
        meta_c.append((n, oq, vals))
    for prof in a.profiles.split(','):
        f = out / f'{name}__{prof}.npz'
        if f.exists(): print('exists, skip', f, flush=True); continue
        t0 = time.time()
        if prof == 'ideal':
            probs = []
            for qc in circs:
                sv = Statevector(qc.remove_final_measurements(inplace=False))
                probs.append(np.abs(sv.data) ** 2)
        else:
            profile, cls = NOISY[prof]
            cfg = copy.deepcopy(config)
            cfg['emulate'].update(noise=True, noise_profile=profile, backend='density_matrix',
                                  aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False})
            nb = BackendManager(cfg, a.qmax, hardware_profile=getattr(fake_provider, cls)())
            tc = nb.transpile(circs, batch_size=256, optimization_level=1)
            nb.backend.set_options(seed_simulator=1000 + a.seed)
            probs = []
            for j in range(0, len(tc), 128):
                res = nb.backend.run(tc[j:j + 128], shots=a.shots).result()
                for jj, qc in enumerate(circs[j:j + 128]):
                    probs.append(counts_to_probs(res.get_counts(jj), qc.num_clbits))
                print(f'{prof}: {j + len(tc[j:j + 128])}/{len(tc)} circuits, {time.time() - t0:.0f}s', flush=True)
        N = len(rows)
        P, U, G, Q, KK = np.zeros((N, 2, K + 1)), np.zeros((N, 2, K + 1)), np.zeros((N, K + 1)), np.zeros((N, K + 1)), np.zeros((N, 2), int)
        for r, (n, oq, vals) in enumerate(meta_c):
            for j in range(2):
                P[r, j], U[r, j] = stats(probs[3 * r + j], oq, n)
                KK[r, j] = n - len(oq)
            G[r], Q[r] = stats(probs[3 * r + 2], oq, n, vals)
        np.savez(f, P=P, U=U, G=G, Q=Q, k=KK, idx=np.array(rows))
        m = (U[:, :, 0] / np.maximum(P[:, :, 0], 1e-300)) if not discard else U.sum(-1)
        meta = {'checkpoint': str(ckpt), 'discard': discard, 'set': 'std', 'profile': prof, 'pairs': N, 'qmax': a.qmax, 'shots': a.shots,
                'text_n': config['text'].get('n'), 'git': git_state(), 'wall_s': time.time() - t0,
                'acc_cu': float(np.mean(m[:, 0] > m[:, 1])), 'acc_O_native': float(np.mean((G.sum(-1) if discard else G[:, 0]) > 0))}
        json.dump(meta, open(f.with_suffix('.json'), 'w'), indent=1, default=str)
        print(name, prof, {k: meta[k] for k in ('pairs', 'acc_cu', 'acc_O_native', 'wall_s')}, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
