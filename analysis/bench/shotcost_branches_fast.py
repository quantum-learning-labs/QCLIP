# Shot-cost E4 (exact branch eval), fast engines. Same inputs/outputs as shotcost_branches.py, but only two branch bins are kept:
#   bin 0 = aux outcome 0...0 (post-selection readout), bin 1 = all other aux outcomes together (discard = bin0 + bin1).
# Every downstream consumer (viz_sc4.py, viz_data.py) reads only [..., 0] and sums over the last axis, so nothing else is needed.
# Engines (what is computed per caption circuit: phi0 = <0_aux|Psi> on the output wires, rho = Tr_aux |Psi><Psi|):
#   sv   Qiskit Statevector on the CPU (full 2^nq state; the reference implementation, grouped into the two bins)
#   tn   tensor-network contraction (opt_einsum) of the circuit and its conjugate: never builds 2^nq
#   gpu  Aer statevector simulator on the GPU (cuStateVec), then the same two-bin reduction
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/shotcost_branches_fast.py --engine tn -o out_dir [--e0_out dir]
#        [--sets std,swap_full] [--min_nq 21] [-l N] -cp ckpt...
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import opt_einsum as oe
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager

SWAP_SETS = {'swap_full': 'data/svo-probes/processed/swap_full.pkl', 'swap_animate': 'data/svo-probes/processed/swap_animate.pkl'}
K = 32
SKIP = {'barrier', 'measure', 'delay'}


# ---------------------------------------------------------------- engines: (circuit, out qubits) -> (phi0 [d], rho [d, d], k)
def _split_state(psi, n, oq):
    """psi: flat 2^n state (qiskit little-endian). Returns phi0 (aux = 0) and rho = sum_aux phi phi^dag, row index = sum_i bit(oq[i]) 2^i."""
    t = psi.reshape([2] * n)                               # axis j <-> qubit n-1-j
    aux = [q for q in range(n) if q not in oq]
    order = [n - 1 - q for q in reversed(oq)] + [n - 1 - q for q in reversed(aux)]
    t = np.transpose(t, order); d = 2 ** len(oq)
    phi0 = np.ascontiguousarray(t[(slice(None),) * len(oq) + (0,) * len(aux)]).reshape(d)
    m = t.reshape(d, -1)
    # chunk over the aux columns: avoids a second full-size copy for the 2^30 cases
    rho = np.zeros((d, d), complex); step = max(1, (1 << 24) // d)
    for s in range(0, m.shape[1], step):
        c = np.asarray(m[:, s:s + step], dtype=complex); rho += c @ c.conj().T
    return phi0, rho, len(aux)


def eng_sv(qc, oq, **kw):
    return _split_state(Statevector(qc).data, qc.num_qubits, oq)


_AER = {}
def eng_gpu(qc, oq, **kw):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    if 'sim' not in _AER:
        _AER['sim'] = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True, precision='single' if kw.get('single') else 'double')
    c = qc.copy(); c.save_statevector()
    c = transpile(c, _AER['sim'], optimization_level=0)
    psi = np.asarray(_AER['sim'].run(c).result().get_statevector(c).data)
    return _split_state(psi, qc.num_qubits, oq)


_PATHS = {}
def eng_tn(qc, oq, **kw):
    n = qc.num_qubits; sym = iter(range(10 ** 6)); S = lambda: oe.get_symbol(next(sym))
    cur = [S() for _ in range(n)]; init = list(cur)
    subs, ops = [], []
    e0 = np.array([1.0, 0.0], complex)
    for q in range(n): subs.append(cur[q]); ops.append(e0)
    for ci in qc.data:
        op = ci.operation
        if op.name in SKIP: continue
        qs = [qc.find_bit(q).index for q in ci.qubits]; k = len(qs)
        M = Operator(op).data.reshape((2,) * (2 * k))      # axes: out [q_{k-1}..q_0], in [q_{k-1}..q_0]
        new = [S() for _ in qs]
        subs.append(''.join(new[j] for j in reversed(range(k))) + ''.join(cur[qs[j]] for j in reversed(range(k))))
        ops.append(M)
        for j, q in enumerate(qs): cur[q] = new[j]
    aux = [q for q in range(n) if q not in oq]
    out_ket = ''.join(cur[q] for q in reversed(oq))
    # ---- phi0: ket network with aux wires projected onto |0>
    s0 = ','.join(subs + [cur[q] for q in aux]) + '->' + out_ket
    o0 = ops + [e0] * len(aux)
    # ---- rho: ket and bra networks; aux final labels shared (traced), everything else renamed in the bra copy
    keep = {cur[q] for q in aux}; ren = {}
    def r(l):
        if l in keep: return l
        if l not in ren: ren[l] = S()
        return ren[l]
    bra_subs = [''.join(r(c) for c in s) for s in subs]
    out_bra = ''.join(r(cur[q]) for q in reversed(oq))
    s1 = ','.join(subs + bra_subs) + '->' + out_ket + out_bra
    o1 = ops + [o.conj() for o in ops]
    d = 2 ** len(oq); res = []
    for s, o in ((s0, o0), (s1, o1)):
        key = (s, tuple(x.shape for x in o))
        if key not in _PATHS: _PATHS[key] = oe.contract_path(s, *o, optimize=kw.get('opt', 'greedy'))[0]
        res.append(oe.contract(s, *o, optimize=_PATHS[key]))
    return res[0].reshape(d), res[1].reshape(d, d), len(aux)


ENGINES = {'sv': eng_sv, 'tn': eng_tn, 'gpu': eng_gpu}


# ---------------------------------------------------------------- driver
def capture(ckpt, s, limit, seed, trace_mode=None):
    config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=seed)
    for sn, path in SWAP_SETS.items():
        if sn == s and sn not in config['dataset']['test']:
            config['dataset']['test'][sn] = dict(config['dataset']['test']['swap'], text_path=path)
    if trace_mode: config['text']['trace_mode'] = trace_mode
    discard = config['text'].get('discard', False)
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    c_train, c_val = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([c_train, c_val], ignore_index=True))
    de.image_init(image_model)
    ck = CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    image_model.eval(); text_model.eval()
    config['emulate'].update(set=s, measurement_method='compute_uncompute', noise=False)
    c_eval = c_val if s == 'val' else de.compile_text(s)
    qlimit = de.describe_einsum(text_model, c_eval, return_metrics=True)['max'][0]
    bm = BackendManager(config, qlimit); bm.transpile = lambda circs, **kw: circs
    cap, cur = [], [None]; ds = de.get_dataset(c_eval, s)
    class Track:
        def __len__(self): return len(ds)
        def __getitem__(self, i): cur[0] = i; return ds[i]
    def cu(qt, qi, oq):
        if len(oq) == 0: raise ValueError('empty output register')
        cap.append((cur[0], bm._strip_clbits(qt), bm._strip_clbits(qi), list(oq))); return QuantumCircuit(1)
    bm._compute_uncompute = cu
    bm.compile_circuits(Track(), text_model, image_model, limit)
    by_idx = {}
    for c in cap: by_idx.setdefault(c[0], []).append(c[1:])
    rows = [i for i in sorted(by_idx) if len(by_idx[i]) == 2]
    return config, discard, ck, [by_idx[i] for i in rows], rows, len(ds)


def run_one(ckpt, sets, out, limit, seed, engine, e0_out=None, min_nq=0, opt='greedy', single=False):
    name = ckpt.parent.name; eng = ENGINES[engine]
    for s in sets:
        f = out / f'{name}__{s}.npz'
        if f.exists(): print('exists, skip', f, flush=True); continue
        t0 = time.time()
        config, discard, ck, frames, rows, nds = capture(ckpt, s, limit, seed)
        if min_nq:
            keep = [i for i, fr in enumerate(frames) if max(fr[0][0].num_qubits, fr[1][0].num_qubits) >= min_nq]
            frames = [frames[i] for i in keep]; rows = [rows[i] for i in keep]
        n = len(rows); image_choice = s in ('val', 'std')
        print(f'{name} {s}: {n} pairs (capture {time.time() - t0:.0f}s) engine={engine}', flush=True)
        P, U, G, Q = [np.zeros((n, 2, K + 1)) for _ in range(2)] + [np.zeros((n, K + 1)) for _ in range(2)]; KK = np.zeros((n, 2), int)
        t1 = time.time(); kw = dict(opt=opt, single=single)
        for i in range(n):
            (tp, ip, op), (tn_, iN, on) = frames[i]
            vp, vn = Statevector(ip).data, Statevector(iN).data
            if image_choice and i < 3: assert op == on
            shared = eng(tp, op, **kw) if image_choice else None
            for j, (qt, oq, v) in enumerate(((tp, op, vp), (tn_, on, vn))):
                phi0, rho, k = shared if shared is not None else eng(qt, oq, **kw)
                p0 = float(np.vdot(phi0, phi0).real); ptot = float(np.trace(rho).real)
                u0 = abs(np.vdot(v, phi0)) ** 2; utot = float((v.conj() @ rho @ v).real)
                P[i, j, 0], P[i, j, 1] = p0, ptot - p0; U[i, j, 0], U[i, j, 1] = u0, utot - u0; KK[i, j] = k
            if image_choice:
                phi0, rho, _ = shared
                a, b, c = vp.conj() @ phi0, vn.conj() @ phi0, vp.conj() @ vn
                rpp, rnn, rnp = float((vp.conj() @ rho @ vp).real), float((vn.conj() @ rho @ vn).real), vn.conj() @ rho @ vp
                g0 = abs(a) ** 2 - abs(b) ** 2; q0 = abs(a) ** 2 + abs(b) ** 2 - 2 * np.real(np.conj(a) * c * b)
                G[i, 0], G[i, 1] = g0, (rpp - rnn) - g0
                Q[i, 0], Q[i, 1] = q0, (rpp + rnn - 2 * np.real(c * rnp)) - q0
            if (i + 1) % 200 == 0: print(f'  {i + 1}/{n} {time.time() - t1:.0f}s', flush=True)
        wall = time.time() - t1
        idx = np.array(rows)
        np.savez(f, P=P, U=U, G=G, Q=Q, k=KK, idx=idx)
        m = U.sum(-1) if discard else U[:, :, 0] / np.maximum(P[:, :, 0], 1e-300)
        meta = dict(checkpoint=str(ckpt), epoch=ck.get('epoch'), discard=discard, set=s, pairs=n, engine=engine, bins='h0_and_rest', min_nq=min_nq,
                    exact_acc=float(np.mean((m[:, 0] > m[:, 1]) + 0.5 * (m[:, 0] == m[:, 1]))), k_mean=float(KK.mean()),
                    wall_s=wall, capture_s=t1 - t0, git=git_state())
        json.dump(meta, open(f.with_suffix('.json'), 'w'), indent=1, default=str)
        if e0_out is not None and s != 'val':
            a = P[:, :, 0] if not discard else np.ones_like(m)
            nq = KK + np.array([[len(frames[i][j][2]) for j in range(2)] for i in range(n)])
            np.savez(e0_out / f'{name}__{s}.npz', m_pos=m[:, 0], m_neg=m[:, 1], a_pos=a[:, 0], a_neg=a[:, 1], k_pos=KK[:, 0], k_neg=KK[:, 1], nq_pos=nq[:, 0], nq_neg=nq[:, 1], idx=idx)
            json.dump({**{k: meta[k] for k in ('discard', 'set', 'pairs', 'engine', 'exact_acc')}, 'checkpoint': str(ckpt), 'accept_mean': float(a.mean()),
                       'k_mean': float(KK.mean()), 'embedding_qubits': config['embedding_qubits'], 'text_n': config['text'].get('n'), 'derived_from': str(f)},
                      open(e0_out / f'{name}__{s}.json', 'w'), indent=1, default=str)
        print(name, s, {x: meta[x] for x in ('pairs', 'exact_acc', 'wall_s')}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    p.add_argument('--engine', choices=list(ENGINES), required=True)
    p.add_argument('--sets', default='std,swap_full'); p.add_argument('-l', '--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=0); p.add_argument('-o', '--out', required=True); p.add_argument('--e0_out', default=None)
    p.add_argument('--min_nq', type=int, default=0, help='only pairs with a circuit of at least this many qubits (engine tests on the heavy tail)')
    p.add_argument('--opt', default='greedy'); p.add_argument('--single', action='store_true', help='gpu engine: single precision')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    e0 = Path(a.e0_out) if a.e0_out else None
    if e0: e0.mkdir(parents=True, exist_ok=True)
    for c in a.checkpoints: run_one(Path(c), a.sets.split(','), out, a.limit, a.seed, a.engine, e0, a.min_nq, a.opt, a.single)
    print('done', flush=True)


if __name__ == '__main__':
    main()
