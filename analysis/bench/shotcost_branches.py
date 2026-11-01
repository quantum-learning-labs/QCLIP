# Shot-cost study E4 (exact, no sampling): branch-resolved text states. For every pair, the text circuit's pure state |Psi> on
# (output, aux) qubits and the image states |v>. Per aux outcome b (grouped by Hamming weight h = 0..K), unnormalised
# phi_b = <b|_aux Psi and:
#   p_h  = sum_{|b|=h} |phi_b|^2                           (branch probability; PS acceptance a = p_0)
#   u_h  = sum_{|b|=h} |<v|phi_b>|^2  per circuit (pos/neg)  (compute-uncompute: P(out = v, branch h))
#   G_h  = sum_{|b|=h} <phi_b|O|phi_b>, Q_h = sum <phi_b|O^2|phi_b>  for image choice (std/val), O = |v_p><v_p| - |v_n><v_n|
#          (one circuit measuring O's eigenbasis: the minimum-variance unbiased single-shot readout of the margin).
# PS readout = weights w = (1,0,0,...) on h, discard = (1,1,...): both models can then be read out with any weights.
# Check: u_0/p_0 (PS) and sum_h u_h (discard) must equal shotcost.py's exact m.
# Usage: PYTHONPATH=.:analysis/bench python analysis/bench/shotcost_branches.py -o results/shotcost/e4 --sets val,std,swap_full -cp ...
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager

SWAP_SETS = {'swap_full': 'data/svo-probes/processed/swap_full.pkl', 'swap_animate': 'data/svo-probes/processed/swap_animate.pkl'}
K = 16   # max Hamming weight stored


def branches(qc_txt, out_q):
    n = qc_txt.num_qubits
    psi = Statevector(qc_txt).data.reshape([2] * n)          # axis j <-> qubit n-1-j
    aux = [q for q in range(n) if q not in out_q]
    order = [n - 1 - q for q in reversed(out_q)] + [n - 1 - q for q in reversed(aux)]
    phi = np.transpose(psi, order).reshape(2 ** len(out_q), 2 ** len(aux))   # row: sum_i bit(out_q[i]) 2^i ; col: aux bits
    hw = np.array([bin(c).count('1') for c in range(2 ** len(aux))])
    return phi, hw, len(aux)


def by_h(vals, hw):
    out = np.zeros(K + 1)
    np.add.at(out, hw, vals)
    return out


def run_one(ckpt, sets, out, limit, seed):
    name = ckpt.parent.name
    config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=seed)
    for s, path in SWAP_SETS.items():
        if s in sets and s not in config['dataset']['test']:
            config['dataset']['test'][s] = dict(config['dataset']['test']['swap'], text_path=path)
    discard = config['text'].get('discard', False)
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    c_train, c_val = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([c_train, c_val], ignore_index=True))
    de.image_init(image_model)
    ck = CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    image_model.eval(); text_model.eval()
    for s in sets:
        f = out / f'{name}__{s}.npz'
        if f.exists():
            print('exists, skip', f, flush=True); continue
        t = time.time()
        config['emulate'].update(set=s, measurement_method='compute_uncompute', noise=False)
        c_eval = c_val if s == 'val' else de.compile_text(s)
        qlimit = de.describe_einsum(text_model, c_eval, return_metrics=True)['max'][0]
        bm = BackendManager(config, qlimit)
        bm.transpile = lambda circs, **kw: circs
        cap = []
        bm._compute_uncompute = lambda qt, qi, oq: (cap.append((bm._strip_clbits(qt), bm._strip_clbits(qi), list(oq))), QuantumCircuit(1))[1]
        pos, _ = bm.compile_circuits(de.get_dataset(c_eval, s), text_model, image_model, limit)
        n = len(pos)
        assert len(cap) == 2 * n
        P, U, G, Q, KK = [np.zeros((n, 2, K + 1)) for _ in range(2)] + [np.zeros((n, K + 1)) for _ in range(2)] + [np.zeros((n, 2), int)]
        image_choice = s in ('val', 'std')
        for i in range(n):
            (tp, ip, op), (tn, iN, on) = cap[2 * i], cap[2 * i + 1]
            vp, vn = Statevector(ip).data, Statevector(iN).data
            for j, (qt, oq, v) in enumerate(((tp, op, vp), (tn, on, vn))):
                phi, hw, k = branches(qt, oq)
                P[i, j] = by_h((np.abs(phi) ** 2).sum(0), hw)
                U[i, j] = by_h(np.abs(v.conj() @ phi) ** 2, hw)
                KK[i, j] = k
            if image_choice:   # O = |vp><vp| - |vn><vn| on the shared text state (one caption, two images)
                if i < 3:
                    assert tp == tn and op == on, 'image choice: pos/neg text circuits differ'
                phi, hw, _ = branches(tp, op)
                a, b = vp.conj() @ phi, vn.conj() @ phi            # <vp|phi_b>, <vn|phi_b>
                c = vp.conj() @ vn
                G[i] = by_h(np.abs(a) ** 2 - np.abs(b) ** 2, hw)
                # <phi|O^2|phi> = |a|^2 + |b|^2 - 2 Re(conj(a) c b)   (O^2 = Pp + Pn - c|vp><vn| - c*|vn><vp|)
                Q[i] = by_h(np.abs(a) ** 2 + np.abs(b) ** 2 - 2 * np.real(np.conj(a) * c * b), hw)
        np.savez(f, P=P, U=U, G=G, Q=Q, k=KK)
        m = (U[:, :, 0] / np.maximum(P[:, :, 0], 1e-300)) if not discard else U.sum(-1)
        meta = {'checkpoint': str(ckpt), 'epoch': ck.get('epoch'), 'discard': discard, 'set': s, 'pairs': n,
                'text_n': config['text'].get('n'), 'git': git_state(), 'exact_acc': float(np.mean((m[:, 0] > m[:, 1]) + 0.5 * (m[:, 0] == m[:, 1]))),
                'image_choice': image_choice, 'k_mean': float(KK.mean()), 'wall_s': time.time() - t}
        json.dump(meta, open(f.with_suffix('.json'), 'w'), indent=1, default=str)
        print(name, s, {x: meta[x] for x in ('pairs', 'exact_acc', 'k_mean', 'wall_s')}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    p.add_argument('--sets', default='val,std,swap_full')
    p.add_argument('-l', '--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for c in a.checkpoints:
        run_one(Path(c), a.sets.split(','), out, a.limit, a.seed)
    print('done', flush=True)


if __name__ == '__main__':
    main()
