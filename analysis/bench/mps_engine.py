# E4: approximate classical inference with a bond-dimension-limited MPS on the GPU, against the exact tensor-network result.
# MPS details: qubits laid on the chain in reverse Cuthill-McKee order of the 2q-gate interaction graph (short-range gates); 1q gates folded
# into the next 2q gate (inf_engines.gate_list); a 2q gate on distant sites first SWAPs the qubits together (the mapping is updated, no swap back);
# every gate is a contraction + truncated SVD (torch.linalg.svd, driver gesvd: the default cuSOLVER driver is only ~5e-4 accurate) to bond
# dimension chi, singular values absorbed to the right, with the orthogonality centre moved to the gate first (QR), so truncations are locally
# optimal. torch.linalg on CUDA needs the CUDA-12 cuSOLVER preloaded in our envs (LD_PRELOAD, see tools/sc5 job scripts): the cu11 copy pulled
# in by cuquantum overwrote it. (A cuTensorNet contract_decompose version spent ~30 ms of planning per gate: overhead, not the method.)
# Readout: phi0 (aux wires projected on |0>) for post-selection; rho (aux traced) for discard.
# Per (checkpoint, chi): NVML energy and time per decision, discarded weight, and against the exact result: relative amplitude error,
# relative margin error and decision agreement (margin = image-pair contrast, as in shotcost_branches_fast).
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/mps_engine.py -o out -cp ckpt... [--chis 4,8,16,32,64,128] [--pairs 20]
import argparse, csv, time
from pathlib import Path
import numpy as np, torch, networkx as nx
from qiskit.quantum_info import Statevector
from inf_engines import gate_list, TN, load

SWAP = torch.tensor(np.eye(4)[[0, 2, 1, 3]].reshape(2, 2, 2, 2), dtype=torch.complex64)


class MPS:
    def __init__(self, n, chi, dev, dtype=torch.complex64, order=None, cutoff=1e-7):
        self.chi, self.dev, self.dtype, self.cutoff = chi, dev, dtype, cutoff
        order = list(range(n)) if order is None else list(order)
        self.pos = {q: i for i, q in enumerate(order)}; self.qat = list(order)
        t = torch.zeros((1, 2, 1), dtype=dtype, device=dev); t[0, 0, 0] = 1
        self.T = [t.clone() for _ in range(n)]; self.c = 0; self.dw = 0.0
        self.swap = SWAP.to(dev, dtype)

    def _move_centre(self, p):
        while self.c < p:                                   # left QR: T[c] = Q, R into T[c+1]
            A = self.T[self.c]; Dl, _, Dr = A.shape
            Q, R = torch.linalg.qr(A.reshape(Dl * 2, Dr))
            self.T[self.c] = Q.reshape(Dl, 2, -1); self.T[self.c + 1] = torch.einsum('kj,jql->kql', R, self.T[self.c + 1]); self.c += 1
        while self.c > p:                                   # right QR: T[c] = R^H Q^H
            A = self.T[self.c]; Dl, _, Dr = A.shape
            Q, R = torch.linalg.qr(A.reshape(Dl, 2 * Dr).mH)
            self.T[self.c] = Q.mH.reshape(-1, 2, Dr); self.T[self.c - 1] = torch.einsum('iqj,jk->iqk', self.T[self.c - 1], R.mH); self.c -= 1

    def _apply2(self, p, G):
        """G (out_p, out_p+1, in_p, in_p+1) on sites p, p+1; truncated SVD (torch, accurate gesvd driver), S absorbed to the right."""
        self._move_centre(p)
        A, B = self.T[p], self.T[p + 1]; Dl, Dr = A.shape[0], B.shape[2]
        th = torch.einsum('ipj,jqk,rspq->irsk', A, B, G).reshape(Dl * 2, 2 * Dr)
        U, S, Vh = torch.linalg.svd(th, full_matrices=False, driver='gesvd')
        s2 = S * S; tot = s2.sum()
        keep = int(min(self.chi, int((s2 > self.cutoff * tot).sum().item()) or 1))
        self.dw += float((s2[keep:].sum() / tot).item())
        self.T[p] = U[:, :keep].reshape(Dl, 2, keep)
        self.T[p + 1] = (S[:keep, None].to(Vh.dtype) * Vh[:keep]).reshape(keep, 2, Dr); self.c = p + 1

    def gate(self, qs, M4):
        q0, q1 = qs                                         # M4 axes: out [q1, q0], in [q1, q0]
        while abs(self.pos[q0] - self.pos[q1]) > 1:         # bring q1 next to q0 with SWAPs (mapping updated, no swap back)
            p = self.pos[q1]; step = -1 if self.pos[q0] < p else 1
            a, b = (p + step, p) if step == -1 else (p, p + 1)
            self._apply2(a, self.swap)
            qa, qb = self.qat[a], self.qat[b]; self.qat[a], self.qat[b] = qb, qa; self.pos[qa], self.pos[qb] = b, a
        G = M4.permute(1, 0, 3, 2) if self.pos[q0] < self.pos[q1] else M4
        self._apply2(min(self.pos[q0], self.pos[q1]), G)

    def one(self, q, U):
        p = self.pos[q]; self.T[p] = torch.einsum('rp,ipj->irj', U, self.T[p])

    def phi0(self, oq, aux):
        env = torch.ones((1,), dtype=self.dtype, device=self.dev); opened = []
        e0 = torch.tensor([1, 0], dtype=self.dtype, device=self.dev)
        for p, q in enumerate(self.qat):
            if q in aux: env = torch.einsum('...a,apb,p->...b', env, self.T[p], e0)
            else: env = torch.einsum('...a,apb->...pb', env, self.T[p]); opened.append(q)
        env = env[..., 0]
        perm = [opened.index(q) for q in reversed(oq)]       # index sum_i bit(oq[i]) 2^i
        return env.permute(perm).reshape(-1)

    def rho(self, oq, aux):
        env = torch.ones((1, 1), dtype=self.dtype, device=self.dev); opened = []
        for p, q in enumerate(self.qat):
            A = self.T[p]
            if q in aux: env = torch.einsum('...ab,apc,bpd->...cd', env, A, A.conj())
            else:
                env = torch.einsum('...ab,apc,bqd->...pqcd', env, A, A.conj()); opened.append(q)
        env = env[..., 0, 0]                                # axes: (ket_1, bra_1, ket_2, bra_2, ...)
        e = len(opened); ket = [2 * opened.index(q) for q in reversed(oq)]; bra = [2 * opened.index(q) + 1 for q in reversed(oq)]
        return env.permute(ket + bra).reshape(2 ** e, 2 ** e)


def rcm_order(n, gates):
    G = nx.Graph(); G.add_nodes_from(range(n))
    for qs, _ in gates:
        if len(qs) == 2: G.add_edge(*qs)
    return list(nx.utils.reverse_cuthill_mckee_ordering(G))


class MPSRun:
    def __init__(self, qc, oq, chi, dev, rho=False):
        n = qc.num_qubits; self.n = n; self.aux = set(q for q in range(n) if q not in oq); self.oq = list(oq)
        gates, pend = gate_list(qc)
        self.order = rcm_order(n, gates); self.chi, self.dev, self.rho_mode = chi, dev, rho
        self.ops = [(qs, torch.tensor(M.reshape((2,) * 4), dtype=torch.complex64, device=dev)) for qs, M in gates]
        if any(len(qs) != 2 for qs, _ in gates): raise ValueError('only 1q/2q gates supported')
        self.pend = {q: torch.tensor(U, dtype=torch.complex64, device=dev) for q, U in pend.items()}

    def __call__(self):
        m = MPS(self.n, self.chi, self.dev, order=self.order)
        for qs, M4 in self.ops: m.gate(qs, M4)
        for q, U in self.pend.items(): m.one(q, U)
        self.dw = m.dw; self.maxD = max(t.shape[2] for t in m.T)
        return m.rho(self.oq, self.aux) if self.rho_mode else m.phi0(self.oq, self.aux)


def margin(x, vp, vn, rho):
    if rho: return float((vp.conj() @ x @ vp).real - (vn.conj() @ x @ vn).real)
    nrm = max(float(np.vdot(x, x).real), 1e-300)
    return float((abs(np.vdot(vp, x)) ** 2 - abs(np.vdot(vn, x)) ** 2) / nrm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--chis', default='2,4,8,16,32,64,128')
    ap.add_argument('--pairs', type=int, default=20)
    ap.add_argument('--nets', default='ps,rho')
    ap.add_argument('--min_s', type=float, default=5.0)
    ap.add_argument('--no_energy', action='store_true')
    ap.add_argument('--tn_width_max', type=float, default=29)
    ap.add_argument('--max_s_per_dec', type=float, default=120, help='stop raising chi once one decision takes longer than this')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = torch.device('cuda')
    w = csv.writer(open(out / 'mps.csv', 'a', newline=''))
    w.writerow(['gpu', 'ckpt', 'text_n', 'layers', 'nq', 'net', 'chi', 'decisions', 'J_gross_per_decision', 's_per_decision', 'idle_W',
                'dw_mean', 'maxD', 'amp_rel_err_mean', 'amp_rel_err_max', 'margin_rel_err_median', 'agree', 'reference'])
    m = None
    if not a.no_energy:
        from v3_energy import Meter
        m = Meter(); idle = m.idle_W(); print(f'GPU {m.name}, idle {idle:.1f} W', flush=True)
        if idle > 100: raise SystemExit('GPU not idle')
    else: idle = float('nan')
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers')
        print(f'{ckpt.parent.name}: {len(D)} decisions, {nq} qubits, layers {L}', flush=True)
        V = [(Statevector(qip).data, Statevector(qin).data) for _, qip, qin, _ in D]
        for net in a.nets.split(','):
            rho = net == 'rho'
            # exact reference (GPU TN, complex64) where its width allows
            ref = []
            for qt, _, _, oq in D:
                T = TN(qt, oq, dev, torch.complex64, rho=rho)
                ref.append(T().cpu().numpy() if T.width <= a.tn_width_max else None)
            ok = all(r is not None for r in ref); refkind = 'exact'
            chis = [int(x) for x in a.chis.split(',')]
            if not ok:                                       # exact too wide: the largest-chi MPS is the reference (convergence check)
                t0 = time.time(); ref = [MPSRun(qt, oq, max(chis), dev, rho)().cpu().numpy() for qt, _, _, oq in D]; ok = True; refkind = f'chi{max(chis)}'
                if (time.time() - t0) / len(D) > a.max_s_per_dec: chis = [c for c in chis if c < max(chis)]
            print(f'  {net}: reference {refkind}', flush=True)
            for chi in chis:
                t0 = time.time(); runs = [MPSRun(qt, oq, chi, dev, rho) for qt, _, _, oq in D]
                outs = [r().cpu().numpy() for r in runs]
                slow = (time.time() - t0) / len(D) > a.max_s_per_dec
                dw = float(np.mean([r.dw for r in runs])); maxD = max(r.maxD for r in runs)
                if ok:
                    rel = [np.linalg.norm(o - x) / max(np.linalg.norm(x), 1e-30) for o, x in zip(outs, ref)]
                    me = [margin(o, vp, vn, rho) for o, (vp, vn) in zip(outs, V)]; mx = [margin(x, vp, vn, rho) for x, (vp, vn) in zip(ref, V)]
                    mrel = float(np.median([abs(p - q) / max(abs(q), 1e-12) for p, q in zip(me, mx)])); agree = float(np.mean([(p > 0) == (q > 0) for p, q in zip(me, mx)]))
                    re_mean, re_max = float(np.mean(rel)), float(np.max(rel))
                else: re_mean = re_max = mrel = agree = float('nan')
                J = s = float('nan')
                if m is not None:
                    def fn():
                        for r in runs: r()
                        torch.cuda.synchronize()
                    J, s = m.run(fn, a.min_s); J /= len(runs); s /= len(runs)
                w.writerow([m.name if m else '', ckpt.parent.name, config['text'].get('n'), L, nq, net, chi, len(runs), J, s, idle, dw, maxD, re_mean, re_max, mrel, agree, refkind])
                print(f'    chi {chi}: {J:.3g} J/dec, {s:.3g} s/dec, dw {dw:.2e}, maxD {maxD}, amp err {re_mean:.2e}/{re_max:.2e}, margin rel err {mrel:.2e}, agree {agree:.3f}', flush=True)
                torch.cuda.empty_cache()
                if slow: print(f'    chi {chi} over {a.max_s_per_dec} s/decision: larger chi skipped', flush=True); break
    print('done', flush=True)


if __name__ == '__main__':
    main()
