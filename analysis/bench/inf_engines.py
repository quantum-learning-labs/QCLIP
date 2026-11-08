# Inference energy per decision on one GPU for several classical engines, on the same typical circuits as v3_energy.py --nq_mode.
# A post-selection decision needs phi0 = <0_aux|Psi> on the output register (2^e amplitudes); a discard decision needs
# rho = Tr_aux |Psi><Psi| (2^e x 2^e). The margin against the candidate images is then a 2^e-dim dot product (negligible).
# Engines:
#   aer       page-v14 baseline: Aer GPU statevector of the O-readout circuit, output-register probabilities, no cuStateVec
#   cusv      Aer GPU statevector + cuStateVec on the text circuit, single precision, gate fusion; saves only phi0 (save_amplitudes)
#   cusv_rho  same, saves the output-register reduced density matrix (discard decision)
#   lazy      torch GPU statevector: a wire is allocated at its first gate and a post-selected wire is projected onto |0> right
#             after its last gate, so the state has 2^(live wires) entries, not 2^nq; gates are scheduled greedily to keep few wires
#             live; 1q gates are folded into the next 2q gate (or into the projection)
#   tn        opt_einsum contraction (torch backend on the GPU, complex64) of the ket network with the aux wires projected -> phi0
#   tn_rho    circuit + conjugate network, aux wires traced -> rho
# Every engine is checked against the CPU reference (shotcost_branches_fast.eng_tn, complex128) on the first --check decisions
# before it is timed; the max abs error goes into the CSV. --diag only writes the structure statistics (no GPU, no timing).
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/inf_engines.py -o out -cp ckpt... [--engines cusv,lazy,tn] [--pairs 40]
import argparse, csv, time
from collections import Counter
from pathlib import Path
import numpy as np, opt_einsum as oe, torch
from qiskit import transpile
from qiskit.quantum_info import Operator
from g_count import frames, build

SKIP = {'barrier', 'measure', 'delay'}
I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------- circuit -> gate list with 1q gates folded
def gate_list(qc):
    """[(qubits, matrix)] with every 1q gate folded into the next multi-qubit gate on its wire; leftover 1q gates returned per wire."""
    pend, out = {}, []
    for ci in qc.data:
        op = ci.operation
        if op.name in SKIP: continue
        qs = [qc.find_bit(q).index for q in ci.qubits]; M = Operator(op).data
        if len(qs) == 1: pend[qs[0]] = M @ pend.get(qs[0], I2); continue
        P = np.array([[1.0]], complex)
        for j in reversed(range(len(qs))): P = np.kron(P, pend.pop(qs[j], I2))   # big-endian over (q_{k-1}..q_0)
        out.append((qs, M @ P))
    return out, pend


def schedule(gates, n, aux):
    """Greedy topological order that keeps few wires live: prefer gates that close wires and open none. Returns (order, peak)."""
    queues = {q: [] for q in range(n)}
    for g, (qs, _) in enumerate(gates):
        for q in qs: queues[q].append(g)
    head = {q: 0 for q in range(n)}; live = set(); order = []; peak = 0; auxs = set(aux)
    remaining = {q: len(queues[q]) for q in range(n)}
    ready = lambda g: all(queues[q][head[q]] == g for q in gates[g][0])
    cand = {queues[q][0] for q in range(n) if queues[q]}
    while cand:
        best, bs = None, None
        for g in cand:
            if not ready(g): continue
            qs = gates[g][0]
            opens = sum(q not in live for q in qs); closes = sum(remaining[q] == 1 and q in auxs for q in qs)
            s = (opens - closes, opens, g)
            if bs is None or s < bs: best, bs = g, s
        g = best; order.append(g); cand.discard(g)
        for q in gates[g][0]:
            live.add(q); peak = max(peak, len(live)); head[q] += 1; remaining[q] -= 1
            if remaining[q] == 0:
                if q in auxs: live.discard(q)
            else: cand.add(queues[q][head[q]])
    return order, peak


def natural_peak(gates, n, aux):
    return schedule_peak(list(range(len(gates))), gates, n, aux)


def schedule_peak(order, gates, n, aux):
    remaining = Counter(q for qs, _ in gates for q in qs); live, peak = set(), 0
    for g in order:
        for q in gates[g][0]:
            live.add(q); peak = max(peak, len(live)); remaining[q] -= 1
            if remaining[q] == 0 and q in aux: live.discard(q)
    return peak


# ---------------------------------------------------------------- lazy statevector (torch)
class Lazy:
    def __init__(self, qc, oq, dev, dtype):
        n = qc.num_qubits; aux = [q for q in range(n) if q not in oq]
        gates, pend = gate_list(qc)
        order, self.peak = schedule(gates, n, aux)
        self.nat_peak = schedule_peak(list(range(len(gates))), gates, n, set(aux))
        remaining = Counter(q for qs, _ in gates for q in qs)
        prog, scal = [], 1.0 + 0j
        for q in aux:                                        # aux wires with no 2q gate: <0|U|0> scalar
            if remaining[q] == 0: scal *= pend.pop(q, I2)[0, 0]
        for g in order:
            qs, M = gates[g]
            prog.append(('g', qs, torch.tensor(M.reshape((2,) * (2 * len(qs))), dtype=dtype, device=dev)))
            for q in qs:
                remaining[q] -= 1
                if remaining[q] == 0 and q in aux:
                    prog.append(('p', q, torch.tensor(pend.pop(q, I2)[0], dtype=dtype, device=dev)))
        for q in oq:
            if q in pend: prog.append(('g', [q], torch.tensor(pend.pop(q), dtype=dtype, device=dev)))
        self.prog, self.scal, self.oq, self.dev, self.dtype = prog, scal, list(oq), dev, dtype

    def __call__(self):
        st = torch.ones((), dtype=self.dtype, device=self.dev) * self.scal; live = []
        def alloc(q):
            nonlocal st
            st = torch.stack([st, torch.zeros_like(st)], dim=-1); live.append(q)
        for kind, qs, M in self.prog:
            if kind == 'p':
                ax = live.index(qs); st = torch.tensordot(M, st, dims=([0], [ax])); live.pop(ax); continue
            for q in qs:
                if q not in live: alloc(q)
            k = len(qs); axes = [live.index(q) for q in reversed(qs)]
            st = torch.tensordot(M, st, dims=(list(range(k, 2 * k)), axes))
            live[:] = list(reversed(qs)) + [l for l in live if l not in qs]
        for q in self.oq:
            if q not in live: alloc(q)
        st = st.permute([live.index(q) for q in reversed(self.oq)])
        return st.reshape(-1)


# ---------------------------------------------------------------- tensor network (opt_einsum, torch backend)
class TN:
    def __init__(self, qc, oq, dev, dtype, rho=False, opt='greedy'):
        n = qc.num_qubits; sym = iter(range(10 ** 7)); S = lambda: oe.get_symbol(next(sym))
        gates, pend = gate_list(qc)
        cur = [S() for _ in range(n)]; subs, ops = [], []
        e0 = np.array([1.0, 0.0], complex)
        for q in range(n): subs.append(cur[q]); ops.append(e0)
        for qs, M in gates:
            k = len(qs); new = [S() for _ in qs]
            subs.append(''.join(new[j] for j in reversed(range(k))) + ''.join(cur[qs[j]] for j in reversed(range(k))))
            ops.append(M.reshape((2,) * (2 * k)))
            for j, q in enumerate(qs): cur[q] = new[j]
        for q, U in pend.items():
            new = S(); subs.append(new + cur[q]); ops.append(U); cur[q] = new
        aux = [q for q in range(n) if q not in oq]
        out_ket = ''.join(cur[q] for q in reversed(oq))
        if not rho:
            eq = ','.join(subs + [cur[q] for q in aux]) + '->' + out_ket; arrs = ops + [e0] * len(aux)
        else:
            keep = {cur[q] for q in aux}; ren = {}
            def r(l):
                if l in keep: return l
                if l not in ren: ren[l] = S()
                return ren[l]
            eq = ','.join(subs + [''.join(r(c) for c in s) for s in subs]) + '->' + out_ket + ''.join(r(cur[q]) for q in reversed(oq))
            arrs = ops + [o.conj() for o in ops]
        t0 = time.time()
        self.path, info = oe.contract_path(eq, *arrs, optimize=opt)
        self.path_s = time.time() - t0
        self.flops, self.width = float(info.opt_cost), float(np.log2(max(float(info.largest_intermediate), 1.0)))
        self.eq = eq; self.arrs = [torch.tensor(a, dtype=dtype, device=dev) for a in arrs]
        self.d = 2 ** len(oq); self.rho = rho

    def __call__(self):
        r = oe.contract(self.eq, *self.arrs, optimize=self.path)
        return r.reshape(self.d, self.d) if self.rho else r.reshape(self.d)


# ---------------------------------------------------------------- Aer engines
def aer_sim(cusv, precision, fusion):
    from qiskit_aer import AerSimulator
    kw = dict(method='statevector', device='GPU', precision=precision)
    if cusv: kw['cuStateVec_enable'] = True
    if fusion: kw['fusion_max_qubit'] = fusion
    return AerSimulator(**kw)


def amp_index(oq):
    return [sum(((j >> i) & 1) << q for i, q in enumerate(oq)) for j in range(2 ** len(oq))]


# ---------------------------------------------------------------- driver
def load(ckpt, a):
    config, qlimit, fr = frames(ckpt, 0)
    mode = Counter(x[1][0][0].num_qubits for x in fr).most_common(1)[0][0]
    fr = [x for x in fr if x[1][0][0].num_qubits == mode and x[1][1][0].num_qubits == mode][:a.pairs]
    return config, mode, [(qt, qip, qin, oq) for _, ((qt, qip, oq), (_, qin, _)) in fr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--engines', default='aer,cusv,cusv_rho,lazy,tn,tn_rho')
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--check', type=int, default=4, help='decisions checked against the CPU reference per engine')
    ap.add_argument('--check_rho_max', type=int, default=24, help='CPU rho reference only up to this many qubits (else cross-check cusv_rho vs tn_rho)')
    ap.add_argument('--fusion', type=int, default=0, help='Aer fusion_max_qubit (0 = Aer default)')
    ap.add_argument('--opt', default='greedy', help='opt_einsum path optimizer')
    ap.add_argument('--sv_max', type=int, default=31, help='skip full-statevector engines above this many qubits (24 GB GPU, single precision)')
    ap.add_argument('--live_max', type=int, default=31, help='skip lazy when its peak live width exceeds this')
    ap.add_argument('--tn_width_max', type=float, default=31, help='skip tn when log2(largest intermediate) exceeds this')
    ap.add_argument('--min_s', type=float, default=10.0)
    ap.add_argument('--diag', action='store_true', help='structure statistics only (CPU)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    engines = a.engines.split(',')
    from shotcost_branches_fast import eng_tn
    dw = csv.writer(open(out / 'structure.csv', 'a', newline='', buffering=1))
    dw.writerow(['ckpt', 'text_n', 'nq', 'e', 'decisions', 'n2q_mean', 'live_peak_nat', 'live_peak_sched', 'tn_width', 'tn_log10_flops', 'tn_rho_width', 'tn_rho_log10_flops', 'path_s'])
    if not a.diag:
        from v3_energy import Meter
        m = Meter(); idle = m.idle_W()
        print(f'GPU {m.name}, idle {idle:.1f} W', flush=True)
        if idle > 100: raise SystemExit(f'GPU is not idle ({idle:.0f} W): another process is on it; rerun on another node')
        dev = torch.device('cuda')
        ew = csv.writer(open(out / 'gpu_engines.csv', 'a', newline='', buffering=1))
        ew.writerow(['gpu', 'ckpt', 'text_n', 'nq', 'engine', 'decisions', 'J_gross', 'seconds', 'J_gross_per_decision', 'J_net_per_decision', 's_per_decision', 'idle_W', 'max_abs_err', 'note'])
    for c in a.checkpoints:
        ckpt = Path(c); t0 = time.time(); cross = None
        config, nq, D = load(ckpt, a); tn_ = config['text'].get('n'); e = len(D[0][3])
        print(f'{ckpt.parent.name}: {len(D)} typical decisions with {nq} qubits, e={e} (capture {time.time() - t0:.0f}s)', flush=True)
        # structure statistics on every decision (CPU)
        st = []
        for qt, _, _, oq in D:
            L = Lazy(qt, oq, 'cpu', torch.complex64)
            T0 = TN(qt, oq, 'cpu', torch.complex64, rho=False, opt=a.opt)
            T1 = TN(qt, oq, 'cpu', torch.complex64, rho=True, opt=a.opt)
            n2 = sum(1 for ci in qt.data if len(ci.qubits) > 1 and ci.operation.name not in SKIP)
            st.append((n2, L.nat_peak, L.peak, T0.width, np.log10(T0.flops), T1.width, np.log10(T1.flops), T0.path_s + T1.path_s))
        S = np.array(st)
        dw.writerow([ckpt.parent.name, tn_, nq, e, len(D), S[:, 0].mean(), int(S[:, 1].max()), int(S[:, 2].max()), S[:, 3].max(), S[:, 4].mean(), S[:, 5].max(), S[:, 6].mean(), S[:, 7].sum()])
        print(f'  structure: live peak natural {int(S[:, 1].max())} scheduled {int(S[:, 2].max())}; tn width {S[:, 3].max():.0f} (log10 flops {S[:, 4].mean():.1f}); '
              f'tn_rho width {S[:, 5].max():.0f} (log10 flops {S[:, 6].mean():.1f}); path search {S[:, 7].sum():.1f}s', flush=True)
        if a.diag: continue
        # CPU reference on the first --check decisions
        refs = []
        for qt, _, _, oq in D[:a.check]:
            if S[:, 3].max() > 26: refs.append((None, None))              # CPU reference too slow: error not checked (NaN)
            elif nq <= a.check_rho_max and S[:, 5].max() <= 26: phi0, rho, _ = eng_tn(qt, oq); refs.append((phi0, rho))
            else:
                T = TN(qt, oq, 'cpu', torch.complex128, rho=False, opt=a.opt); refs.append((T().numpy(), None))
        for eng in engines:
            note = ''
            if eng in ('aer', 'cusv', 'cusv_rho') and nq > a.sv_max: print(f'  {eng}: skip ({nq} > sv_max)', flush=True); continue
            if eng == 'lazy' and S[:, 2].max() > a.live_max: print(f'  {eng}: skip (live peak > live_max)', flush=True); continue
            if eng == 'tn' and S[:, 3].max() > a.tn_width_max: print(f'  {eng}: skip', flush=True); continue
            if eng == 'tn_rho' and S[:, 5].max() > a.tn_width_max: print(f'  {eng}: skip', flush=True); continue
            try:
                err = np.nan
                if eng in ('aer', 'cusv', 'cusv_rho'):
                    if eng == 'aer':
                        prec = 'double' if nq <= 30 else 'single'; sim = aer_sim(False, prec, 0); note = prec
                        circs = []
                        for qt, qip, qin, oq in D:
                            cs = build(qt, qip, qin, oq, 6); o = (cs[2] if cs[2] is not None else cs[0]).remove_final_measurements(inplace=False)
                            o.save_probabilities(list(oq)); circs.append(o)
                    else:
                        sim = aer_sim(True, 'single', a.fusion); note = f'single fusion={a.fusion or "default"}'
                        circs = []
                        for qt, _, _, oq in D:
                            o = qt.copy()
                            (o.save_amplitudes(amp_index(oq)) if eng == 'cusv' else o.save_density_matrix(qubits=list(oq))); circs.append(o)
                    tc = transpile(circs, sim, optimization_level=1, seed_transpiler=0)
                    res = sim.run(tc[:a.check]).result()
                    if eng == 'cusv' and refs[0][0] is not None:
                        err = max(np.abs(np.asarray(res.data(i)['amplitudes']) - refs[i][0]).max() for i in range(a.check))
                    elif eng == 'cusv_rho' and refs[0][1] is not None:
                        err = max(np.abs(np.asarray(res.data(i)['density_matrix']) - refs[i][1]).max() for i in range(a.check))
                    if eng == 'cusv_rho': cross = [np.asarray(res.data(i)['density_matrix']) for i in range(a.check)]
                    fn = lambda: sim.run(tc).result()
                else:
                    if eng == 'lazy': objs = [Lazy(qt, oq, dev, torch.complex64) for qt, _, _, oq in D]
                    else: objs = [TN(qt, oq, dev, torch.complex64, rho=(eng == 'tn_rho'), opt=a.opt) for qt, _, _, oq in D]
                    outs = [objs[i]().cpu().numpy() for i in range(a.check)]
                    if eng == 'tn_rho':
                        if refs[0][1] is not None: err = max(np.abs(outs[i] - refs[i][1]).max() for i in range(a.check))
                        elif cross is not None: err = max(np.abs(outs[i] - cross[i]).max() for i in range(a.check)); note = 'err vs cusv_rho'
                    elif refs[0][0] is not None: err = max(np.abs(outs[i] - refs[i][0]).max() for i in range(a.check))
                    def fn():
                        for o in objs: o()
                        torch.cuda.synchronize()
                fn()                                                    # warm-up
                J, s = m.run(fn, a.min_s)
                n_ = len(D)
                ew.writerow([m.name, ckpt.parent.name, tn_, nq, eng, n_, J, s, J / n_, (J - idle * s) / n_, s / n_, idle, err, note])
                print(f'  {eng}: {J / n_:.4g} J/decision gross, {s / n_:.4g} s/decision, max abs err {err:.2e} {note}', flush=True)
            except Exception as ex:
                print(f'  {eng}: FAILED {type(ex).__name__}: {str(ex)[:300]}', flush=True)
                ew.writerow([m.name, ckpt.parent.name, tn_, nq, eng, 0, '', '', '', '', '', idle, '', f'FAILED {type(ex).__name__}'])
            finally:
                torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
