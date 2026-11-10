# E5: QPU cost and estimated fidelity of one decision circuit at every size/depth (structure only, CPU).
# IBM: transpile the O-readout circuit to FakeKingston (same settings as v3_energy.qpu_stats) -> duration, native gate counts, and the
#      estimated success probability ESP = prod(1 - gate error) over gates and prod(1 - readout error) over measured qubits (calibration snapshot).
# IonQ: transpile to {rz, ry, rx, rxx} (all-to-all) -> native counts; ESP is computed later from assumed fidelities (viz side), so constants stay in one place.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/qpu_esp.py -o out.csv -cp ckpt... [--k 5]
import argparse, csv, time
from pathlib import Path
import numpy as np
from qiskit import transpile
import qiskit_ibm_runtime.fake_provider as fake_provider
from g_count import build
from inf_engines import load

IONQ_BASIS = ['rz', 'ry', 'rx', 'rxx']
TWO_Q = {'cx', 'cz', 'ecr', 'rzz', 'rxx'}
NOT_GATE = {'measure', 'barrier', 'delay'}


def esp_ibm(qc, target):
    lg = 0.0; lr = 0.0; miss = 0
    for ci in qc.data:
        name = ci.operation.name
        if name in ('barrier', 'delay'): continue
        qs = tuple(qc.find_bit(q).index for q in ci.qubits)
        try:
            e = target[name][qs].error
        except Exception:
            e = None
        if e is None: miss += 1; continue
        if name == 'measure': lr += np.log1p(-e)
        else: lg += np.log1p(-e)
    return float(np.exp(lg)), float(np.exp(lr)), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--circ', default='O', help="O: one circuit per decision (O-readout if e <= o_max, else CU-pos); all: cu_pos, cu_neg and O, labelled")
    ap.add_argument('--seeds', type=int, default=1, help='Kingston transpile seeds; keep the fewest two-qubit gates per circuit')
    ap.add_argument('--o_max', type=int, default=6, help='largest output register for the O-readout unitary (its synthesis grows ~4^e)')
    a = ap.parse_args()
    kingston = fake_provider.FakeKingston(); tg = kingston.target
    out = Path(a.out); new = not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'a', newline='', buffering=1))
    if new: w.writerow(['ckpt', 'text_n', 'layers', 'nq', 'idx'] + (['circ'] if a.circ == 'all' else []) + ['ibm_duration_s', 'ibm_n2q', 'ibm_n1q', 'ibm_esp_gates', 'ibm_esp_readout', 'ibm_missing_err', 'ionq_n1q', 'ionq_n2q'] + (['ibm_seed'] if a.seeds > 1 else []) + ['ibm_transpile_s', 'ionq_transpile_s'])   # seconds per circuit and seed (wall, one process)
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers'); t0 = time.time()
        circs, lab = [], []
        for j, (qt, qip, qin, oq) in enumerate(D[:a.k]):
            cs = build(qt, qip, qin, oq, a.o_max)
            for name, qc in zip(('cu_pos', 'cu_neg', 'O'), cs):
                if a.circ == 'O' and name != ('O' if cs[2] is not None else 'cu_pos'): continue   # old behaviour: O-readout, else CU-pos
                if qc is not None: circs.append(qc); lab.append((j, name))
        # Kingston layout/routing is randomised: transpile with --seeds seeds and keep, per circuit, the one with fewest two-qubit gates
        # (what a user would run); one seed reproduces the earlier counts
        n2 = lambda qc: sum(v for k, v in qc.count_ops().items() if k in TWO_Q)
        ibm, bseed = None, None; tib = [0.0] * len(circs); tio = []
        for sd in range(a.seeds):
            t = []
            for i, qc in enumerate(circs):          # one circuit at a time, timed: the one-off compile cost of a circuit shape (same result as the batch call)
                t1 = time.perf_counter(); t.append(transpile(qc, backend=kingston, optimization_level=1, seed_transpiler=sd)); tib[i] += (time.perf_counter() - t1) / a.seeds
            if ibm is None: ibm, bseed = t, [sd] * len(t); continue
            for i, q in enumerate(t):
                if n2(q) < n2(ibm[i]): ibm[i], bseed[i] = q, sd
        ion = []
        for qc in circs:
            t1 = time.perf_counter(); ion.append(transpile(qc, basis_gates=IONQ_BASIS, optimization_level=1, seed_transpiler=0)); tio.append(time.perf_counter() - t1)
        for (i, name), bi, io, sd, ts_ib, ts_io in zip(lab, ibm, ion, bseed, tib, tio):
            ob, oi = bi.count_ops(), io.count_ops()
            n2b = sum(v for k, v in ob.items() if k in TWO_Q); n1b = sum(v for k, v in ob.items() if k not in TWO_Q | NOT_GATE)
            n2i = sum(v for k, v in oi.items() if k in TWO_Q); n1i = sum(v for k, v in oi.items() if k not in TWO_Q | NOT_GATE)
            eg, er, miss = esp_ibm(bi, tg)
            w.writerow([ckpt.parent.name, config['text'].get('n'), L, nq, i] + ([name] if a.circ == 'all' else []) + [float(bi.estimate_duration(tg, unit='s')), n2b, n1b, eg, er, miss, n1i, n2i] + ([sd] if a.seeds > 1 else []) + [ts_ib, ts_io])
        print(f'{ckpt.parent.name}: nq {nq} layers {L}, {len(circs)} circuits, ESP gates {np.mean([esp_ibm(b, tg)[0] for b in ibm]):.3g} ({time.time() - t0:.0f}s)', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
