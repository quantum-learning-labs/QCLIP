# IQM cost of one decision circuit (structure only, CPU): transpile the same circuits as qpu_esp.py (cu_pos, cu_neg, O) onto an IQM
# square-lattice fake backend (iqm-client: Aphrodite, 54 qubits; native prx + cz) with --seeds layout seeds, keeping per circuit the
# transpile with the fewest CZ gates. Circuits wider than the device are not transpiled (row with fits=0): the figure prices them as
# logical two-qubit gates x the median routing overhead measured on the circuits that fit (same rule as for IBM Kingston).
# Duration: critical path with the device's gate durations (prx, cz) from the fake backend's error profile; readout and reset are
# priced in the figure from published per-shot figures. No QPU or cloud calls.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/qpu_iqm.py -cp ckpt... -o out.csv [--k 3] [--seeds 5]
import argparse, csv, time
from pathlib import Path
from qiskit import transpile
import iqm.qiskit_iqm.fake_backends as fb
from g_count import build
from inf_engines import load

TWO_Q = {'cz', 'cx', 'ecr', 'rzz', 'rxx', 'move'}
NOT_GATE = {'measure', 'barrier', 'delay', 'reset'}


def crit_path(qc, dur):
    # longest gate path in seconds (gates not in dur cost 0: barriers, measures; readout is priced separately)
    t = [0.0] * qc.num_qubits
    for ci in qc.data:
        qs = [qc.find_bit(q).index for q in ci.qubits]
        s = max(t[q] for q in qs) + dur.get(ci.operation.name, 0.0)
        for q in qs: t[q] = s
    return max(t) if t else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--o_max', type=int, default=6)
    ap.add_argument('--device', default='IQMFakeAphrodite')
    a = ap.parse_args()
    dev = getattr(fb, a.device)()
    ep = dev.error_profile
    dur = {**{k: v * 1e-9 for k, v in ep.single_qubit_gate_durations.items()}, **{k: v * 1e-9 for k, v in ep.two_qubit_gate_durations.items()}}
    out = Path(a.out); new = not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'a', newline='', buffering=1))
    if new: w.writerow(['ckpt', 'text_n', 'layers', 'nq', 'idx', 'circ', 'device', 'device_qubits', 'fits', 'logical_n2q', 'iqm_n2q', 'iqm_n1q', 'iqm_crit_s', 'iqm_seed', 'iqm_transpile_s'])
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers'); t0 = time.time()
        for j, (qt, qip, qin, oq) in enumerate(D[:a.k]):
            for name, qc in zip(('cu_pos', 'cu_neg', 'O'), build(qt, qip, qin, oq, a.o_max)):
                if qc is None: continue
                lg = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cz'], optimization_level=1, seed_transpiler=0)   # all-to-all: logical two-qubit count
                ln2 = sum(v for k, v in lg.count_ops().items() if k in TWO_Q)
                if qc.num_qubits > dev.num_qubits:
                    w.writerow([ckpt.parent.name, config['text'].get('n'), L, nq, j, name, a.device, dev.num_qubits, 0, ln2, '', '', '', '', '']); continue
                best, bs, tt = None, None, 0.0
                for sd in range(a.seeds):
                    t1 = time.perf_counter(); q = transpile(qc, backend=dev, optimization_level=1, seed_transpiler=sd); tt += (time.perf_counter() - t1) / a.seeds
                    n2 = sum(v for k, v in q.count_ops().items() if k in TWO_Q)
                    if best is None or n2 < best[0]: best, bs = (n2, q), sd
                n2, q = best; ops = q.count_ops()
                n1 = sum(v for k, v in ops.items() if k not in TWO_Q | NOT_GATE)
                w.writerow([ckpt.parent.name, config['text'].get('n'), L, nq, j, name, a.device, dev.num_qubits, 1, ln2, n2, n1, crit_path(q, dur), bs, tt])
        print(f'{ckpt.parent.name}: nq {nq} layers {L} ({time.time() - t0:.0f}s)', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
