# V2: 2-qubit gate counts per pair (image choice, test std) for the CU-pos, CU-neg and O-readout circuits of shotcost_noise.py:
# logical ({rz,sx,x,cx}, no coupling map) for all pairs, and Kingston-native on the qlimit coupling map (same transpile seed as E6)
# for the first --native pairs. One CSV row per pair: ckpt, n, idx, nq, cx_{cu_pos,cu_neg,O}, nat2q_{...}, natdepth_O.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/g_count.py -o results/shotcost/viz/gates.csv -cp ckpt...
import argparse, copy, csv, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import qiskit_ibm_runtime.fake_provider as fake_provider
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager
from shotcost_noise import o_unitary

TWO_Q = {'cx', 'cz', 'ecr', 'rzz', 'swap', 'iswap', 'rxx', 'ryy', 'cp', 'crz', 'csx'}
n2q = lambda qc: sum(v for k, v in qc.count_ops().items() if k in TWO_Q)


def frames(ckpt, seed, typical=None):
    # typical=k: compile only the first k sentences of the most common circuit width, read from the parse
    # (qubits = number of '0' wire starts, as in tn2qiskit), instead of all ~1.8k (same decisions as filtering after compiling)
    config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=seed)
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(text_model, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True))
    de.image_init(image_model)
    CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    config['emulate'].update(set='std', measurement_method='compute_uncompute', noise=False)
    c_eval = de.compile_text('std')
    qlimit = de.describe_einsum(text_model, c_eval, return_metrics=True)['max'][0]
    bm = BackendManager(config, qlimit); bm.transpile = lambda c, **kw: c
    cap, cur, ds = [], [None], de.get_dataset(c_eval, 'std')
    keep = list(range(len(ds)))
    if typical:
        w = [sum(g['op_type'] == '0' for g in s) if isinstance(s, list) else -1 for s in c_eval['corrected_sentence_symbols']]
        mode = Counter(x for x in w if x > 0).most_common(1)[0][0]
        keep = [i for i, x in enumerate(w) if x == mode][:typical]
        print(f'frames: width {mode} for {w.count(mode)}/{len(w)} sentences; compiling {len(keep)}', flush=True)
    class Track:
        def __len__(self): return len(keep)
        def __getitem__(self, j): cur[0] = keep[j]; return ds[keep[j]]
    def cu(qt, qi, oq):
        if len(oq) == 0: raise ValueError('empty output register')
        cap.append((cur[0], bm._strip_clbits(qt), bm._strip_clbits(qi), list(oq))); return QuantumCircuit(1)
    bm._compute_uncompute = cu
    bm.compile_circuits(Track(), text_model, image_model, None)
    by = {}
    for c in cap: by.setdefault(c[0], []).append(c[1:])
    return config, qlimit, [(i, by[i]) for i in sorted(by) if len(by[i]) == 2]


def build(qt, qip, qin, oq, o_max=6):
    # O-readout circuit = a general unitary on the output register: its synthesis grows ~4^e (killed the job at e=7),
    # so above o_max output qubits it is skipped (None) and only the CU circuits are built
    n = qt.num_qubits; out = []
    W = o_unitary(Statevector(qip).data, Statevector(qin).data)[0] if len(oq) <= o_max else None
    for img in (qip, qin, None):
        if img is None and W is None: out.append(None); continue
        qc = QuantumCircuit(n, n); qc.compose(qt, inplace=True)
        if img is not None: qc.compose(img.inverse(), qubits=oq, inplace=True)
        else: qc.unitary(W, oq)
        qc.measure(range(n), range(n)); out.append(qc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--native', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--o_max', type=int, default=6, help='skip the O-readout circuit above this many output qubits')
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(['ckpt', 'text_n', 'idx', 'nq', 'cx_cu_pos', 'cx_cu_neg', 'cx_O', 'nat2q_cu_pos', 'nat2q_cu_neg', 'nat2q_O', 'natdepth_O'])
        for c in a.checkpoints:
            t0 = time.time(); ckpt = Path(c)
            config, qlimit, fr = frames(ckpt, a.seed)
            circ = [build(qt, qip, qin, oq, a.o_max) for _, ((qt, qip, oq), (_, qin, _)) in fr]
            has_o = all(cs[2] is not None for cs in circ)
            flat = [x if x is not None else cs[0] for cs in circ for x in cs]   # placeholder (CU-pos) where O was skipped
            logical = transpile(flat, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=a.seed, num_processes=4)
            cfg = copy.deepcopy(config); cfg['emulate'].update(noise=False)
            nb = BackendManager(cfg, qlimit, hardware_profile=fake_provider.FakeKingston())
            nat = nb.transpile(flat[:3 * a.native], batch_size=256, optimization_level=1)
            for r, (i, _) in enumerate(fr):
                L = [n2q(logical[3 * r + j]) for j in range(3)]
                N = [n2q(nat[3 * r + j]) for j in range(3)] if r < a.native else ['', '', '']
                d = nat[3 * r + 2].depth(lambda ins: ins.operation.name not in ('barrier', 'delay')) if r < a.native else ''
                if not has_o: L[2] = ''; N[2] = ''; d = ''
                w.writerow([ckpt.parent.name, config['text'].get('n'), i, flat[3 * r].num_qubits, *L, *N, d])
            fh.flush()
            print(ckpt.parent.name, f'{len(fr)} pairs, qlimit {qlimit}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
