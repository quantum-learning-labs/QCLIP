# V3: GPU energy per exact decision vs qubits (NVML), plus the device statistics the QPU energy model needs.
#  (a) our models (one ckpt per size): Aer GPU statevector of the O-readout circuit (one per decision, exact margin) for --pairs pairs,
#      batched; NVML energy minus idle power x time, per decision.
#  (b) scaling probe: brickwork circuits (cx + random 1q) at 10..--nmax qubits, --layers 2q layers, energy/time per circuit.
#  (c) QPU stats on the same pairs (CPU transpile): Kingston duration (s) and native 2q count; IonQ native n1q/n2q (all-to-all).
# Writes CSV rows to <out>/{gpu_models,gpu_scaling,qpu_stats}.csv.  Run alone on the GPU (shared runs pollute NVML energy).
import argparse, csv, os, time
from pathlib import Path
import numpy as np, pynvml
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as fake_provider
from g_count import frames, build

IONQ_BASIS = ['rz', 'ry', 'rx', 'rxx']
TWO_Q = {'cx', 'cz', 'ecr', 'rzz', 'rxx'}


class Meter:
    def __init__(self):
        # NVML ignores CUDA_VISIBLE_DEVICES: on a multi-GPU node index 0 is physical card 0, not the card the job was given.
        # Pick the NVML device whose UUID matches CUDA device 0 (fallback: the CUDA_VISIBLE_DEVICES entry, then index 0).
        pynvml.nvmlInit(); self.h, how = None, ''
        uuids = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i); u = pynvml.nvmlDeviceGetUUID(h); uuids.append((h, u if isinstance(u, str) else u.decode()))
        try:
            import torch
            cu = str(torch.cuda.get_device_properties(0).uuid)
            for h, u in uuids:
                if cu.replace('GPU-', '') in u: self.h, how = h, 'uuid'
        except Exception:
            pass
        if self.h is None:
            cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
            if cvd.startswith('GPU-'): self.h, how = [h for h, u in uuids if u.startswith(cvd)][0], 'cvd-uuid'
            elif cvd.isdigit() and len(uuids) > int(cvd): self.h, how = uuids[int(cvd)][0], 'cvd-index'
            else: self.h, how = uuids[0][0], 'index0'
        self.name = pynvml.nvmlDeviceGetName(self.h)
        print(f'NVML: {len(uuids)} devices, using {pynvml.nvmlDeviceGetUUID(self.h)} ({how}); CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    def mj(self): return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
    def idle_W(self, s=15):
        e0, t0 = self.mj(), time.time(); time.sleep(s); return (self.mj() - e0) / 1e3 / (time.time() - t0)
    def run(self, fn, min_s=10.0):
        # repeat until >= min_s (the NVML energy counter updates too coarsely for sub-second runs); per-repetition J and s
        e0, t0, k = self.mj(), time.time(), 0
        while True:
            fn(); k += 1
            if time.time() - t0 >= min_s: break
        return (self.mj() - e0) / 1e3 / k, (time.time() - t0) / k


def qpu_stats(a, out):
    kingston = fake_provider.FakeKingston()
    with open(out / 'qpu_stats.csv', 'w', newline='') as f3:
        w3 = csv.writer(f3)
        w3.writerow(['ckpt', 'text_n', 'idx', 'nq', 'kind', 'ibm_duration_s', 'ibm_n2q', 'ibm_n1q', 'ionq_n1q', 'ionq_n2q'])
        for c in a.checkpoints:
            ckpt = Path(c)
            config, qlimit, fr = frames(ckpt, 0)
            # QPU statistics (CPU): CU-pos and O circuits of the first qpu_pairs pairs
            sample = []
            for i, ((qt, qip, oq), (_, qin, _)) in fr[:a.qpu_pairs]:
                cs = build(qt, qip, qin, oq, a.o_max)
                sample += [(i, 'cu', cs[0])] + ([(i, 'O', cs[2])] if cs[2] is not None else [])
            ibm = transpile([x for _, _, x in sample], backend=kingston, optimization_level=1, seed_transpiler=0)
            ion = transpile([x for _, _, x in sample], basis_gates=IONQ_BASIS, optimization_level=1, seed_transpiler=0)
            for (i, kind, x), bi, io in zip(sample, ibm, ion):
                ob, oi = bi.count_ops(), io.count_ops()
                n2b = sum(v for k, v in ob.items() if k in TWO_Q); n1b = sum(v for k, v in ob.items() if k not in TWO_Q | {'measure', 'barrier', 'delay'})
                n2i = sum(v for k, v in oi.items() if k in TWO_Q); n1i = sum(v for k, v in oi.items() if k not in TWO_Q | {'measure', 'barrier'})
                w3.writerow([ckpt.parent.name, config['text'].get('n'), i, x.num_qubits, kind, float(bi.estimate_duration(kingston.target, unit='s')),
                             n2b, n1b, n1i, n2i])
            f3.flush()
            print(ckpt.parent.name, 'qpu stats done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--pairs', type=int, default=300)
    ap.add_argument('--qpu_pairs', type=int, default=100)
    ap.add_argument('--nmax', type=int, default=30)
    ap.add_argument('--layers', type=int, default=20)
    ap.add_argument('--qpu_only', action='store_true', help='only the CPU part (c); runs without a GPU')
    ap.add_argument('--scaling_only', action='store_true', help='only the scaling probe (b)')
    ap.add_argument('--no_scaling', action='store_true', help='skip the scaling probe (b)')
    ap.add_argument('--nq_mode', action='store_true', help='keep only decisions whose circuit has the most common qubit count (typical sentences): comparable across sizes')
    ap.add_argument('--readout', default='state', choices=['state', 'probs'], help='probs: save only the output-register probabilities (no 2^n host copy of the state)')
    ap.add_argument('--precision', default='double', choices=['double', 'single'])
    ap.add_argument('--nq_cap', type=int, default=0, help='skip decisions whose circuit has more qubits than this (statevector memory)')
    ap.add_argument('--o_max', type=int, default=6, help='O-readout circuits only up to this many output qubits')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.qpu_only or not (out / 'qpu_stats.csv').exists():
        qpu_stats(a, out)
    if a.qpu_only:
        return
    m = Meter(); idle = m.idle_W()
    print(f'GPU {m.name}, idle {idle:.1f} W', flush=True)
    if idle > 100: raise SystemExit(f'GPU is not idle ({idle:.0f} W): another process is on it, energy would be polluted; rerun on another node')
    sim = AerSimulator(method='statevector', device='GPU', precision=a.precision)
    if a.scaling_only:
        a.checkpoints = []
    with open(out / 'gpu_models.csv' if not a.scaling_only else out / 'gpu_models_unused.csv', 'w', newline='') as f1:
        w1 = csv.writer(f1)
        w1.writerow(['gpu', 'ckpt', 'text_n', 'pairs', 'nq_mean', 'nq_max', 'J_gross', 'J_net', 'seconds', 'J_net_per_decision', 'J_gross_per_decision', 'idle_W', 'precision', 'readout'])
        for c in a.checkpoints:
            ckpt = Path(c)
            config, qlimit, fr = frames(ckpt, 0)
            fr = [x for x in fr if not a.nq_cap or max(x[1][0][0].num_qubits, x[1][1][0].num_qubits) <= a.nq_cap]
            if a.nq_mode:
                from collections import Counter
                mode = Counter(x[1][0][0].num_qubits for x in fr).most_common(1)[0][0]
                fr = [x for x in fr if x[1][0][0].num_qubits == mode and x[1][1][0].num_qubits == mode]
                print(f'{ckpt.parent.name}: typical circuits with {mode} qubits: {len(fr)} decisions', flush=True)
            fr = fr[:a.pairs]
            circs = []
            for _, ((qt, qip, oq), (_, qin, _)) in fr:
                cs = build(qt, qip, qin, oq, a.o_max)
                o = (cs[2] if cs[2] is not None else cs[0]).remove_final_measurements(inplace=False)   # CU-pos above o_max
                (o.save_probabilities(list(oq)) if a.readout == 'probs' else o.save_statevector()); circs.append(o)
            tc = transpile(circs, sim, optimization_level=1, seed_transpiler=0)
            sim.run(tc[:4]).result()                       # warm-up (JIT / context)
            J, s = m.run(lambda: sim.run(tc).result())
            nq = [x.num_qubits for x in circs]
            w1.writerow([m.name, ckpt.parent.name, config['text'].get('n'), len(circs), np.mean(nq), max(nq), J, J - idle * s, s,
                         (J - idle * s) / len(circs), J / len(circs), idle, a.precision, a.readout]); f1.flush()
            print(ckpt.parent.name, f'GPU {J:.1f} J gross over {s:.1f} s for {len(circs)} decisions', flush=True)
    if a.no_scaling:
        return
    with open(out / 'gpu_scaling.csv', 'w', newline='') as f2:
        w2 = csv.writer(f2); w2.writerow(['gpu', 'n', 'layers', 'circuits', 'J_gross', 'J_net', 'seconds', 'J_net_per_circuit', 'idle_W'])
        rng = np.random.default_rng(0)
        for n in range(10, a.nmax + 1, 2):
            cs = []
            for r in range(3):
                qc = QuantumCircuit(n)
                for L in range(a.layers):
                    for q in range(n): qc.ry(rng.uniform(0, 2 * np.pi), q); qc.rz(rng.uniform(0, 2 * np.pi), q)
                    for q in range(L % 2, n - 1, 2): qc.cx(q, q + 1)
                qc.save_expectation_value(Pauli('Z' * n), list(range(n))); cs.append(qc)   # all-qubit observable: no light-cone truncation; tiny result
            tcs = transpile(cs, sim, optimization_level=0)
            try:
                sim.run(tcs[:1]).result()
                J, s = m.run(lambda: sim.run(tcs).result())
            except Exception as e:
                print(f'n={n}: {e}', flush=True); break
            w2.writerow([m.name, n, a.layers, len(cs), J, J - idle * s, s, (J - idle * s) / len(cs), idle]); f2.flush()
            print(f'scaling n={n}: {J / len(cs):.2f} J gross per circuit, {s / len(cs):.3f} s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
