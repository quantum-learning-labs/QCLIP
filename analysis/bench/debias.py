# IonQ-style debiasing (symmetrization) on the Aer noisy backends, for a results dir written by backends.py.
# Each rep runs V random variants of every circuit with shots / V shots each and sums their counts:
#  - Pauli twirling of every 2q gate (coherent 2q errors -> stochastic), 1q Paulis synthesised into the target basis
#  - readout symmetrization: X before a random subset of measurements, the classical bits flipped back
# Same total shots and reps as the unmitigated rows. Writes summary_debias.json and raw_debias.pkl next to summary.json.
#   PYTHONPATH=.:analysis/bench python analysis/bench/debias.py results/bench_b1/<set> --variants 4
import argparse, copy, json, pickle, time
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.circuit import pauli_twirl_2q_gates
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit.circuit import Delay
from qiskit_aer.noise import RelaxationNoisePass
import qiskit_ibm_runtime.fake_provider as fake_provider

from modules.utils.general import setup_exp, resolve_eval_config
from modules.compilation.quantum.emul import BackendManager
import bench_lib as B
from backends import NOISY, gpu_energy_mj


def flip_readout(qc, mask):
    # X before the measurement of every clbit c with mask[c] = 1
    out = qc.copy_empty_like()
    for ins in qc.data:
        if ins.operation.name == 'measure' and mask[qc.find_bit(ins.clbits[0]).index]:
            out.x(ins.qubits[0])
        out.append(ins)
    return out


def unflip(counts, mask):
    flip = [i for i, m in enumerate(mask) if m]
    out = {}
    for k, c in counts.items():
        b = list(k)
        for i in flip:
            p = len(b) - 1 - i
            b[p] = '1' if b[p] == '0' else '0'
        k2 = ''.join(b)
        out[k2] = out.get(k2, 0) + c
    return out


def variants(qc, n, rng, target=None):
    tw = pauli_twirl_2q_gates(qc, seed=int(rng.integers(2**31)), num_twirls=n, target=target)
    tw = tw if isinstance(tw, list) else [tw]
    masks = [rng.integers(0, 2, qc.num_clbits).tolist() for _ in range(n)]
    return [(flip_readout(t, m), m) for t, m in zip(tw, masks)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--variants', type=int, default=4, help='random variants per circuit per rep')
    p.add_argument('--backends', default='ibm_miami,honest_miami,ibm_kingston')
    p.add_argument('--method', default='density_matrix')
    a = p.parse_args()
    d = Path(a.dir)
    out = d / 'summary_debias.json'
    if out.exists():
        raise FileExistsError(out)
    meta = json.load(open(d / 'summary.json'))['meta']
    raw = pickle.load(open(d / 'raw.pkl', 'rb'))
    circs = qpy.load(open(d / 'logical.qpy', 'rb'))
    infos, ex, n_pairs = raw['infos'], raw['exact'], meta['pairs']
    config, _, _ = setup_exp(resolve_eval_config(None, Path(meta['checkpoint'])), seed=0)
    config['emulate'].update(set=meta['eval_set'], measurement_method=meta['meas'])
    shots_v = meta['shots'] // a.variants
    rows, rawd = [], {}
    for name in a.backends.split(','):
        profile, cls = NOISY[name]
        cfg = copy.deepcopy(config)
        cfg['emulate'].update(noise=True, noise_profile=profile, backend=a.method,
                              aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False})
        nb = BackendManager(cfg, None, hardware_profile=getattr(fake_provider, cls)())
        idle = profile == 'honest' and nb.honest['idle']
        nb.honest['idle'] = False
        plain = nb.transpile(circs, batch_size=256, optimization_level=1)   # same seed -> same layout as the unmitigated run
        tgt = nb.hardware_profile.target
        if idle:
            qp = tgt.qubit_properties
            t1s = [q.t1 if q and q.t1 else float('inf') for q in qp]
            t2s = [min(q.t2, 1.99 * q.t1) if q and q.t1 and q.t2 else float('inf') for q in qp]
            sched = PassManager([ALAPScheduleAnalysis(target=tgt), PadDelay(target=tgt)])
            relax = PassManager([RelaxationNoisePass(t1s, t2s, dt=nb.hardware_profile.dt, op_types=[Delay])])
        rng = np.random.default_rng(12345)
        e0, t0 = gpu_energy_mj(), time.time()
        rep_counts = []
        for r in range(meta['reps']):
            nb.backend.set_options(seed_simulator=7000 * (r + 1))
            var, owner = [], []
            for j, qc in enumerate(plain):
                for v, m in variants(qc, a.variants, rng, target=tgt):
                    var.append((relax.run(sched.run(v)) if idle else v, m)); owner.append(j)
            cnt = [dict() for _ in plain]
            for i in range(0, len(var), 64):
                chunk = var[i:i + 64]
                res = nb.backend.run([c for c, _ in chunk], shots=shots_v).result()
                for k, (c, m) in enumerate(chunk):
                    for key, n in unflip(res.get_counts(k), m).items():
                        cnt[owner[i + k]][key] = cnt[owner[i + k]].get(key, 0) + n
            rep_counts.append(cnt)
            print(f'{name}_debias rep {r}: {time.time() - t0:.0f}s', flush=True)
        e1 = gpu_energy_mj()
        row, extra = B.backend_metrics(f'{name}_debias', rep_counts, infos, ex, meta['meas'], meta['discard'], n_pairs, meta['shots'])
        row.update(stats=B.summarise_stats(B.circ_stats(plain, target=tgt)), sim_wall_s=time.time() - t0,
                   sim_gpu_energy_J=(e1 - e0) / 1e3 if e0 is not None and e1 is not None else None,
                   profile=nb.hardware_profile.name, noise_profile=profile, mitigation=f'debias: 2q Pauli twirl + readout flips, {a.variants} variants x {shots_v} shots')
        print(name + '_debias', 'acc', row['acc_mean'], '+-', row['acc_std'], 'hellinger', row['hellinger_fid_mean'], flush=True)
        rows.append(row); rawd[name + '_debias'] = {'extra': extra, 'counts_rep0': rep_counts[0]}
        json.dump({'meta': meta, 'rows': rows}, open(out, 'w'), indent=1, default=str)
    pickle.dump(rawd, open(d / 'raw_debias.pkl', 'wb'))
    print('done', d, flush=True)


if __name__ == '__main__':
    main()
